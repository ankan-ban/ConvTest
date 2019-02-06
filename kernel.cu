#include "utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cudnn.h>

#include <cstdio>
#include <cstdlib>

constexpr int loops = 10000;


#define INDEX_NCHW(n,c,h,w) ((n)*C*H*W + (c)*H*W + (h)*W + w)
#define FILTER_IDX_NCHW(k,c,h,w) ((k)*C*S*R + (c)*S*R + (h)*R + w)

// cpu implementation of convolution
// (convolution with zero padding)
// N - mini-batch size
// K - num of output channels
// C - num of input channels
// H - height
// W - width
// S - height of filter
// R - width of filter
template<int N, int K, int C, int H, int W, int S, int R, typename T>
void convRef(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                {
                    float op = 0.0f;
                    for (int c = 0; c < C; c++)
                    {
                        for (int s = 0; s < S; s++)
                        {
                            for (int r = 0; r < R; r++)
                            {
                                float filter = (float)(weight[FILTER_IDX_NCHW(k, c, s, r)]);
                                int y = h + s - S / 2;
                                int x = w + r - R / 2;
                                float ip = 0;
                                if (y >= 0 && y < H && x >= 0 && x < W)
                                    ip = (float) (input[INDEX_NCHW(n, c, y, x)]);
                                op += ip * filter;
                            }   // r
                        }   // s
                    }   // c

                    if (bias)
                        op += (float) (bias[k]);

                    if (relu && op < 0)
                        op = 0;

                    output[INDEX_NCHW(n, k, h, w)] = (T) op;
                }   // w
            } // h
        } // k
    } // n
}



// based on example here:
// https://www.intel.ai/winograd-2/#gs.xDwcjzbC
// and paper here:
// https://arxiv.org/pdf/1509.09308.pdf


template<int M, int N, int K, typename T>
void matrixMulCPU(T *c, T *a, T *b)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            float S = 0;
            for (int k = 0; k < K; ++k)
                S += (float)(a[i*K + k]) * (float)(b[k*N + j]);
            c[i*N + j] = (T)S;
        }
}

template<typename T>
void outputTransform2x2(T *output, T *transformedOutput)
{
    // transform applied to result
    T At[2 * 4] = {
        1, 1, 1, 0,
        0, 1, -1, -1
    };

    T A[4 * 2] = {
        1,  0,
        1,  1,
        1, -1,
        0, -1
    };

    T tempOp[2 * 4];
    matrixMulCPU<2, 4, 4, T>(tempOp, At, transformedOutput);
    matrixMulCPU<2, 2, 4, T>(output, tempOp, A);
}

// input is padded input needed to compute 2x2 output tile, 
template<typename T>
void processWinogradTile2x2(T *output, T *input, T *filter)
{
    T transformedInput[4 * 4];
    T transformedFilter[4 * 4];

    // transform applied to input tile (of size 4x4)
    T Bt[4 * 4] = {
        1,  0, -1,  0,
        0,  1,  1,  0,
        0, -1,  1,  0,
        0,  1,  0, -1
    };

    T B[4 * 4] =
    {
        1, 0, 0, 0,
        0, 1,-1, 1,
       -1, 1, 1, 0,
        0, 0, 0,-1
    };

    // transform applied to filter (of size 3x3)
    T G[4 * 3] = {
        1,    0,    0,
        0.5,  0.5,  0.5,
        0.5, -0.5,  0.5,
        0,    0,    1
    };

    T Gt[3 * 4] = {
        1,  0.5,  0.5, 0, 
        0,  0.5, -0.5, 0,
        0,  0.5,  0.5, 1
    };

    // 1. transform the input
    T tempIp1[4 * 4];
    matrixMulCPU<4, 4, 4, T>(tempIp1, Bt, input);
    matrixMulCPU<4, 4, 4, T>(transformedInput, tempIp1, B);

    // 2. transform the filter
    T tempFilter[4 * 3];
    matrixMulCPU<4, 3, 3, T>(tempFilter, G, filter);
    matrixMulCPU<4, 4, 3, T>(transformedFilter, tempFilter, Gt);


    // 3. element wise product of transformed filter and transformed input
    T transformedOutput[4 * 4];
    for (int i = 0; i < 4 * 4; i++)
        transformedOutput[i] = transformedFilter[i] * transformedInput[i];


    // 4. transform back the output into pixel space
    outputTransform2x2(output, transformedOutput);

    // Note: so many computations! where are the savings?
    // 1. input transform can be done only once per convolution (1/K times its read)
    // 2. filter transform is free at time of inference (just once at program start)
    // 3. output transform can be done after the addition along C dimension? (1/C times it's updated)

    // the transform has effectively reduced 4 * 9 = 36 multiplications (+8*4 adds) to just 16 multiplications (+extra 12 adds)
}

// same as above function but doesn't transform output (it's transformed after summing it up in C dimension)
template<typename T>
void processWinogradTile2x2NoOutTransform(T *transformedOutput, T *input, T *filter)
{
    T transformedInput[4 * 4];
    T transformedFilter[4 * 4];

    // transform applied to input tile (of size 4x4)
    T Bt[4 * 4] = {
        1,  0, -1,  0,
        0,  1,  1,  0,
        0, -1,  1,  0,
        0,  1,  0, -1
    };

    T B[4 * 4] =
    {
        1, 0, 0, 0,
        0, 1,-1, 1,
        -1, 1, 1, 0,
        0, 0, 0,-1
    };

    // transform applied to filter (of size 3x3)
    T G[4 * 3] = {
        1,    0,    0,
        0.5,  0.5,  0.5,
        0.5, -0.5,  0.5,
        0,    0,    1
    };

    T Gt[3 * 4] = {
        1,  0.5,  0.5, 0,
        0,  0.5, -0.5, 0,
        0,  0.5,  0.5, 1
    };

    // 1. transform the input
    T tempIp1[4 * 4];
    matrixMulCPU<4, 4, 4, T>(tempIp1, Bt, input);
    matrixMulCPU<4, 4, 4, T>(transformedInput, tempIp1, B);

    // 2. transform the filter
    T tempFilter[4 * 3];
    matrixMulCPU<4, 3, 3, T>(tempFilter, G, filter);
    matrixMulCPU<4, 4, 3, T>(transformedFilter, tempFilter, Gt);


    // 3. element wise product of transformed filter and transformed input
    for (int i = 0; i < 4 * 4; i++)
        transformedOutput[i] = transformedFilter[i] * transformedInput[i];
}



// assumes S == R == 3
// AND H, W to be multiple of 2
template<int N, int K, int C, int H, int W, int S, int R, typename T>
void convRef_Winograd2x2(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            for (int h = 0; h < H; h+=2)
            {
                for (int w = 0; w < W; w+=2)    // process 2x2 output tile a time
                {
                    T op[2][2];
                    T transformedOutputAccum[4][4];

                    for (int i = 0; i < 4; i++)
                        for (int j = 0; j < 4; j++)
                            transformedOutputAccum[i][j] = 0;

                    for (int c = 0; c < C; c++)
                    {

                        T transformedOpTile[4][4];

                        T inputTile[4][4];          // window of input tile needed to compute the 2x2 output tile

                        for (int i = 0; i < 4; i++)
                            for (int j = 0; j < 4; j++)
                            {
                                int y = h + i - 1;
                                int x = w + j - 1;

                                if (y >= 0 && y < H && x >= 0 && x < W)
                                    inputTile[i][j] = input[INDEX_NCHW(n, c, y, x)];
                                else
                                    inputTile[i][j] = 0;
                            }


                        T filterTile[3][3];
                        for (int s = 0; s < S; s++)
                            for (int r = 0; r < R; r++)
                            {
                                filterTile[s][r] = weight[FILTER_IDX_NCHW(k, c, s, r)];
                            }

                        processWinogradTile2x2NoOutTransform(&(transformedOpTile[0][0]), &(inputTile[0][0]), &(filterTile[0][0]));

                        // accumulate in transformed op space
                        for (int i = 0; i < 4; i++)
                            for (int j = 0; j < 4; j++)
                            {
                                transformedOutputAccum[i][j] += transformedOpTile[i][j];
                            }
                    }   // c

                    // transform output just once
                    outputTransform2x2(&(op[0][0]), &(transformedOutputAccum[0][0]));

                    // relu/bias and write to output
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 2; j++)
                        {
                            if (bias)
                                op[i][j] += (float)(bias[k]);

                            //if (relu && op[i][j] < 0)
                            //    op[i][j] = 0;

                            output[INDEX_NCHW(n, k, h+i, w+j)] = op[i][j];
                        }
                }   // w
            } // h
        } // k
    } // n
}



template<typename T>
void outputTransform4x4(T *output, T *transformedOutput)
{
    // transform applied to result
    T At[4 * 6] = {
        1, 1, 1, 1, 1, 0,
        0, 1,-1, 2,-2, 0,
        0, 1, 1, 4, 4, 0,
        0, 1,-1, 8,-8, 1
    };

    T A[6 * 4] = {
        1, 0, 0, 0,
        1, 1, 1, 1, 
        1,-1, 1,-1,
        1, 2, 4, 8,
        1,-2, 4,-8,
        0, 0, 0, 1
    };

    T tempOp[4 * 6];
    matrixMulCPU<4, 6, 6, T>(tempOp, At, transformedOutput);
    matrixMulCPU<4, 4, 6, T>(output, tempOp, A);
}

template<typename T>
void filterTransform4x4(T *transformedFilter, T *filter)
{
    // transform applied to filter (of size 3x3)
    T G[6 * 3] = {
        1.0/4 ,      0 ,      0 ,
       -1.0/6 , -1.0/6 , -1.0/6 ,
       -1.0/6 ,  1.0/6 , -1.0/6 ,
        1.0/24,  1.0/12,  1.0/6 ,
        1.0/24, -1.0/12,  1.0/6 ,
        0     ,      0 ,      1
    };

    T Gt[3 * 6] = {
        1.0/4,    -1.0/6,     -1.0/6,     1.0/24,     1.0/24,   0,
        0,        -1.0/6,      1.0/6,     1.0/12,    -1.0/12,   0,
        0,        -1.0/6,     -1.0/6,     1.0/6,      1.0/6,    1
    };

    T tempFilter[6 * 3];
    matrixMulCPU<6, 3, 3, T>(tempFilter, G, filter);
    matrixMulCPU<6, 6, 3, T>(transformedFilter, tempFilter, Gt);
}

template<typename T>
void inputTransform4x4(T *transformedInput, T *input)
{
    // transform applied to input tile (of size 4x4)
    T Bt[6 * 6] = {
        4,  0, -5,  0, 1, 0,
        0, -4, -4,  1, 1, 0,
        0,  4, -4, -1, 1, 0,
        0, -2, -1,  2, 1, 0,
        0,  2, -1, -2, 1, 0,
        0,  4,  0, -5, 0, 1
    };

    T B[6 * 6] =
    {
        4,  0,  0,  0,  0,  0,
        0, -4,  4, -2,  2,  4,
       -5, -4, -4, -1, -1,  0,
        0,  1, -1,  2, -2, -5,
        1,  1,  1,  1,  1,  0,
        0,  0,  0,  0,  0,  1
    };

    T tempIp1[6 * 6];
    matrixMulCPU<6, 6, 6, T>(tempIp1, Bt, input);
    matrixMulCPU<6, 6, 6, T>(transformedInput, tempIp1, B);
}

template<typename T>
void processWinogradTile4x4NoOutTransform(T *transformedOutput, T *input, T *filter)
{
    T transformedInput[6 * 6];
    T transformedFilter[6 * 6];


    // 1. transform the input
    inputTransform4x4(transformedInput, input);

    // 2. transform the filter
    filterTransform4x4(transformedFilter, filter);


    // 3. element wise product of transformed filter and transformed input
    for (int i = 0; i < 6 * 6; i++)
        transformedOutput[i] = transformedFilter[i] * transformedInput[i];


    // Savings:
    // 1. input transform can be done only once per convolution (1/K times its read)
    // 2. filter transform is free at time of inference (just do once at program start)
    // 3. output transform can be done after the addition along C dimension? (1/C times it's updated)

    // the transform has effectively reduced 16 * 9 = 144 multiplications (+8*16 adds) to just 36 multiplications (+extra 20 adds)
    //  => 4x theoretical speedup (or more)
}


// assumes S == R == 3
// AND H, W to be multiple of 4
template<int N, int K, int C, int H, int W, int S, int R, typename T>
void convRef_Winograd4x4(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            for (int h = 0; h < H; h += 4)
            {
                for (int w = 0; w < W; w += 4)    // process 4x4 output tile a time
                {
                    T op[4][4];
                    T transformedOutputAccum[6][6];

                    for (int i = 0; i < 6; i++)
                        for (int j = 0; j < 6; j++)
                            transformedOutputAccum[i][j] = 0;

                    for (int c = 0; c < C; c++)
                    {

                        T transformedOpTile[6][6];

                        T inputTile[6][6];          // window of input tile needed to compute the 4x4 output tile

                        for (int i = 0; i < 6; i++)
                            for (int j = 0; j < 6; j++)
                            {
                                int y = h + i - 1;
                                int x = w + j - 1;

                                if (y >= 0 && y < H && x >= 0 && x < W)
                                    inputTile[i][j] = input[INDEX_NCHW(n, c, y, x)];
                                else
                                    inputTile[i][j] = 0;
                            }


                        T filterTile[3][3];
                        for (int s = 0; s < S; s++)
                            for (int r = 0; r < R; r++)
                            {
                                filterTile[s][r] = weight[FILTER_IDX_NCHW(k, c, s, r)];
                            }

                        processWinogradTile4x4NoOutTransform(&(transformedOpTile[0][0]), &(inputTile[0][0]), &(filterTile[0][0]));

                        // accumulate in transformed op space
                        for (int i = 0; i < 6; i++)
                            for (int j = 0; j < 6; j++)
                            {
                                transformedOutputAccum[i][j] += transformedOpTile[i][j];
                            }
                    }   // c

                        // transform output just once
                    outputTransform4x4(&(op[0][0]), &(transformedOutputAccum[0][0]));

                    // relu/bias and write to output
                    for (int i = 0; i < 4; i++)
                        for (int j = 0; j < 4; j++)
                        {
                            if (bias)
                                op[i][j] += (float)(bias[k]);

                            if (relu && op[i][j] < 0)
                                op[i][j] = 0;

                            output[INDEX_NCHW(n, k, h + i, w + j)] = op[i][j];
                        }
                }   // w
            } // h
        } // k
    } // n
}

// same as above but try:
// 1. transform the entire filter tensor at one go (into HWCK layout)
//      -- e.g: 6x6x256x256
// 2. transform the entire input tensor at one go (into HWNC layout)
//      -- e.g: 12x12x100x256
// 3. perform batched matrix multiplications (12x12 of them) to get entire output tensor in transformed space
//      -- again in HWNK layout, e.g: 12x12x100x256
// 4. transform the entire output tensor in one go (into required output layout - i.e, NCHW)
template<int N, int K, int C, int H, int W, int S, int R, typename T>
void convTest_Winograd4x4Matmul(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    T *transformedFilter = new T[6 * 6 * C * K];

    // 2x2 = 4 tiles of size 4x4 per board (8x8 = 64 elements), transformed tiles are 6x6
    T *transformedInput = new T[12 * 12 * N * C];
    T *transformedOutput = new T[12 * 12 * N * K];	

    // 1. transform the filter(s)
    for (int k = 0; k < K; k++)
    {
        for (int c = 0; c < C; c++)
        {
            // 1. read single filter from memory
            T filterTile[3][3];
            for (int s = 0; s < S; s++)
                for (int r = 0; r < R; r++)
                {
                    filterTile[s][r] = weight[FILTER_IDX_NCHW(k, c, s, r)];
                }

            // 2. transform it
            T transformedFilterTile[6][6];
            filterTransform4x4(&(transformedFilterTile[0][0]), &(filterTile[0][0]));

            // 3. write it back to memory (in HWCK layout)
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 6; j++)
                {
                    transformedFilter[i*6*C*K + j*C*K + c*K + k] = transformedFilterTile[i][j];
                }

        }
    }

    // 2. transform the input
    for (int n = 0; n < N; n++)
    {
        for (int h = 0; h < H; h += 4)
        {
            for (int w = 0; w < W; w += 4)    // process 2x2 output tile a time
            {
                for (int c = 0; c < C; c++)
                {
                    T inputTile[6][6];          // window of input tile needed to compute the 4x4 output tile
                    // 1. read input tile from memory
                    for (int i = 0; i < 6; i++)
                        for (int j = 0; j < 6; j++)
                        {
                            int y = h + i - 1;
                            int x = w + j - 1;

                            if (y >= 0 && y < H && x >= 0 && x < W)
                                inputTile[i][j] = input[INDEX_NCHW(n, c, y, x)];
                            else
                                inputTile[i][j] = 0;
                        }

                    // 2. transform it
                    T transformedIpTile[6][6];
                    inputTransform4x4(&(transformedIpTile[0][0]), &(inputTile[0][0]));

                    // 3. write it back to memory (in HWNC layout)
                    for (int i = 0; i < 6; i++)
                        for (int j = 0; j < 6; j++)
                        {
                            int y = i + 6 * (h / 4);
                            int x = j + 6 * (w / 4);
                            transformedInput[y * 12 * N*C + x*N*C + n*C + c] = transformedIpTile[i][j];
                        }
                }   // c
            }   // w
        } // h
    } // n


    // 3. Batch of matrix multiplies to get transformed output (in HWNK layout).
    for (int h=0;h<12;h++)
        for (int w = 0; w < 12; w++)
        {
            int fh = h % 6;
            int fw = w % 6;
            matrixMulCPU<N, K, C, T>(&(transformedOutput[h*12*N*K + w*N*K]), 
                                     &(transformedInput[h*12*N*C + w*N*C]), 
                                     &(transformedFilter[fh*6*C*K + fw*C*K]));
        }

    // 4. transform the result back 
    for (int n = 0; n < N; n++)
    {
        for (int h = 0; h < H; h += 4)
        {
            for (int w = 0; w < W; w += 4)
            {
                for (int k = 0; k < K; k++)
                {
                    // 1. read from memory
                    T transformedOpTile[6][6];
                    for (int i = 0; i < 6; i++)
                        for (int j = 0; j < 6; j++)
                        {
                            int y = i + 6 * (h / 4);
                            int x = j + 6 * (w / 4);
                            transformedOpTile[i][j] = transformedOutput[y *12*N*K + x*N*K + n*K + k];
                        }

                    // 2. transform it
                    T opTile[4][4];
                    outputTransform4x4(&(opTile[0][0]), &(transformedOpTile[0][0]));

                    // 3. write it back to memory (in standard NHWC layout)
                    for (int i = 0; i < 4; i++)
                        for (int j = 0; j < 4; j++)
                        {
                            // apply bias and relu just before writing!
                            if (bias)
                                opTile[i][j] += (float)(bias[k]);

                            if (relu && opTile[i][j] < 0)
                                opTile[i][j] = 0;

                            output[INDEX_NCHW(n, k, h + i, w + j)] = opTile[i][j];
                        }
                }   // k
            }   // w
        } // h
    } // n



    delete[]transformedFilter;
    delete[]transformedInput;
    delete[]transformedOutput;
}




template<int N, int K, int C, int H, int W, int S, int R, typename T>
void cudnnConvTest(T *output, T *input, T *filter, T *bias, bool relu)
{
    bool fp16 = (sizeof(T) == sizeof(half));
    const cudnnDataType_t datatype = fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
    const cudnnTensorFormat_t layout = fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t inputTensor, filterTensor, outputTensor, biasDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t convAlgo;
    cudnnActivationDescriptor_t actDesc;

    cudnnStatus_t status;
    status = cudnnCreate(&cudnnHandle);

    status = cudnnCreateTensorDescriptor(&inputTensor);
    status = cudnnCreateTensorDescriptor(&outputTensor);
    status = cudnnCreateFilterDescriptor(&filterDesc);
    status = cudnnCreateTensorDescriptor(&biasDesc);
    status = cudnnCreateConvolutionDescriptor(&convDesc);
    status = cudnnCreateActivationDescriptor(&actDesc);

    status = cudnnSetTensor4dDescriptor(inputTensor,
        layout,
        datatype,
        N, C,
        H, W);

    status = cudnnSetFilter4dDescriptor(filterDesc,
        datatype,
        layout,
        K,
        C,
        S,
        R);

    status = cudnnSetTensor4dDescriptor(biasDesc, layout, datatype, 1, K, 1, 1);

    status = cudnnSetActivationDescriptor(actDesc, relu ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0);

    status = cudnnSetConvolution2dDescriptor(convDesc,
        S/2, R/2,
        1, 1,
        1, 1,
        CUDNN_CROSS_CORRELATION /* CUDNN_CONVOLUTION*/,
        datatype);


    int n, c, h, w;
    status = cudnnGetConvolution2dForwardOutputDim(convDesc,
        inputTensor,
        filterDesc,
        &n, &c, &h, &w);


    status = cudnnSetTensor4dDescriptor(outputTensor,
        layout,
        datatype,
        n, c,
        h, w);

    convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    if (fp16)
    {
        status = cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);
    }

    void *workspace = NULL;
    size_t workspaceSize;
    status = cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputTensor, filterDesc, convDesc, outputTensor, convAlgo, &workspaceSize);
    cudaMalloc(&workspace, workspaceSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    for (int i = 0; i < loops*2; i++)
    {
        if (i==loops)
            cudaEventRecord(start, NULL);

        float alpha = 1.0;
        float beta = 0.0;

        if (!bias && !relu)
        {
            status = cudnnConvolutionForward(cudnnHandle, &alpha, inputTensor,
                input, filterDesc, filter, convDesc,
                convAlgo, workspace, workspaceSize, &beta,
                outputTensor, output);
        }
        else
        {
            status = cudnnConvolutionBiasActivationForward(cudnnHandle, &alpha,
                inputTensor, input, filterDesc, filter, convDesc,
                convAlgo, workspace, workspaceSize, &beta, outputTensor,
                output, biasDesc, bias, actDesc, outputTensor, output);
        }
    }

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    double TFlops = (2.0 * N * W * H * K * C * S * R * loops) / (msecTotal * 1000000000.0);
    printf("Cudnn TFlops: %g\n\n", TFlops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudnnDestroyTensorDescriptor(inputTensor);
    cudnnDestroyTensorDescriptor(outputTensor);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyTensorDescriptor(biasDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyActivationDescriptor(actDesc);
    cudnnDestroy(cudnnHandle);
}

// simplest naive kernel for convolution
// no. of threads = no of output element
// no shared memory used, no data reuse, etc
#if 0
template<int N, int K, int C, int H, int W, int S, int R, typename T>
__global__ void convKernel(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    int n = blockIdx.y;
    int k = blockIdx.x;

    int h = threadIdx.y;
    int w = threadIdx.x;

    float op = 0.0f;
    //#pragma unroll 16
    for (int c = 0; c < C; c++)
    {
        #pragma unroll
        for (int s = 0; s < S; s++)
        {
            #pragma unroll
            for (int r = 0; r < R; r++)
            {
                float filter = (float)(weight[FILTER_IDX_NCHW(k, c, s, r)]);
                int y = h + s - S / 2;
                int x = w + r - R / 2;
                float ip = 0;
                if (y >= 0 && y < H && x >= 0 && x < W)
                    ip = (float)(input[INDEX_NCHW(n, c, y, x)]);
                op += ip * filter;
            }   // r
        }   // s
    }   // c

    if (bias)
        op += (float)(bias[k]);

    if (relu && op < 0)
        op = 0;

    output[INDEX_NCHW(n, k, h, w)] = (T)op;
}
#endif

#if 0
// simple opt, store filter in shared memory
template<int N, int K, int C, int H, int W, int S, int R, typename T>
__global__ void convKernel(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    int n = blockIdx.y;
    int k = blockIdx.x;

    int h = threadIdx.y;
    int w = threadIdx.x;

    int threadInBlock = h*W + w;
    int laneIndex = threadInBlock & 0x1F;

    // the usage pattern here is more like __constant__ memory, 
    // but we don't have enough to store 256x256x3x3 filter data
    __shared__ T shFilter[C*R*S];

    for (int i = 0; i < C*R*S / (H*W); i++)
    {
        int localIndex = (H*W)*i + threadInBlock;
        shFilter[localIndex] = weight[k * (C*R*S) + localIndex];
    }

    __syncthreads();

    float op = 0.0f;
    #pragma unroll 8
    for (int c = 0; c < C; c++)
    {
        #pragma unroll
        for (int s = 0; s < S; s++)
        {
            #pragma unroll
            for (int r = 0; r < R; r++)
            {
                float filter = 0;
                if (laneIndex == 0)
                {
                    //filter = (float)(weight[FILTER_IDX_NCHW(k, c, s, r)]);
                    filter = (float)(shFilter[FILTER_IDX_NCHW(0, c, s, r)]);
                }
                filter = __shfl_sync(0xFFFFFFFF, filter, 0);

                int y = h + s - S / 2;
                int x = w + r - R / 2;
                float ip = 0;
                if (y >= 0 && y < H && x >= 0 && x < W)
                    ip = (float)(input[INDEX_NCHW(n, c, y, x)]);
                op += ip * filter;
            }   // r
        }   // s
    }   // c

    if (bias)
        op += (float)(bias[k]);

    if (relu && op < 0)
        op = 0;

    output[INDEX_NCHW(n, k, h, w)] = (T)op;
}
#endif


#if 0
constexpr int kPerThread = 1;

// get some spatial reuse for input tensor using shfl
// assumes S == R == 3!
// and also H == W == 8
template<int N, int K, int C, int H, int W, int S, int R, typename T>
__global__ void convKernel(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    int n = blockIdx.y;
    int k = blockIdx.x;

    int h = threadIdx.y;
    int w = threadIdx.x;

    int threadInBlock = h*W + w;
    int laneIndex = threadInBlock & 0x1F;

    // the usage pattern here is more like __constant__ memory, 
    // but we don't have enough to store 256x256x3x3 filter data
    __shared__ T shFilter[C*R*S];

    for (int i = 0; i < C*R*S / (H*W); i++)
    {
        int localIndex = (H*W)*i + threadInBlock;
        shFilter[localIndex] = weight[k * (C*R*S) + localIndex];
    }

    __syncthreads();

    float op = 0.0f;
    #pragma unroll 8
    for (int c = 0; c < C; c++)
    {
        // hardcoded for 3x3 filter
        float ic = (float)(input[INDEX_NCHW(n, c, h, w)]);
        float in = 0;
        float is = 0;
        float iw = 0;
        float ie = 0;
        float inw = 0;
        float ine = 0;
        float isw = 0; 
        float ise = 0;

#if 0
        if (h != 0)
            in = (float)(input[INDEX_NCHW(n, c, h - 1, w)]);

        if (h != 7)
            is = (float)(input[INDEX_NCHW(n, c, h + 1, w)]);
#endif

#if 1
        float sn = __shfl_up_sync(0xFFFFFFFF, ic, 8);
        float ss = __shfl_down_sync(0xFFFFFFFF, ic, 8);

        if (h == 4)
            in = (float)(input[INDEX_NCHW(n, c, h-1, w)]);
        else if (h != 0)
            in = sn;

        if (h == 3)
            is = (float)(input[INDEX_NCHW(n, c, h + 1, w)]);
        else if (h != 7)
            is = ss;
#endif

        float sw = __shfl_up_sync(0xFFFFFFFF, ic, 1);
        float snw = __shfl_up_sync(0xFFFFFFFF, in, 1);
        float ssw = __shfl_up_sync(0xFFFFFFFF, is, 1);
        if (w != 0)
        {
            iw = sw;
            inw = snw;
            isw = ssw;
        }

        float se = __shfl_down_sync(0xFFFFFFFF, ic, 1);
        float sne = __shfl_down_sync(0xFFFFFFFF, in, 1);
        float sse = __shfl_down_sync(0xFFFFFFFF, is, 1);

        if (w != 7)
        {
            ie = se;
            ine = sne;
            ise = sse;
        }

        union
        {
            struct
            {
                float nw;
                float n;
                float ne;
                float w;
                float c;
                float e;
                float sw;
                float s;
                float se;
            };
            float arr[3][3];
        } wt;

        #pragma unroll
        for (int s = 0; s < S; s++)
        {
            #pragma unroll
            for (int r = 0; r < R; r++)
            {
                wt.arr[s][r] = (float)(shFilter[FILTER_IDX_NCHW(0, c, s, r)]);
            }
        }


        op +=   ic   * wt.c +
                in   * wt.n +
                is   * wt.s +
                iw   * wt.w +
                ie   * wt.e +
                inw  * wt.nw +
                ine  * wt.ne +
                isw  * wt.sw +
                ise  * wt.se;
    }   // c

    if (bias)
        op += (float)(bias[k]);

    if (relu && op < 0)
        op = 0;

    output[INDEX_NCHW(n, k, h, w)] = (T)op;
}
#endif


#if 0
// get some spatial reuse for input tensor using shfl
// assumes S == R == 3!
// and also H == W == 8

// also do multiple elements in K dimension per thread
constexpr int kPerThread = 1;

constexpr int blockWidth = 8;
constexpr int blockHeight = 8;
constexpr int kPerBlock = kPerThread;


template<int N, int K, int C, int H, int W, int S, int R, typename T>
__global__ void convKernel(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    int n = blockIdx.y;
    int kStart = blockIdx.x * kPerThread;

    int h = threadIdx.y;
    int w = threadIdx.x;

    int threadInBlock = h*W + w;
    int laneIndex = threadInBlock & 0x1F;

    // the usage pattern here is more like __constant__ memory, 
    // but we don't have enough to store 256x256x3x3 filter data
    __shared__ T shFilter[kPerThread * C*R*S];

    #pragma unroll
    for (int k = 0; k < kPerThread; k++)
    {
        #pragma unroll
        for (int i = 0; i < C*R*S / (H*W); i++)
        {
            int localIndex = (H*W)*i + threadInBlock + k*(C*R*S);
            shFilter[localIndex] = weight[kStart * (C*R*S) + localIndex];
        }
    }

    // accumulators
    float op[kPerThread];
    #pragma unroll
    for (int i = 0; i < kPerThread; i++)
        op[i] = 0.0f;

    __syncthreads();



    #pragma unroll 8
    for (int c = 0; c < C; c++)
    {
        // hardcoded for 3x3 filter
        float ic = (float)(input[INDEX_NCHW(n, c, h, w)]);
        float in = 0;
        float is = 0;
        float iw = 0;
        float ie = 0;
        float inw = 0;
        float ine = 0;
        float isw = 0; 
        float ise = 0;

#if 0
        if (h != 0)
            in = (float)(input[INDEX_NCHW(n, c, h - 1, w)]);

        if (h != 7)
            is = (float)(input[INDEX_NCHW(n, c, h + 1, w)]);
#endif

#if 1
        float sn = __shfl_up_sync(0xFFFFFFFF, ic, 8);
        float ss = __shfl_down_sync(0xFFFFFFFF, ic, 8);

        if (h == 4)
            in = (float)(input[INDEX_NCHW(n, c, h - 1, w)]);
        else if (h != 0)
            in = sn;

        if (h == 3)
            is = (float)(input[INDEX_NCHW(n, c, h + 1, w)]);
        else if (h != 7)
            is = ss;
#endif

        float sw = __shfl_up_sync(0xFFFFFFFF, ic, 1);
        float snw = __shfl_up_sync(0xFFFFFFFF, in, 1);
        float ssw = __shfl_up_sync(0xFFFFFFFF, is, 1);
        if (w != 0)
        {
            iw = sw;
            inw = snw;
            isw = ssw;
        }

        float se = __shfl_down_sync(0xFFFFFFFF, ic, 1);
        float sne = __shfl_down_sync(0xFFFFFFFF, in, 1);
        float sse = __shfl_down_sync(0xFFFFFFFF, is, 1);

        if (w != 7)
        {
            ie = se;
            ine = sne;
            ise = sse;
        }

        union
        {
            struct
            {
                float nw;
                float n;
                float ne;
                float w;
                float c;
                float e;
                float sw;
                float s;
                float se;
            };
            float arr[3][3];
        } wt;


        #pragma unroll
        for (int i = 0; i < kPerThread; i++)
        {
            #pragma unroll
            for (int s = 0; s < S; s++)
            {
                #pragma unroll
                for (int r = 0; r < R; r++)
                {
                    wt.arr[s][r] = (float)(shFilter[FILTER_IDX_NCHW(i, c, s, r)]);
                }
            }

            op[i] += ic  * wt.c +
                     in  * wt.n +
                     is  * wt.s +
                     iw  * wt.w +
                     ie  * wt.e +
                     inw * wt.nw +
                     ine * wt.ne +
                     isw * wt.sw +
                     ise * wt.se;
        }
    }   // c

    #pragma unroll
    for (int i = 0; i < kPerThread; i++)
    {
        if (bias)
            op[i] += (float)(bias[kStart + i]);

        if (relu && op[i] < 0)
            op[i] = 0;

        output[INDEX_NCHW(n, kStart + i, h, w)] = (T)op[i];
    }


}
#endif


#if 0
// get some spatial reuse for input tensor using shfl
// assumes S == R == 3!
// and also H == W == 8

// do multiple elements in H and W dimensions (2x2)
constexpr int wPerThread = 2;
constexpr int hPerThread = 2;

// 64 threads in block
constexpr int blockWidth = 8;
constexpr int blockHeight = 8;

// as every thread processes 2x2 output elements a time, we have 2x2 boards per block
constexpr int kPerBlock = blockWidth * blockHeight * wPerThread * hPerThread / 64;

// these many filter elements from c dimension are loaded into 
// shared memory at a time
constexpr int cPerIter = 64;

#define SHFILTER_IDX_NCHW(k,c,h,w) ((k)*cPerIter*S*R + (c)*S*R + (h)*R + w)

template<int N, int K, int C, int H, int W, int S, int R, typename T>
__global__ void convKernel(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    int n = blockIdx.y;
    int kStart = blockIdx.x * kPerBlock;

    int hStart = (threadIdx.y & 0x3) * hPerThread;
    int wStart = (threadIdx.x & 0x3) * wPerThread;
    int kLocal = (threadIdx.y >> 2) * 2 + (threadIdx.x >> 2);

    int threadInBlock = threadIdx.y * blockWidth + threadIdx.x;

    // the usage pattern here is more like __constant__ memory, 
    // but we don't have enough to store 256x256x3x3 filter data
    __shared__ T shFilter[kPerBlock * cPerIter * R * S];

    // accumulators
    float op[2][2];
    op[0][0] = 0.0f;
    op[0][1] = 0.0f;
    op[1][0] = 0.0f;
    op[1][1] = 0.0f;


    //#pragma unroll 2
    // outer loop
    for (int cbase = 0; cbase < C; cbase+= cPerIter)
    {
        // load filters into shared memory
        #pragma unroll
        for (int k = 0; k < kPerBlock; k++)
        {
            #pragma unroll
            for (int i = 0; i < 9; i++)     // 64 threads read cPerIter (64) filters of 3x3 size
            {
                int index = 64*i + threadInBlock;
                int localIndex = k*cPerIter * 9 + index;
                int globalIndex = (kStart + k) * C * 9 + index + cbase * 9;
                shFilter[localIndex] = weight[globalIndex];
            }
        }
        __syncthreads();

        #pragma unroll 16
        for (int lc = 0; lc < cPerIter; lc++)
        {
            int c = cbase + lc;

            // hardcoded for 3x3 filter, and 2x2 spatial elements per thread
            float inEl[4][4];
            #pragma unroll
            for (int y = 0; y < 4; y++)
                #pragma unroll
                for (int x = 0; x < 4; x++)
                    inEl[y][x] = 0.0f;

            
            /*
            inEl[1][1] = (float)(input[INDEX_NCHW(n, c, hStart, wStart)]);
            inEl[1][2] = (float)(input[INDEX_NCHW(n, c, hStart, wStart+1)]);
            inEl[2][1] = (float)(input[INDEX_NCHW(n, c, hStart+1, wStart)]);
            inEl[2][2] = (float)(input[INDEX_NCHW(n, c, hStart+1, wStart+1)]);
            */
            *((uint2*)(&inEl[1][1])) = *((uint2*)(&input[INDEX_NCHW(n, c, hStart + 0, wStart)]));
            *((uint2*)(&inEl[2][1])) = *((uint2*)(&input[INDEX_NCHW(n, c, hStart + 1, wStart)]));

            // need temps because shfl needs all threads in warp to participate
            float t01 = __shfl_up_sync(0xFFFFFFFF, inEl[2][1], 8);
            float t02 = __shfl_up_sync(0xFFFFFFFF, inEl[2][2], 8);
            if (hStart != 0)
            {
                inEl[0][1] = t01;
                inEl[0][2] = t02;
            }

            float t31 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 8);
            float t32 = __shfl_down_sync(0xFFFFFFFF, inEl[1][2], 8);
            if (hStart != 6)
            {
                inEl[3][1] = t31;
                inEl[3][2] = t32;
            }

            float t00 = __shfl_up_sync(0xFFFFFFFF, inEl[0][2], 1);
            float t10 = __shfl_up_sync(0xFFFFFFFF, inEl[1][2], 1);
            float t20 = __shfl_up_sync(0xFFFFFFFF, inEl[2][2], 1);
            float t30 = __shfl_up_sync(0xFFFFFFFF, inEl[3][2], 1);
            if (wStart != 0)
            {
                inEl[0][0] = t00;
                inEl[1][0] = t10;
                inEl[2][0] = t20;
                inEl[3][0] = t30;
            }

            float t03 = __shfl_down_sync(0xFFFFFFFF, inEl[0][1], 1);
            float t13 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 1);
            float t23 = __shfl_down_sync(0xFFFFFFFF, inEl[2][1], 1);
            float t33 = __shfl_down_sync(0xFFFFFFFF, inEl[3][1], 1);
            if (wStart != 6)
            {
                inEl[0][3] = t03;
                inEl[1][3] = t13;
                inEl[2][3] = t23;
                inEl[3][3] = t33;
            }


            #pragma unroll
            for (int s = 0; s < S; s++)
            {
                #pragma unroll
                for (int r = 0; r < R; r++)
                {
                    float weight = (float)(shFilter[SHFILTER_IDX_NCHW(kLocal, lc, s, r)]);
                    #pragma unroll
                    for (int y = 0; y < 2; y++)
                    {
                        #pragma unroll
                        for (int x = 0; x < 2; x++)
                        {
                            op[y][x] += inEl[y + s][x + r] * weight;
                        }
                    }
                }
            }
        } // lc

        __syncthreads();
    }   // cbase

    int k = kStart + kLocal;
    float b = bias ? bias[k] : 0;

    #pragma unroll
    for (int y = 0; y < 2; y++)
    {
        #pragma unroll
        for (int x = 0; x < 2; x++)
        {
            op[y][x] += b;

            if (relu && op[y][x] < 0)
                op[y][x] = 0;

            // TODO: consider 64-bit writes
            // tried below - doesn't make any difference at all! (or maybe about 2% slower)
            output[INDEX_NCHW(n, k, hStart+y, wStart+x)] = (T)op[y][x];
        }
    }

    //((uint2 *)output)[INDEX_NCHW(n, k, hStart, wStart) >> 1] = *((uint2 *)&op[0][0]);
    //((uint2 *)output)[INDEX_NCHW(n, k, hStart+1, wStart) >> 1] = *((uint2 *)&op[1][0]);

}
#endif

// quite a bit slower!
// (too many registers per thread?)
//  --- NO! misaligned/uncoleased reads were the problem
// TFlops: 0.984174 without C dimension split, great hopes with split!
#if 0
// get some spatial reuse for input tensor using shfl
// assumes S == R == 3!
// and also H == W == 8

// do multiple elements in H and W dimensions (4x4)
constexpr int wPerThread = 4;
constexpr int hPerThread = 4;

// 32 threads in block
constexpr int blockWidth  = 4;  // one board (4 threads, with 16 elements per thread)
constexpr int blockHeight = 8;  // different 'C' dimensions

constexpr int kPerBlock = 1;

template<int N, int K, int C, int H, int W, int S, int R, typename T>
__global__ void convKernel(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    int n = blockIdx.y;
    int k = blockIdx.x;

    int hStart = (threadIdx.x >> 1) * hPerThread;
    int wStart = (threadIdx.x  & 1) * wPerThread;
    constexpr int cPerThread = C / blockHeight;
    int cBase  = threadIdx.y * cPerThread;

    int threadInBlock = threadIdx.y * blockWidth + threadIdx.x;

    __shared__ T shFilter[C * R * S];

    // accumulators
    float op[4][4];
    #pragma unroll
    for (int y = 0; y < hPerThread; y++)
        #pragma unroll
        for (int x = 0; x < wPerThread; x++)
            op[y][x] = 0;


    // load filters into shared memory
    #pragma unroll
    for (int i = 0; i < C*R*S / 32; i++)
    {
        int localIndex = (32)*i + threadInBlock;
        shFilter[localIndex] = weight[k * (C*R*S) + localIndex];
    }

    #pragma unroll 8
    for (int lc = 0; lc < cPerThread; lc++)
    {
        int c = cBase + lc;

        // hardcoded for 3x3 filter, and 4x4 spatial elements per thread
        float inEl[hPerThread+2][wPerThread+2];
        #pragma unroll
        for (int y = 0; y < hPerThread+2; y++)
            #pragma unroll
            for (int x = 0; x < wPerThread+2; x++)
                inEl[y][x] = 0.0f;

        /*
        #pragma unroll
        for (int y = 0; y < hPerThread; y++)
            #pragma unroll
            for (int x = 0; x < wPerThread; x++)
            {
                inEl[y+1][x+1] = (float)(input[INDEX_NCHW(n, c, hStart+y, wStart+x)]);
            }
            */

        // assume wPerThread == 4, and use a 128 bit reads
        *((uint4*)(&inEl[1][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart, wStart)]));
        *((uint4*)(&inEl[2][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart+1, wStart)]));
        *((uint4*)(&inEl[3][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart+2, wStart)]));
        *((uint4*)(&inEl[4][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart+3, wStart)]));



        // need temps because shfl needs all threads in warp to participate
        float t01 = __shfl_up_sync(0xFFFFFFFF, inEl[4][1], 2);
        float t02 = __shfl_up_sync(0xFFFFFFFF, inEl[4][2], 2);
        float t03 = __shfl_up_sync(0xFFFFFFFF, inEl[4][3], 2);
        float t04 = __shfl_up_sync(0xFFFFFFFF, inEl[4][4], 2);
        if (hStart != 0)
        {
            inEl[0][1] = t01;
            inEl[0][2] = t02;
            inEl[0][3] = t03;
            inEl[0][4] = t04;
        }

        float t51 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 2);
        float t52 = __shfl_down_sync(0xFFFFFFFF, inEl[1][2], 2);
        float t53 = __shfl_down_sync(0xFFFFFFFF, inEl[1][3], 2);
        float t54 = __shfl_down_sync(0xFFFFFFFF, inEl[1][4], 2);
        if (hStart == 0)
        {
            inEl[5][1] = t51;
            inEl[5][2] = t52;
            inEl[5][3] = t53;
            inEl[5][4] = t54;
        }

        float t00 = __shfl_up_sync(0xFFFFFFFF, inEl[0][4], 1);
        float t10 = __shfl_up_sync(0xFFFFFFFF, inEl[1][4], 1);
        float t20 = __shfl_up_sync(0xFFFFFFFF, inEl[2][4], 1);
        float t30 = __shfl_up_sync(0xFFFFFFFF, inEl[3][4], 1);
        float t40 = __shfl_up_sync(0xFFFFFFFF, inEl[4][4], 1);
        float t50 = __shfl_up_sync(0xFFFFFFFF, inEl[5][4], 1);
        if (wStart != 0)
        {
            inEl[0][0] = t00;
            inEl[1][0] = t10;
            inEl[2][0] = t20;
            inEl[3][0] = t30;
            inEl[4][0] = t40;
            inEl[5][0] = t50;
        }

        float t05 = __shfl_down_sync(0xFFFFFFFF, inEl[0][1], 1);
        float t15 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 1);
        float t25 = __shfl_down_sync(0xFFFFFFFF, inEl[2][1], 1);
        float t35 = __shfl_down_sync(0xFFFFFFFF, inEl[3][1], 1);
        float t45 = __shfl_down_sync(0xFFFFFFFF, inEl[4][1], 1);
        float t55 = __shfl_down_sync(0xFFFFFFFF, inEl[5][1], 1);
        if (wStart == 0)
        {
            inEl[0][5] = t05;
            inEl[1][5] = t15;
            inEl[2][5] = t25;
            inEl[3][5] = t35;
            inEl[4][5] = t45;
            inEl[5][5] = t55;
        }


        #pragma unroll
        for (int s = 0; s < S; s++)
        {
            #pragma unroll
            for (int r = 0; r < R; r++)
            {
                float weight = (float)(shFilter[FILTER_IDX_NCHW(0, c, s, r)]);
                #pragma unroll
                for (int y = 0; y < hPerThread; y++)
                {
                    #pragma unroll
                    for (int x = 0; x < wPerThread; x++)
                    {
                        op[y][x] += inEl[y + s][x + r] * weight;
                    }
                }
            }
        }
    } // lc / c

    float b = bias ? bias[k] : 0;

    #pragma unroll
    for (int y = 0; y < hPerThread; y++)
    {
        #pragma unroll
        for (int x = 0; x < wPerThread; x++)
        {
            // sum across C dimension
            op[y][x] += __shfl_down_sync(0xFFFFFFFF, op[y][x], 4);
            op[y][x] += __shfl_down_sync(0xFFFFFFFF, op[y][x], 8);
            op[y][x] += __shfl_down_sync(0xFFFFFFFF, op[y][x], 16);

            op[y][x] += b;

            if (relu && op[y][x] < 0)
                op[y][x] = 0;

            // TODO: consider 128-bit writes
            // tried below - little bit (2%) faster!

            //if (threadIdx.y == 0)
            //    output[INDEX_NCHW(n, k, hStart+y, wStart+x)] = (T)op[y][x];
        }
    }

    if (threadIdx.y == 0)
    {
        ((uint4 *)output)[INDEX_NCHW(n, k, hStart, wStart) >> 2] = *((uint4 *)&op[0][0]);
        ((uint4 *)output)[INDEX_NCHW(n, k, hStart+1, wStart) >> 2] = *((uint4 *)&op[1][0]);
        ((uint4 *)output)[INDEX_NCHW(n, k, hStart+2, wStart) >> 2] = *((uint4 *)&op[2][0]);
        ((uint4 *)output)[INDEX_NCHW(n, k, hStart+3, wStart) >> 2] = *((uint4 *)&op[3][0]);
    }

}
#endif

// significantly slower than 2x2 boards per thread block! ??
// this is because of shared memory usage, no of blocks per SM limit causes less no of warps per SM -> very low occupancy
#if 0
// AGAIN, some bug causing minor differences (max difference, 4.57%, avg difference 0.99%)
//   ... fixed, be careful of shfl offsets
//  .. 0.74 TFlops
// get some spatial reuse for input tensor using shfl
// assumes S == R == 3!
// and also H == W == 8

// do multiple elements in H and W dimensions (2x2)
constexpr int wPerThread = 2;
constexpr int hPerThread = 2;

// 32 threads in block
constexpr int blockWidth  = 16;  // one board (16 threads, with 4 elements per thread)
constexpr int blockHeight =  2;  // different 'C' dimensions

constexpr int kPerBlock = 1;

template<int N, int K, int C, int H, int W, int S, int R, typename T>
__global__ void convKernel(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    int n = blockIdx.y;
    int k = blockIdx.x;

    int hStart = (threadIdx.x >> 2) * hPerThread;
    int wStart = (threadIdx.x  & 3) * wPerThread;
    constexpr int cPerThread = C / blockHeight;
    int cBase  = threadIdx.y * cPerThread;

    int threadInBlock = threadIdx.y * blockWidth + threadIdx.x;

    __shared__ T shFilter[C * R * S];

    // accumulators
    float op[2][2];
    #pragma unroll
    for (int y = 0; y < hPerThread; y++)
        #pragma unroll
        for (int x = 0; x < wPerThread; x++)
            op[y][x] = 0;


    // load filters into shared memory
    #pragma unroll
    for (int i = 0; i < C*R*S / 32; i++)
    {
        int localIndex = (32)*i + threadInBlock;
        shFilter[localIndex] = weight[k * (C*R*S) + localIndex];
    }

    #pragma unroll 8
    for (int lc = 0; lc < cPerThread; lc++)
    {
        int c = cBase + lc;

        // hardcoded for 3x3 filter, and 2x2 spatial elements per thread
        float inEl[hPerThread+2][wPerThread+2];
        #pragma unroll
        for (int y = 0; y < hPerThread+2; y++)
            #pragma unroll
            for (int x = 0; x < wPerThread+2; x++)
                inEl[y][x] = 0;

        #pragma unroll
        for (int y = 0; y < hPerThread; y++)
            #pragma unroll
            for (int x = 0; x < wPerThread; x++)
            {
                inEl[y+1][x+1] = (float)(input[INDEX_NCHW(n, c, hStart+y, wStart+x)]);
            }

        // need temps because shfl needs all threads in warp to participate
        float t01 = __shfl_up_sync(0xFFFFFFFF, inEl[2][1], 4);
        float t02 = __shfl_up_sync(0xFFFFFFFF, inEl[2][2], 4);
        if (hStart != 0)
        {
            inEl[0][1] = t01;
            inEl[0][2] = t02;
        }

        float t31 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 4);
        float t32 = __shfl_down_sync(0xFFFFFFFF, inEl[1][2], 4);
        if (hStart != 6)
        {
            inEl[3][1] = t31;
            inEl[3][2] = t32;
        }

        float t00 = __shfl_up_sync(0xFFFFFFFF, inEl[0][2], 1);
        float t10 = __shfl_up_sync(0xFFFFFFFF, inEl[1][2], 1);
        float t20 = __shfl_up_sync(0xFFFFFFFF, inEl[2][2], 1);
        float t30 = __shfl_up_sync(0xFFFFFFFF, inEl[3][2], 1);
        if (wStart != 0)
        {
            inEl[0][0] = t00;
            inEl[1][0] = t10;
            inEl[2][0] = t20;
            inEl[3][0] = t30;
        }

        float t03 = __shfl_down_sync(0xFFFFFFFF, inEl[0][1], 1);
        float t13 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 1);
        float t23 = __shfl_down_sync(0xFFFFFFFF, inEl[2][1], 1);
        float t33 = __shfl_down_sync(0xFFFFFFFF, inEl[3][1], 1);
        if (wStart != 6)
        {
            inEl[0][3] = t03;
            inEl[1][3] = t13;
            inEl[2][3] = t23;
            inEl[3][3] = t33;
        }

        #pragma unroll
        for (int s = 0; s < S; s++)
        {
            #pragma unroll
            for (int r = 0; r < R; r++)
            {
                float weight = (float)(shFilter[FILTER_IDX_NCHW(0, c, s, r)]);
                #pragma unroll
                for (int y = 0; y < hPerThread; y++)
                {
                    #pragma unroll
                    for (int x = 0; x < wPerThread; x++)
                    {
                        op[y][x] += inEl[y + s][x + r] * weight;
                    }
                }
            }
        }
    } // lc / c

    float b = bias ? bias[k] : 0;

    #pragma unroll
    for (int y = 0; y < hPerThread; y++)
    {
        #pragma unroll
        for (int x = 0; x < wPerThread; x++)
        {
            // sum across C dimension
            op[y][x] += __shfl_down_sync(0xFFFFFFFF, op[y][x], 16);

            op[y][x] += b;

            if (relu && op[y][x] < 0)
                op[y][x] = 0;

            // TODO: consider 64-bit writes
            // tried below - doesn't make any difference at all! (or maybe about 2% slower)

            if (threadIdx.y == 0)
                output[INDEX_NCHW(n, k, hStart+y, wStart+x)] = (T)op[y][x];
        }
    }

    /*
    if (threadIdx.y == 0)
    {
        ((uint2 *)output)[INDEX_NCHW(n, k, hStart, wStart) >> 1] = *((uint2 *)&op[0][0]);
        ((uint2 *)output)[INDEX_NCHW(n, k, hStart + 1, wStart) >> 1] = *((uint2 *)&op[1][0]);
    }
    */
}
#endif


#if 0
// Assumes  
// - S == R == 3
// - H == W == 8
// 
// Optimizations:
// - some spatial reuse for input tensor using shfl
// - do multiple elements in H and W dimensions (2x2) per thread, to get more spatial reuse for input tensor as well as filter
// - C dimension is sliced into multiple chunks (of 32) to reduce shared memory usage to get better occupancy (allow more blocks per SM)

constexpr int wPerThread = 2;
constexpr int hPerThread = 2;

// 32 threads in block
constexpr int blockWidth  = 16;  // one board (16 threads, with 4 elements per thread)
constexpr int blockHeight =  2;  // different 'C' dimensions

// these many filter elements from c dimension are loaded into 
// shared memory at a time (should be a multiple of warp size)
constexpr int cPerIter = 32;    
constexpr int cPerIterPerThread = cPerIter / blockHeight;

constexpr int kPerBlock = 1;

template<int N, int K, int C, int H, int W, int S, int R, typename T>
__global__ void convKernel(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    int n = blockIdx.y;
    int k = blockIdx.x;

    int hStart = (threadIdx.x >> 2) * hPerThread;
    int wStart = (threadIdx.x  & 3) * wPerThread;

    // either 0 or 16 to indicate offset to be added to get C index
    int cOffset  = threadIdx.y * cPerIterPerThread;

    int threadInBlock = threadIdx.y * blockWidth + threadIdx.x;

    __shared__ T shFilter[cPerIter * R * S];

    // accumulators
    float op[2][2];
    #pragma unroll
    for (int y = 0; y < hPerThread; y++)
        #pragma unroll
        for (int x = 0; x < wPerThread; x++)
            op[y][x] = 0;



    // outer loop
    // #pragma unroll 4
    for (int cBase = 0; cBase < C; cBase += cPerIter)
    {
        // load filters into shared memory
        #pragma unroll
        for (int i = 0; i < cPerIter*R*S / 32; i++)
        {
            int localIndex = 32 * i + threadInBlock;
            shFilter[localIndex] = weight[k * (C*R*S) + cBase * (R*S) + localIndex];
        }


        #pragma unroll 8
        for (int lc = 0; lc < cPerIterPerThread; lc++)
        {
            int shc = cOffset + lc;     // offset of filter for index c in shared memory
            int c = cBase + shc;        // real c dimension

            // hardcoded for 3x3 filter, and 2x2 spatial elements per thread
            float inEl[hPerThread+2][wPerThread+2];
            #pragma unroll
            for (int y = 0; y < hPerThread+2; y++)
                #pragma unroll
                for (int x = 0; x < wPerThread+2; x++)
                    inEl[y][x] = 0;

#if 0
            #pragma unroll
            for (int y = 0; y < hPerThread; y++)
                #pragma unroll
                for (int x = 0; x < wPerThread; x++)
                {
                    inEl[y+1][x+1] = (float)(input[INDEX_NCHW(n, c, hStart+y, wStart+x)]);
                }
#endif


            // assume wPerThread == 2, and use a single 64 bit read
            *((uint2*)(&inEl[1][1])) = *((uint2*)(&input[INDEX_NCHW(n, c, hStart, wStart)]));
            *((uint2*)(&inEl[2][1])) = *((uint2*)(&input[INDEX_NCHW(n, c, hStart+1, wStart)]));

            // buggy! and slow as needs local memory spill (due to indexed register usage for shfl)
#if 0
            // read entire board in a single request, and then use shfl to get elements in right place
            union {
                float4 vec;
                float arr[4];
            } temp;

            temp.vec = *((float4*)(&input[INDEX_NCHW(n, c, 0, threadIdx.x*4)]));

            int i0 = (hStart * 8 + wStart) & 3;
            int i1 = (hStart * 8 + wStart) >> 2;
            inEl[1][1] = __shfl_sync(0xFFFFFFFF, temp.arr[i0], i1);
            inEl[1][2] = __shfl_sync(0xFFFFFFFF, temp.arr[i0+1], i1);
            inEl[2][1] = __shfl_sync(0xFFFFFFFF, temp.arr[i0], i1+2);
            inEl[2][2] = __shfl_sync(0xFFFFFFFF, temp.arr[i0+1], i1+2);
#endif

            // need temps because shfl needs all threads in warp to participate
            float t01 = __shfl_up_sync(0xFFFFFFFF, inEl[2][1], 4);
            float t02 = __shfl_up_sync(0xFFFFFFFF, inEl[2][2], 4);
            if (hStart != 0)
            {
                inEl[0][1] = t01;
                inEl[0][2] = t02;
            }

            float t31 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 4);
            float t32 = __shfl_down_sync(0xFFFFFFFF, inEl[1][2], 4);
            if (hStart != 6)
            {
                inEl[3][1] = t31;
                inEl[3][2] = t32;
            }

            float t00 = __shfl_up_sync(0xFFFFFFFF, inEl[0][2], 1);
            float t10 = __shfl_up_sync(0xFFFFFFFF, inEl[1][2], 1);
            float t20 = __shfl_up_sync(0xFFFFFFFF, inEl[2][2], 1);
            float t30 = __shfl_up_sync(0xFFFFFFFF, inEl[3][2], 1);
            if (wStart != 0)
            {
                inEl[0][0] = t00;
                inEl[1][0] = t10;
                inEl[2][0] = t20;
                inEl[3][0] = t30;
            }

            float t03 = __shfl_down_sync(0xFFFFFFFF, inEl[0][1], 1);
            float t13 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 1);
            float t23 = __shfl_down_sync(0xFFFFFFFF, inEl[2][1], 1);
            float t33 = __shfl_down_sync(0xFFFFFFFF, inEl[3][1], 1);
            if (wStart != 6)
            {
                inEl[0][3] = t03;
                inEl[1][3] = t13;
                inEl[2][3] = t23;
                inEl[3][3] = t33;
            }

            #pragma unroll
            for (int s = 0; s < S; s++)
            {
                #pragma unroll
                for (int r = 0; r < R; r++)
                {
                    float weight = (float)(shFilter[FILTER_IDX_NCHW(0, shc, s, r)]);
                    #pragma unroll
                    for (int y = 0; y < hPerThread; y++)
                    {
                        #pragma unroll
                        for (int x = 0; x < wPerThread; x++)
                        {
                            op[y][x] += inEl[y + s][x + r] * weight;
                        }
                    }
                }
            }
        } // lc / c
    }   // cBase

    float b = bias ? bias[k] : 0;

    #pragma unroll
    for (int y = 0; y < hPerThread; y++)
    {
        #pragma unroll
        for (int x = 0; x < wPerThread; x++)
        {
            // sum across C dimension
            op[y][x] += __shfl_down_sync(0xFFFFFFFF, op[y][x], 16);

            op[y][x] += b;

            if (relu && op[y][x] < 0)
                op[y][x] = 0;

            // TODO: consider 64-bit writes
            // tried below - doesn't make any difference at all! (or maybe about 2% slower)

            if (threadIdx.y == 0)
                output[INDEX_NCHW(n, k, hStart+y, wStart+x)] = (T)op[y][x];
        }
    }

    /*
    if (threadIdx.y == 0)
    {
        ((uint2 *)output)[INDEX_NCHW(n, k, hStart, wStart) >> 1] = *((uint2 *)&op[0][0]);
        ((uint2 *)output)[INDEX_NCHW(n, k, hStart + 1, wStart) >> 1] = *((uint2 *)&op[1][0]);
    }
    */
    
}
#endif



#if 0
// Assumes  
// - S == R == 3
// - H == W == 8
// 
// Optimizations:
// - some spatial reuse for input tensor using shfl
// - do multiple elements in H and W dimensions (4x4) per thread, to get more spatial reuse for input tensor as well as filter
// - C dimension is sliced into multiple chunks (of 64) to reduce shared memory usage to get better occupancy (allow more blocks per SM)

constexpr int wPerThread = 4;
constexpr int hPerThread = 4;

// 32 threads in block
constexpr int blockWidth  = 4;  // one board (4 threads, with 16 elements per thread)
constexpr int blockHeight = 8;  // different 'C' dimensions

// these many filter elements from c dimension are loaded into 
// shared memory at a time (should be a multiple of warp size)
constexpr int cPerIter = 32;    
constexpr int cPerIterPerThread = cPerIter / blockHeight;

constexpr int kPerBlock = 1;

template<int N, int K, int C, int H, int W, int S, int R, typename T>
__global__ void convKernel(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    int n = blockIdx.y;
    int k = blockIdx.x;

    int hStart = (threadIdx.x >> 1) * hPerThread;
    int wStart = (threadIdx.x  & 1) * wPerThread;

    // offset to be added to get C index
    int cOffset  = threadIdx.y * cPerIterPerThread;

    int threadInBlock = threadIdx.y * blockWidth + threadIdx.x;

    __shared__ T shFilter[cPerIter * R * S];

    // accumulators
    float op[4][4];
    #pragma unroll
    for (int y = 0; y < hPerThread; y++)
        #pragma unroll
        for (int x = 0; x < wPerThread; x++)
            op[y][x] = 0;

    // outer loop
    // #pragma unroll 4
    for (int cBase = 0; cBase < C; cBase += cPerIter)
    {
        // load filters into shared memory
        #pragma unroll
        for (int i = 0; i < cPerIter*R*S / 32; i++)
        {
            int localIndex = 32 * i + threadInBlock;
            shFilter[localIndex] = weight[k * (C*R*S) + cBase * (R*S) + localIndex];
        }


        #pragma unroll
        for (int lc = 0; lc < cPerIterPerThread; lc++)
        {
            int shc = cOffset + lc;     // offset of filter for index c in shared memory
            int c = cBase + shc;        // real c dimension

            // hardcoded for 3x3 filter, and 4x4 spatial elements per thread
            float inEl[hPerThread+2][wPerThread+2];
            #pragma unroll
            for (int y = 0; y < hPerThread+2; y++)
                #pragma unroll
                for (int x = 0; x < wPerThread+2; x++)
                    inEl[y][x] = 0.0f;


            // assume wPerThread == 4, and use a 128 bit reads
            *((uint4*)(&inEl[1][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart, wStart)]));
            *((uint4*)(&inEl[2][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart + 1, wStart)]));
            *((uint4*)(&inEl[3][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart + 2, wStart)]));
            *((uint4*)(&inEl[4][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart + 3, wStart)]));

            // need temps because shfl needs all threads in warp to participate
            float t01 = __shfl_up_sync(0xFFFFFFFF, inEl[4][1], 2);
            float t02 = __shfl_up_sync(0xFFFFFFFF, inEl[4][2], 2);
            float t03 = __shfl_up_sync(0xFFFFFFFF, inEl[4][3], 2);
            float t04 = __shfl_up_sync(0xFFFFFFFF, inEl[4][4], 2);
            if (hStart != 0)
            {
                inEl[0][1] = t01;
                inEl[0][2] = t02;
                inEl[0][3] = t03;
                inEl[0][4] = t04;
            }

            float t51 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 2);
            float t52 = __shfl_down_sync(0xFFFFFFFF, inEl[1][2], 2);
            float t53 = __shfl_down_sync(0xFFFFFFFF, inEl[1][3], 2);
            float t54 = __shfl_down_sync(0xFFFFFFFF, inEl[1][4], 2);
            if (hStart == 0)
            {
                inEl[5][1] = t51;
                inEl[5][2] = t52;
                inEl[5][3] = t53;
                inEl[5][4] = t54;
            }

            float t00 = __shfl_up_sync(0xFFFFFFFF, inEl[0][4], 1);
            float t10 = __shfl_up_sync(0xFFFFFFFF, inEl[1][4], 1);
            float t20 = __shfl_up_sync(0xFFFFFFFF, inEl[2][4], 1);
            float t30 = __shfl_up_sync(0xFFFFFFFF, inEl[3][4], 1);
            float t40 = __shfl_up_sync(0xFFFFFFFF, inEl[4][4], 1);
            float t50 = __shfl_up_sync(0xFFFFFFFF, inEl[5][4], 1);
            if (wStart != 0)
            {
                inEl[0][0] = t00;
                inEl[1][0] = t10;
                inEl[2][0] = t20;
                inEl[3][0] = t30;
                inEl[4][0] = t40;
                inEl[5][0] = t50;
            }

            float t05 = __shfl_down_sync(0xFFFFFFFF, inEl[0][1], 1);
            float t15 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 1);
            float t25 = __shfl_down_sync(0xFFFFFFFF, inEl[2][1], 1);
            float t35 = __shfl_down_sync(0xFFFFFFFF, inEl[3][1], 1);
            float t45 = __shfl_down_sync(0xFFFFFFFF, inEl[4][1], 1);
            float t55 = __shfl_down_sync(0xFFFFFFFF, inEl[5][1], 1);
            if (wStart == 0)
            {
                inEl[0][5] = t05;
                inEl[1][5] = t15;
                inEl[2][5] = t25;
                inEl[3][5] = t35;
                inEl[4][5] = t45;
                inEl[5][5] = t55;
            }

            #pragma unroll
            for (int s = 0; s < S; s++)
            {
                #pragma unroll
                for (int r = 0; r < R; r++)
                {
                    float weight = (float)(shFilter[FILTER_IDX_NCHW(0, shc, s, r)]);
                    #pragma unroll
                    for (int y = 0; y < hPerThread; y++)
                    {
                        #pragma unroll
                        for (int x = 0; x < wPerThread; x++)
                        {
                            op[y][x] += inEl[y + s][x + r] * weight;
                        }
                    }
                }
            }
        } // lc / c
    }   // cBase

    float b = bias ? bias[k] : 0;

    #pragma unroll
    for (int y = 0; y < hPerThread; y++)
    {
        #pragma unroll
        for (int x = 0; x < wPerThread; x++)
        {
            // sum across C dimension
            op[y][x] += __shfl_down_sync(0xFFFFFFFF, op[y][x], 4);
            op[y][x] += __shfl_down_sync(0xFFFFFFFF, op[y][x], 8);
            op[y][x] += __shfl_down_sync(0xFFFFFFFF, op[y][x], 16);

            op[y][x] += b;

            if (relu && op[y][x] < 0)
                op[y][x] = 0;

            // TODO: consider 128-bit writes
            // tried below - little bit (2%) faster!

            //if (threadIdx.y == 0)
            //    output[INDEX_NCHW(n, k, hStart+y, wStart+x)] = (T)op[y][x];
        }
    }

    if (threadIdx.y == 0)
    {
        ((uint4 *)output)[INDEX_NCHW(n, k, hStart, wStart) >> 2] = *((uint4 *)&op[0][0]);
        ((uint4 *)output)[INDEX_NCHW(n, k, hStart + 1, wStart) >> 2] = *((uint4 *)&op[1][0]);
        ((uint4 *)output)[INDEX_NCHW(n, k, hStart + 2, wStart) >> 2] = *((uint4 *)&op[2][0]);
        ((uint4 *)output)[INDEX_NCHW(n, k, hStart + 3, wStart) >> 2] = *((uint4 *)&op[3][0]);
    }
    
}
#endif


#if 0
// Assumes  
// - S == R == 3
// - H == W == 8
// 
// Optimizations:
// - some spatial reuse for input tensor using shfl
// - do multiple elements in H and W dimensions (4x4) per thread, to get more spatial reuse for input tensor as well as filter
// - C dimension is sliced into multiple chunks (of 64) to reduce shared memory usage to get better occupancy (allow more blocks per SM)
// - multiple elements (2) in K dimension processed by thread
//   -- this gets more reuse of the input tensor

constexpr int wPerThread = 4;
constexpr int hPerThread = 4;

// 32 threads in block
constexpr int blockWidth  = 4;  // one board (4 threads, with 16 elements per thread)
constexpr int blockHeight = 8;  // different 'C' dimensions

// these many filter elements from c dimension are loaded into 
// shared memory at a time (should be a multiple of warp size)
constexpr int cPerIter = 32;    
constexpr int cPerIterPerThread = cPerIter / blockHeight;

constexpr int kPerBlock = 2;

#define SH_FILTER_IDX(k,c,h,w) ((k)*cPerIter*S*R + (c)*S*R + (h)*R + w)

template<int N, int K, int C, int H, int W, int S, int R, typename T>
__global__ void convKernel(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    int n = blockIdx.y;
    int kStart = blockIdx.x * kPerBlock;

    int hStart = (threadIdx.x >> 1) * hPerThread;
    int wStart = (threadIdx.x  & 1) * wPerThread;

    // offset to be added to get C index
    int cOffset  = threadIdx.y * cPerIterPerThread;

    int threadInBlock = threadIdx.y * blockWidth + threadIdx.x;

    __shared__ T shFilter[kPerBlock * cPerIter * R * S];

    // accumulators
    float op[kPerBlock][4][4];

    #pragma unroll 
    for (int lk = 0; lk < kPerBlock; lk++)
        #pragma unroll
        for (int y = 0; y < hPerThread; y++)
            #pragma unroll
            for (int x = 0; x < wPerThread; x++)
                op[lk][y][x] = 0;

    // outer loop
    // #pragma unroll 4
    for (int cBase = 0; cBase < C; cBase += cPerIter)
    {
        // load filters into shared memory
        #pragma unroll 
        for (int lk = 0; lk < kPerBlock; lk++)
        {
            int k = kStart + lk;
            #pragma unroll
            for (int i = 0; i < cPerIter*R*S / 32; i++)
            {
                int localIndex = 32 * i + threadInBlock;
                shFilter[localIndex + lk * (cPerIter*R*S)] = weight[k * (C*R*S) + cBase * (R*S) + localIndex];
            }
        }

        #pragma unroll
        for (int lc = 0; lc < cPerIterPerThread; lc++)
        {
            int shc = cOffset + lc;     // offset of filter for index c in shared memory
            int c = cBase + shc;        // real c dimension

            // hardcoded for 3x3 filter, and 4x4 spatial elements per thread
            float inEl[hPerThread+2][wPerThread+2];
            #pragma unroll
            for (int y = 0; y < hPerThread+2; y++)
                #pragma unroll
                for (int x = 0; x < wPerThread+2; x++)
                    inEl[y][x] = 0.0f;


            // assume wPerThread == 4, and use a 128 bit reads
            *((uint4*)(&inEl[1][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart, wStart)]));
            *((uint4*)(&inEl[2][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart + 1, wStart)]));
            *((uint4*)(&inEl[3][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart + 2, wStart)]));
            *((uint4*)(&inEl[4][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart + 3, wStart)]));

            // need temps because shfl needs all threads in warp to participate
            float t01 = __shfl_up_sync(0xFFFFFFFF, inEl[4][1], 2);
            float t02 = __shfl_up_sync(0xFFFFFFFF, inEl[4][2], 2);
            float t03 = __shfl_up_sync(0xFFFFFFFF, inEl[4][3], 2);
            float t04 = __shfl_up_sync(0xFFFFFFFF, inEl[4][4], 2);
            if (hStart != 0)
            {
                inEl[0][1] = t01;
                inEl[0][2] = t02;
                inEl[0][3] = t03;
                inEl[0][4] = t04;
            }

            float t51 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 2);
            float t52 = __shfl_down_sync(0xFFFFFFFF, inEl[1][2], 2);
            float t53 = __shfl_down_sync(0xFFFFFFFF, inEl[1][3], 2);
            float t54 = __shfl_down_sync(0xFFFFFFFF, inEl[1][4], 2);
            if (hStart == 0)
            {
                inEl[5][1] = t51;
                inEl[5][2] = t52;
                inEl[5][3] = t53;
                inEl[5][4] = t54;
            }

            float t00 = __shfl_up_sync(0xFFFFFFFF, inEl[0][4], 1);
            float t10 = __shfl_up_sync(0xFFFFFFFF, inEl[1][4], 1);
            float t20 = __shfl_up_sync(0xFFFFFFFF, inEl[2][4], 1);
            float t30 = __shfl_up_sync(0xFFFFFFFF, inEl[3][4], 1);
            float t40 = __shfl_up_sync(0xFFFFFFFF, inEl[4][4], 1);
            float t50 = __shfl_up_sync(0xFFFFFFFF, inEl[5][4], 1);
            if (wStart != 0)
            {
                inEl[0][0] = t00;
                inEl[1][0] = t10;
                inEl[2][0] = t20;
                inEl[3][0] = t30;
                inEl[4][0] = t40;
                inEl[5][0] = t50;
            }

            float t05 = __shfl_down_sync(0xFFFFFFFF, inEl[0][1], 1);
            float t15 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 1);
            float t25 = __shfl_down_sync(0xFFFFFFFF, inEl[2][1], 1);
            float t35 = __shfl_down_sync(0xFFFFFFFF, inEl[3][1], 1);
            float t45 = __shfl_down_sync(0xFFFFFFFF, inEl[4][1], 1);
            float t55 = __shfl_down_sync(0xFFFFFFFF, inEl[5][1], 1);
            if (wStart == 0)
            {
                inEl[0][5] = t05;
                inEl[1][5] = t15;
                inEl[2][5] = t25;
                inEl[3][5] = t35;
                inEl[4][5] = t45;
                inEl[5][5] = t55;
            }

            #pragma unroll
            for (int s = 0; s < S; s++)
            {
                #pragma unroll
                for (int r = 0; r < R; r++)
                {
                    #pragma unroll 
                    for (int lk = 0; lk < kPerBlock; lk++)
                    {
                        float weight = (float)(shFilter[SH_FILTER_IDX(lk, shc, s, r)]);
                        #pragma unroll
                        for (int y = 0; y < hPerThread; y++)
                        {
                            #pragma unroll
                            for (int x = 0; x < wPerThread; x++)
                            {
                                op[lk][y][x] += inEl[y + s][x + r] * weight;
                            }   // x
                        }   // y
                    }   // k
                }   // r
            } // s
        } // lc
    }   // cBase


    float b[kPerBlock];
    #pragma unroll 
    for (int lk = 0; lk < kPerBlock; lk++)
        b[lk] = bias ? bias[kStart+lk] : 0;

    #pragma unroll
    for (int y = 0; y < hPerThread; y++)
    {
        #pragma unroll
        for (int x = 0; x < wPerThread; x++)
        {
            // sum across C dimension
            #pragma unroll 
            for (int lk = 0; lk < kPerBlock; lk++)
            {
                op[lk][y][x] += __shfl_down_sync(0xFFFFFFFF, op[lk][y][x], 4);
                op[lk][y][x] += __shfl_down_sync(0xFFFFFFFF, op[lk][y][x], 8);
                op[lk][y][x] += __shfl_down_sync(0xFFFFFFFF, op[lk][y][x], 16);

                op[lk][y][x] += b[lk];

                if (relu && op[lk][y][x] < 0)
                    op[lk][y][x] = 0;
            }

            // TODO: consider 128-bit writes
            // tried below - little bit (2%) faster!

            //if (threadIdx.y == 0)
            //    output[INDEX_NCHW(n, kStart+lk, hStart+y, wStart+x)] = (T)op[lk][y][x];
        }
    }

    if (threadIdx.y == 0)
    {
        #pragma unroll 
        for (int lk = 0; lk < kPerBlock; lk++)
        {
            int k = kStart + lk;
            ((uint4 *)output)[INDEX_NCHW(n, k, hStart, wStart) >> 2] = *((uint4 *)&op[lk][0][0]);
            ((uint4 *)output)[INDEX_NCHW(n, k, hStart + 1, wStart) >> 2] = *((uint4 *)&op[lk][1][0]);
            ((uint4 *)output)[INDEX_NCHW(n, k, hStart + 2, wStart) >> 2] = *((uint4 *)&op[lk][2][0]);
            ((uint4 *)output)[INDEX_NCHW(n, k, hStart + 3, wStart) >> 2] = *((uint4 *)&op[lk][3][0]);
        }
    }
    
}
#endif


#if 1
// Assumes  
// - S == R == 3
// - H == W == 8
// 
// Optimizations:
// - some spatial reuse for input tensor using shfl
// - do multiple elements in H and W dimensions (4x4) per thread, to get more spatial reuse for input tensor as well as filter
// - C dimension is sliced into multiple chunks (of 32) to reduce shared memory usage to get better occupancy (allow more blocks per SM)
// - multiple elements (2) in K dimension processed by thread
//   -- this gets more reuse of the input tensor
// - another slicing along C dimension to increase occupancy
//   -- every alternate thread writes output (i.e, two threads compute 

constexpr int wPerThread = 4;
constexpr int hPerThread = 4;

// 32 threads in block
constexpr int blockWidth  = 4;  // one board (4 threads, with 16 elements per thread)
constexpr int blockHeight = 8;  // different 'C' dimensions

// the code to do final sums and to write outputs below assumes this to be 2
constexpr int blockDepth = 2;   // again different 'C' dimension (in a block of shared memory accessed by different warps)

// these many filter elements from c dimension are loaded into 
// shared memory at a time (should be a multiple of warp size)
constexpr int cPerIter = 32;
constexpr int cPerIterPerThread = cPerIter / (blockHeight);

constexpr int kPerBlock = 2;

#define SH_FILTER_IDX(k,c,h,w) ((k)*cPerIter*S*R + (c)*S*R + (h)*R + w)

template<int N, int K, int C, int H, int W, int S, int R, typename T>
__global__ void convKernel(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    int n = blockIdx.y;
    int kStart = blockIdx.x * kPerBlock;

    int hStart = (threadIdx.x >> 1) * hPerThread;
    int wStart = (threadIdx.x  & 1) * wPerThread;

    // extra offset
    int cPerSlice = C / blockDepth;
    int cStart = threadIdx.z * cPerSlice;
    int shStart = (kPerBlock*cPerIter*R*S)*threadIdx.z;

    // offset to be added to get C index
    int cOffset = threadIdx.y * cPerIterPerThread;

    int threadInWarp = threadIdx.y * blockWidth + threadIdx.x;

    __shared__ float shData[blockDepth * kPerBlock * cPerIter * R * S];

    // accumulators
    float op[kPerBlock][4][4];

    #pragma unroll 
    for (int lk = 0; lk < kPerBlock; lk++)
        #pragma unroll
        for (int y = 0; y < hPerThread; y++)
            #pragma unroll
            for (int x = 0; x < wPerThread; x++)
                op[lk][y][x] = 0;

    // outer loop
    // #pragma unroll 4
    for (int cBase = 0; cBase < cPerSlice; cBase += cPerIter)
    {
        // load filters into shared memory
        #pragma unroll 
        for (int lk = 0; lk < kPerBlock; lk++)
        {
            int k = kStart + lk;
            #pragma unroll
            for (int i = 0; i < cPerIter*R*S / 32; i++)
            {
                int localIndex = 32 * i + threadInWarp;
                int sharedIndex = shStart + lk * (cPerIter*R*S) + localIndex;
                int globalIndex = k * (C*R*S) + (cStart + cBase) * (R*S) + localIndex;
                shData[sharedIndex] =  weight[globalIndex];
            }
        }

        #pragma unroll
        for (int lc = 0; lc < cPerIterPerThread; lc++)
        {
            int shc = cOffset + lc;     // offset of filter for index c in shared memory
            int c = cStart + cBase + shc;        // real c dimension

            // hardcoded for 3x3 filter, and 4x4 spatial elements per thread
            float inEl[hPerThread+2][wPerThread+2];
            #pragma unroll
            for (int y = 0; y < hPerThread+2; y++)
                #pragma unroll
                for (int x = 0; x < wPerThread+2; x++)
                    inEl[y][x] = 0.0f;


            // assume wPerThread == 4, and use a 128 bit reads
            *((uint4*)(&inEl[1][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart, wStart)]));
            *((uint4*)(&inEl[2][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart + 1, wStart)]));
            *((uint4*)(&inEl[3][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart + 2, wStart)]));
            *((uint4*)(&inEl[4][1])) = *((uint4*)(&input[INDEX_NCHW(n, c, hStart + 3, wStart)]));

            // need temps because shfl needs all threads in warp to participate
            float t01 = __shfl_up_sync(0xFFFFFFFF, inEl[4][1], 2);
            float t02 = __shfl_up_sync(0xFFFFFFFF, inEl[4][2], 2);
            float t03 = __shfl_up_sync(0xFFFFFFFF, inEl[4][3], 2);
            float t04 = __shfl_up_sync(0xFFFFFFFF, inEl[4][4], 2);
            if (hStart != 0)
            {
                inEl[0][1] = t01;
                inEl[0][2] = t02;
                inEl[0][3] = t03;
                inEl[0][4] = t04;
            }

            float t51 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 2);
            float t52 = __shfl_down_sync(0xFFFFFFFF, inEl[1][2], 2);
            float t53 = __shfl_down_sync(0xFFFFFFFF, inEl[1][3], 2);
            float t54 = __shfl_down_sync(0xFFFFFFFF, inEl[1][4], 2);
            if (hStart == 0)
            {
                inEl[5][1] = t51;
                inEl[5][2] = t52;
                inEl[5][3] = t53;
                inEl[5][4] = t54;
            }

            float t00 = __shfl_up_sync(0xFFFFFFFF, inEl[0][4], 1);
            float t10 = __shfl_up_sync(0xFFFFFFFF, inEl[1][4], 1);
            float t20 = __shfl_up_sync(0xFFFFFFFF, inEl[2][4], 1);
            float t30 = __shfl_up_sync(0xFFFFFFFF, inEl[3][4], 1);
            float t40 = __shfl_up_sync(0xFFFFFFFF, inEl[4][4], 1);
            float t50 = __shfl_up_sync(0xFFFFFFFF, inEl[5][4], 1);
            if (wStart != 0)
            {
                inEl[0][0] = t00;
                inEl[1][0] = t10;
                inEl[2][0] = t20;
                inEl[3][0] = t30;
                inEl[4][0] = t40;
                inEl[5][0] = t50;
            }

            float t05 = __shfl_down_sync(0xFFFFFFFF, inEl[0][1], 1);
            float t15 = __shfl_down_sync(0xFFFFFFFF, inEl[1][1], 1);
            float t25 = __shfl_down_sync(0xFFFFFFFF, inEl[2][1], 1);
            float t35 = __shfl_down_sync(0xFFFFFFFF, inEl[3][1], 1);
            float t45 = __shfl_down_sync(0xFFFFFFFF, inEl[4][1], 1);
            float t55 = __shfl_down_sync(0xFFFFFFFF, inEl[5][1], 1);
            if (wStart == 0)
            {
                inEl[0][5] = t05;
                inEl[1][5] = t15;
                inEl[2][5] = t25;
                inEl[3][5] = t35;
                inEl[4][5] = t45;
                inEl[5][5] = t55;
            }

            #pragma unroll
            for (int s = 0; s < S; s++)
            {
                #pragma unroll
                for (int r = 0; r < R; r++)
                {
                    #pragma unroll 
                    for (int lk = 0; lk < kPerBlock; lk++)
                    {
                        float wt = (float)(shData[shStart + SH_FILTER_IDX(lk, shc, s, r)]);
                        //float wt = weight[FILTER_IDX_NCHW(kStart + lk, c, s, r)];
                        #pragma unroll
                        for (int y = 0; y < hPerThread; y++)
                        {
                            #pragma unroll
                            for (int x = 0; x < wPerThread; x++)
                            {
                                op[lk][y][x] += inEl[y + s][x + r] * wt;
                            }   // x
                        }   // y
                    }   // k
                }   // r
            } // s
        } // lc
    }   // cBase


    #pragma unroll
    for (int y = 0; y < hPerThread; y++)
    {
        #pragma unroll
        for (int x = 0; x < wPerThread; x++)
        {
            // sum across C dimension
            #pragma unroll 
            for (int lk = 0; lk < kPerBlock; lk++)
            {
                op[lk][y][x] += __shfl_down_sync(0xFFFFFFFF, op[lk][y][x], 4);
                op[lk][y][x] += __shfl_down_sync(0xFFFFFFFF, op[lk][y][x], 8);
                op[lk][y][x] += __shfl_down_sync(0xFFFFFFFF, op[lk][y][x], 16);
            }
        }
    }


    //__shared__ float shResult[blockWidth][kPerBlock][hPerThread][wPerThread];
    static_assert(sizeof(shData) >= 2*sizeof(float)*blockWidth*kPerBlock*hPerThread*wPerThread, "shared mem not enough");

    if (threadIdx.y == 0 && threadIdx.z == 0)
    {
        #pragma unroll 
        for (int lk = 0; lk < kPerBlock; lk++)
            #pragma unroll 
            for (int y = 0; y < hPerThread; y++)
                #pragma unroll
                for (int x = 0; x < wPerThread; x++)
                {
                    //shResult[threadIdx.x][lk][y][x] = op[lk][y][x];
                    shData[threadIdx.x * kPerBlock * hPerThread * wPerThread + lk * hPerThread * wPerThread
                           + y * wPerThread + x] = op[lk][y][x];
                }
    }
    
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.z == 1)
    {
        float b[kPerBlock];
        #pragma unroll 
        for (int lk = 0; lk < kPerBlock; lk++)
            b[lk] = bias ? bias[kStart + lk] : 0;


        #pragma unroll
        for (int y = 0; y < hPerThread; y++)
        {
            // sum across C dimension
            #pragma unroll 
            for (int lk = 0; lk < kPerBlock; lk++)
            {
                // apply bias and relu
                #pragma unroll
                for (int x = 0; x < wPerThread; x++)
                {
                    //op[lk][y][x] += shResult[threadIdx.x][lk][y][x];
                    op[lk][y][x] += shData[threadIdx.x * kPerBlock * hPerThread * wPerThread + lk * hPerThread * wPerThread
                                           + y * wPerThread + x];

                    op[lk][y][x] += b[lk];

                    if (relu && op[lk][y][x] < 0)
                        op[lk][y][x] = 0;
                }

            }
        }

        // final memory write
        #pragma unroll 
        for (int lk = 0; lk < kPerBlock; lk++)
        {
            int k = kStart + lk;
            ((uint4 *)output)[INDEX_NCHW(n, k, hStart, wStart) >> 2] = *((uint4 *)&op[lk][0][0]);
            ((uint4 *)output)[INDEX_NCHW(n, k, hStart + 1, wStart) >> 2] = *((uint4 *)&op[lk][1][0]);
            ((uint4 *)output)[INDEX_NCHW(n, k, hStart + 2, wStart) >> 2] = *((uint4 *)&op[lk][2][0]);
            ((uint4 *)output)[INDEX_NCHW(n, k, hStart + 3, wStart) >> 2] = *((uint4 *)&op[lk][3][0]);
        }
    }
    
}
#endif


template<int N, int K, int C, int H, int W, int S, int R, typename T>
void convCuda(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    // (N * K * H * W) output elements? (N * 16384)
    // for each of them need to do (R * S * C) multiple-adds  (2304 - per output element)
    // need to re-use input and filter elements to avoid making everything memory bound

    // 1. simple strategy
    // N * K blocks
    // H * W threads per block

    dim3 gridDim(K / kPerBlock, N);
    dim3 blockDim(blockWidth, blockHeight, blockDepth);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    for (int i = 0; i < loops * 2; i++)
    {
        if (i == loops)
            cudaEventRecord(start, NULL);

        convKernel<N, K, C, H, W, S, R> << <gridDim, blockDim >> > (output, input, weight, bias, relu);
    }

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    double TFlops = (2.0 * N * W * H * K * C * S * R * loops) / (msecTotal * 1000000000.0);
    printf("CUDA TFlops: %g\n\n", TFlops);

}


int main()
{
    // cudaSetDevice(1);

    float ip[2 * 2] =  {
        5, 7,
        11, 2
    };

    float fl[3 * 3] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    float op[2 * 2];


    convRef<1, 1, 1, 2, 2, 3, 3, float>(op, ip, fl, nullptr, false);
    printf("ref output: %f, %f,%f, %f\n", op[0], op[1], op[2], op[3]);

    convRef_Winograd2x2<1, 1, 1, 2, 2, 3, 3, float>(op, ip, fl, nullptr, false);
    printf("win output: %f, %f,%f, %f\n", op[0], op[1], op[2], op[3]);


    constexpr bool fp16 = false;

    constexpr int N = 1;
    constexpr int C = 256;
    constexpr int K = 256;
    constexpr int H = 8;
    constexpr int W = 8;
    constexpr int F = 3;

    size_t elementSize = fp16 ? sizeof(half) : sizeof(float);
    size_t inputElements = N*C*H*W;
    size_t outputElements = N*K*H*W;
    size_t filterElements = K*C*F*F;
    size_t biasElements = K;

    size_t inputBytes = inputElements * elementSize;
    size_t outputBytes = outputElements * elementSize;
    size_t filterBytes = filterElements * elementSize;
    size_t biasBytes = biasElements * elementSize;

    void *cinput = malloc(inputBytes);
    void *coutput = malloc(outputBytes);
    void *cfilter = malloc(filterBytes);
    void *cbias = malloc(biasBytes);

    fillRandomArray(cinput, inputElements, fp16);
    fillRandomArray(cfilter, filterElements, fp16);
    fillRandomArray(cbias, biasElements, fp16);


    void *input;
    void *output;
    void *filter;
    void *bias;

    cudaMalloc(&input, inputBytes);
    cudaMalloc(&output, outputBytes);
    cudaMalloc(&filter, filterBytes);
    cudaMalloc(&bias, biasBytes);

    cudaMemcpy(input, cinput, inputBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(filter, cfilter, filterBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bias, cbias, biasBytes, cudaMemcpyHostToDevice);


    // convolution using cpu ref
    void *crefop = malloc(outputBytes);
#if 1
    if (fp16)
    {
        convRef<N, K, C, H, W, F, F>((half*)crefop, (half*)cinput, (half*)cfilter, (half*)cbias, true);
    }
    else
    {
        //convRef<N, K, C, H, W, F, F>((float*)crefop, (float*)cinput, (float*)cfilter, (float*)cbias, true);
        //convRef_Winograd4x4<N, K, C, H, W, F, F>((float*)crefop, (float*)cinput, (float*)cfilter, (float*)cbias, true);
        convTest_Winograd4x4Matmul<N, K, C, H, W, F, F>((float*)crefop, (float*)cinput, (float*)cfilter, (float*)cbias, true);
    }
#endif

#if 1
    // convolution using cudnn
    if (fp16)
    {
        cudnnConvTest<N, K, C, H, W, F, F>((half*)output, (half*)cinput, (half*)cfilter, (half*)cbias, true);
    }
    else
    {
        cudnnConvTest<N, K, C, H, W, F, F>((float*)output, (float*)input, (float*)filter, (float*)bias, true);
    }
#endif

#if 1
    // convolution using our own Cuda C kernel
    if (fp16)
    {
        convCuda<N, K, C, H, W, F, F>((half*)output, (half*)input, (half*)filter, (half*)bias, true);
    }
    else
    {
        convCuda<N, K, C, H, W, F, F>((float*)output, (float*)input, (float*)filter, (float*)bias, true);
    }
#endif


    cudaMemcpy(coutput, output, outputBytes, cudaMemcpyDeviceToHost);


    compareResults(coutput, crefop, outputElements, fp16);

    cudaFree(input);
    cudaFree(output);
    cudaFree(filter);
    cudaFree(bias);

    free(cinput);
    free(coutput);
    free(cfilter);
    free(cbias);
    return 0;
}

