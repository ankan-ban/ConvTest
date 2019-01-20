#include "utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cudnn.h>

#include <cstdio>
#include <cstdlib>

constexpr int loops = 1000;


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
                            for (int r = 0; r < S; r++)
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


template<int N, int K, int C, int H, int W, int S, int R, typename T>
void cudnnConvTest(T *output, T *input, T *filter, T *bias, bool relu)
{
    bool fp16 = (sizeof(T) == sizeof(half));
    const cudnnDataType_t datatype = fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
    const cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

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
        format,
        datatype,
        N, C,
        H, W);

    status = cudnnSetFilter4dDescriptor(filterDesc,
        datatype,
        format,
        K,
        C,
        S,
        R);

    status = cudnnSetTensor4dDescriptor(biasDesc, format, datatype, 1, K, 1, 1);

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
        format,
        datatype,
        n, c,
        h, w);

    convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    if (fp16)
    {
        status = cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);
    }
    else
    {
        convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
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
    printf("TFlops: %g\n\n", TFlops);

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



int main()
{
    constexpr bool fp16 = false;

    constexpr int N = 128;
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

    fillRandomArray(cinput, inputElements);
    fillRandomArray(cfilter, filterElements);
    fillRandomArray(cbias, biasElements);


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


    // convolution using cudnn
    cudnnConvTest<N, K, C, H, W, F, F>((float*)output, (float*)input, (float*)filter, (float*) bias, true);
    cudaMemcpy(coutput, output, outputBytes, cudaMemcpyDeviceToHost);

    // convolution using cpu ref
    void *crefop = malloc(outputBytes);
    convRef<N, K, C, H, W, F, F>((float*)crefop, (float*)cinput, (float*)cfilter, (float*)cbias, true);

    compareResults(coutput, crefop, outputElements);

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

