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


#if 1
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

            
            inEl[1][1] = (float)(input[INDEX_NCHW(n, c, hStart, wStart)]);
            inEl[1][2] = (float)(input[INDEX_NCHW(n, c, hStart, wStart+1)]);
            inEl[2][1] = (float)(input[INDEX_NCHW(n, c, hStart+1, wStart)]);
            inEl[2][2] = (float)(input[INDEX_NCHW(n, c, hStart+1, wStart+1)]);

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
            // output[INDEX_NCHW(n, k, hStart+y, wStart+x)] = (T)op[y][x];
        }
    }

    ((uint2 *)output)[INDEX_NCHW(n, k, hStart, wStart) >> 1] = *((uint2 *)&op[0][0]);
    ((uint2 *)output)[INDEX_NCHW(n, k, hStart+1, wStart) >> 1] = *((uint2 *)&op[1][0]);

}
#endif

// quite a bit slower!
// (too many registers per thread?)
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

        #pragma unroll
        for (int y = 0; y < hPerThread; y++)
            #pragma unroll
            for (int x = 0; x < wPerThread; x++)
            {
                inEl[y+1][x+1] = (float)(input[INDEX_NCHW(n, c, hStart+y, wStart+x)]);
            }

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

            // TODO: consider 64-bit writes
            // tried below - doesn't make any difference at all! (or maybe about 2% slower)

            if (threadIdx.y == 0)
                output[INDEX_NCHW(n, k, hStart+y, wStart+x)] = (T)op[y][x];
        }
    }

    //((uint2 *)output)[INDEX_NCHW(n, k, hStart, wStart) >> 1] = *((uint2 *)&op[0][0]);
    //((uint2 *)output)[INDEX_NCHW(n, k, hStart+1, wStart) >> 1] = *((uint2 *)&op[1][0]);

}
#endif

// significantly slower than 2x2 boards per thread block! ??
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


template<int N, int K, int C, int H, int W, int S, int R, typename T>
void convCuda(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    // (N * K * H * W) output elements? (N * 16384)
    // for each of them need to do (R * S * C) multiple-adds  (2304 - per output element)
    // need to re-use input and filter elements to avoid making everything memory bound

    // 1. simple strategy
    // N * K blocks
    // H * W threads per block

    dim3 gridDim(K/kPerBlock, N);
    dim3 blockDim(blockWidth, blockHeight);

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
    printf("TFlops: %g\n\n", TFlops);

}

int main()
{
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
        convRef<N, K, C, H, W, F, F>((float*)crefop, (float*)cinput, (float*)cfilter, (float*)cbias, true);
    }
#endif

#if 0
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

