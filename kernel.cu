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

        if (h != 0)
            in = (float)(input[INDEX_NCHW(n, c, h - 1, w)]);

        if (h != 7)
            is = (float)(input[INDEX_NCHW(n, c, h + 1, w)]);

#if 0
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


#if 1
// get some spatial reuse for input tensor using shfl
// assumes S == R == 3!
// and also H == W == 8

// also do multiple elements in K dimension per thread
constexpr int kPerThread = 2;

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

        if (h != 0)
            in = (float)(input[INDEX_NCHW(n, c, h - 1, w)]);

        if (h != 7)
            is = (float)(input[INDEX_NCHW(n, c, h + 1, w)]);

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



template<int N, int K, int C, int H, int W, int S, int R, typename T>
void convCuda(T *output, const T *input, const T *weight, const T *bias, bool relu)
{
    // (N * K * H * W) output elements? (N * 16384)
    // for each of them need to do (R * S * C) multiple-adds  (2304 - per output element)
    // need to re-use input and filter elements to avoid making everything memory bound

    // 1. simple strategy
    // N * K blocks
    // H * W threads per block

    dim3 gridDim(K/kPerThread, N);
    dim3 blockDim(W, H);

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
        convCuda<N, K, C, H, W, F, F>((half*)output, (half*)cinput, (half*)cfilter, (half*)cbias, true);
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

