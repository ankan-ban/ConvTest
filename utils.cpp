#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>


int divUp(int a, int b)
{
    return (a + b - 1) / b;
}


void compareResults(void *arr1, void *arr2, int size, bool testFp16)
{
    double maxError = 0;
    double totalError = 0;
    float max_err_a = 0, max_err_b = 0;
    int max_err_index = 0;
    printf("\nFirst few elements: ");
    for (int i = 0; i < size; i++)
    {
        float a, b;
        if (testFp16)
        {
            a = (float)(((half*)arr1)[i]);
            b = (float)(((half*)arr2)[i]);
        }
        else
        {
            a = ((float*)arr1)[i];
            b = ((float*)arr2)[i];
        }

        float error = fabs(a - b);
        float bigger = fabs(std::fmax(a, b));
        double percentError = error;
        if (bigger)
            percentError /= bigger;

        if (i < 20)
        {
            printf("\n%04d:  %12.8f, %12.8f, .... %11.8f", i, a, b, percentError*100);
        }

        if (percentError > maxError)
        {
            maxError = percentError;
            max_err_a = a;
            max_err_b = b;
            max_err_index = i;
        }
        totalError += percentError;
    }

    double avgError = totalError / size;
    avgError *= 100;
    maxError *= 100;

    printf("\nMax error: %f, avg error: %f, max error pair:", maxError, avgError);
    printf("\n%04d:  %12.8f, %12.8f\n", max_err_index, max_err_a, max_err_b);
}

void fillRandomArray(void *out, int size, bool testFp16)
{
    if (testFp16)
    {
        half *arr = (half *)out;

        for (int i = 0; i < size; i++)
        {
            // fill between 0 and 1
            arr[i] = (half)(((float)(rand())) / RAND_MAX);
        }
    }
    else
    {
        float *arr = (float *)out;

        for (int i = 0; i < size; i++)
        {
            // fill between 0 and 1
            arr[i] = ((float)(rand())) / RAND_MAX;
            //arr[i] = (float)(rand() % 2);
            //arr[i] = 0;
        }
    }
}

