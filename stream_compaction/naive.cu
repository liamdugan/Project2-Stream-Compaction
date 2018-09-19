#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 512

        __global__ void kernNaiveScan(int n, int twoToPowerDMinusOne, float* odata, float* idata)
        {
          // get index first
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          if (index >= n)
          {
            return;
          }

          // then add the two numbers and put them into the global output buffer
 //         if (index >= twoToPowerDMinusOne)
 //         {
 //           odata[index] = idata[index - twoToPowerDMinusOne] + idata[index];
 //         }
 //         else
 //         {
 //           odata[index] = idata[index];
 //         }
        }

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
          timer().startGpuTimer();
          dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

          float* dev_gpuScanBuf;
          float* dev_idata;

          int nNextHighestPowTwo = 1 << ilog2ceil(n);

          cudaMalloc((void**)&dev_gpuScanBuf, nNextHighestPowTwo * sizeof(float));
          checkCUDAError("cudaMalloc buf failed");

          cudaMalloc((void**)&dev_idata, nNextHighestPowTwo * sizeof(float));
          checkCUDAError("cudaMalloc idata failed");

          cudaMemcpy((void*)dev_idata, (const void*)idata, nNextHighestPowTwo * sizeof(float), cudaMemcpyHostToDevice);
          checkCUDAError("cudaMemcpy idata failed");

          // call the kernel log2n number of times
          bool flipped = false;
          for (int i = 0; i < ilog2ceil(nNextHighestPowTwo); ++i)
          {
            // call the kernel
            int twoToPowerIMinusOne = 1 << (i - 1);
            std::cout << ((n + blockSize - 1) / blockSize) << ", " << blockSize << std::endl;
            kernNaiveScan<<<((n + blockSize - 1) / blockSize) , blockSize>>>(nNextHighestPowTwo, twoToPowerIMinusOne, dev_gpuScanBuf, dev_idata);

            // flip flop the buffers and keep track with a boolean (flipped = true means dev_idata has the latest data)
            float* temp = dev_gpuScanBuf;
            dev_gpuScanBuf = dev_idata;
            dev_idata = temp;
            flipped = !flipped;
          }

          if (flipped)
          {
            cudaMemcpy(odata, dev_idata, nNextHighestPowTwo * sizeof(float), cudaMemcpyDeviceToHost);
          }
          else
          {
            cudaMemcpy(odata, dev_gpuScanBuf, nNextHighestPowTwo * sizeof(float), cudaMemcpyDeviceToHost);
          }

          cudaFree(dev_gpuScanBuf);
          cudaFree(dev_idata);
          timer().endGpuTimer();
        }
    }
}
