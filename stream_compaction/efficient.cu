
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

int* dev_efficientScanBuf;
int* dev_efficientIdata;
int* dev_efficientBools;
int* dev_efficientIndices;

__global__ void kernEfficientScanUpSweep(int n, int d, int* odata, int* idata)
{
  // get index first
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  int twoToPowDPlusOne = 1 << (d + 1);
  if (index >= n || index % twoToPowDPlusOne != 0)
  {
    return;
  }
  
  int twoToPowD = 1 << d;

  // then add the two numbers and put them into the global output buffer
  odata[index + twoToPowDPlusOne - 1] = idata[index + twoToPowDPlusOne - 1] + idata[index + twoToPowD - 1];
}

__global__ void kernSetFirstElementZero(int n, int* odata)
{
  odata[n - 1] = 0;
}

__global__ void kernEfficientScanDownSweep(int n, int d, int* odata, int* idata)
{
  // get index first
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  int twoToPowDPlusOne = 1 << (d + 1);
  if (index >= n || (index % twoToPowDPlusOne != 0))
  {
    return;
  }
  
  int twoToPowD = 1 << d;

  // then sweep down
  odata[index + twoToPowD - 1] = idata[index + twoToPowDPlusOne - 1];
  odata[index + twoToPowDPlusOne - 1] = idata[index + twoToPowDPlusOne - 1] + idata[index + twoToPowD - 1];
}

namespace StreamCompaction {
    namespace Efficient {
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
          int nNextHighestPowTwo = 1 << ilog2ceil(n);

          cudaMalloc((void**)&dev_efficientScanBuf, nNextHighestPowTwo * sizeof(int));
          checkCUDAError("cudaMalloc buf failed");

          cudaMalloc((void**)&dev_efficientIdata, nNextHighestPowTwo * sizeof(int));
          checkCUDAError("cudaMalloc idata failed");

          timer().startGpuTimer();

          cudaMemcpy((void*)dev_efficientIdata, (const void*)idata, nNextHighestPowTwo * sizeof(int), cudaMemcpyHostToDevice);
          checkCUDAError("cudaMemcpy idata failed");

          // call the upsweep kernel log2n number of times
          for (int d = 0; d < ilog2ceil(nNextHighestPowTwo); ++d)
          {

            // copy all the data to make sure everythings in place
            cudaMemcpy((void*)dev_efficientScanBuf, (const void*)dev_efficientIdata, nNextHighestPowTwo * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy idata failed");

            // call the kernel
            kernEfficientScanUpSweep<<<((nNextHighestPowTwo + blockSize - 1) / blockSize) , blockSize>>>(nNextHighestPowTwo, d, dev_efficientScanBuf, dev_efficientIdata);

            // flip flop the buffers so that idata is always the most recent data
            int* temp = dev_efficientScanBuf;
            dev_efficientScanBuf = dev_efficientIdata;
            dev_efficientIdata = temp;
          }

          // set first element to be zero in a new kernel (unsure how to do this otherwise)
          kernSetFirstElementZero << <1, 1 >> > (nNextHighestPowTwo, dev_efficientIdata);

          // now call the downsweep kernel log2n times
          for (int d = (ilog2ceil(nNextHighestPowTwo) - 1); d >= 0; --d)
          {
            // copy all the data to make sure everything is in place
            cudaMemcpy((void*)dev_efficientScanBuf, (const void*)dev_efficientIdata, nNextHighestPowTwo * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy idata failed");

            // call the kernel
            kernEfficientScanDownSweep<<<((nNextHighestPowTwo + blockSize - 1) / blockSize) , blockSize>>>(nNextHighestPowTwo, d, dev_efficientScanBuf, dev_efficientIdata);

            // flip flop the buffers 
            int* temp = dev_efficientScanBuf;
            dev_efficientScanBuf = dev_efficientIdata;
            dev_efficientIdata = temp;
          }

          // shift it and memcpy to out
          cudaMemcpy(odata, dev_efficientIdata, nNextHighestPowTwo * sizeof(int), cudaMemcpyDeviceToHost);

          timer().endGpuTimer();

          cudaFree(dev_efficientScanBuf);
          cudaFree(dev_efficientIdata);
           

        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int nNextHighestPowTwo = 1 << ilog2ceil(n);

            cudaMalloc((void**)&dev_efficientBools, nNextHighestPowTwo * sizeof(int));
            checkCUDAError("cudaMalloc bool buf failed");

            cudaMalloc((void**)&dev_efficientScanBuf, nNextHighestPowTwo * sizeof(int));
            checkCUDAError("cudaMalloc buf failed");

            cudaMalloc((void**)&dev_efficientIdata, nNextHighestPowTwo * sizeof(int));
            checkCUDAError("cudaMalloc idata failed");

            cudaMalloc((void**)&dev_efficientIndices, nNextHighestPowTwo * sizeof(int));
            checkCUDAError("cudaMalloc indices failed");

            // memcpy all the stuff over to gpu before calling kernel functions
            cudaMemcpy((void*)dev_efficientIdata, (const void*)idata, nNextHighestPowTwo * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed");

            timer().startGpuTimer();

            // map all of the values to booleans (and pad with zeroes for those values higher than original array limit)
            StreamCompaction::Common::kernMapToBoolean<< <((nNextHighestPowTwo + blockSize - 1) / blockSize), blockSize >> > (n, nNextHighestPowTwo, dev_efficientBools, dev_efficientIdata);

            // Start the scan --------------- (copy pasted from the scan function because you can't nest calls to timer. Plus it saves a copy from device to host)

            // make a copy of the bools so we can do the scan and put it into indices
            cudaMemcpy((void*)dev_efficientIndices, (const void*)dev_efficientBools, nNextHighestPowTwo * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy idata failed");

            // call the upsweep kernel log2n number of times
            for (int d = 0; d < ilog2ceil(nNextHighestPowTwo); ++d)
            {
              // copy all the data to make sure everythings in place
              cudaMemcpy((void*)dev_efficientScanBuf, (const void*)dev_efficientIndices, nNextHighestPowTwo * sizeof(int), cudaMemcpyDeviceToDevice);
              checkCUDAError("cudaMemcpy idata failed");
              
              // call the kernel
              kernEfficientScanUpSweep << <((nNextHighestPowTwo + blockSize - 1) / blockSize), blockSize >> > (nNextHighestPowTwo, d, dev_efficientScanBuf, dev_efficientIndices);
              
              // flip flop the buffers so that idata is always the most recent data
              int* temp = dev_efficientScanBuf;
              dev_efficientScanBuf = dev_efficientIndices;
              dev_efficientIndices = temp;
            }
            
            // set first element to be zero in a new kernel (unsure how to do this otherwise)
            kernSetFirstElementZero << <1, 1 >> > (nNextHighestPowTwo, dev_efficientIndices);
            
            // now call the downsweep kernel log2n times
            for (int d = (ilog2ceil(nNextHighestPowTwo) - 1); d >= 0; --d)
            {
              // copy all the data to make sure everythings in place
              cudaMemcpy((void*)dev_efficientScanBuf, (const void*)dev_efficientIndices, nNextHighestPowTwo * sizeof(int), cudaMemcpyDeviceToDevice);
              checkCUDAError("cudaMemcpy idata failed");
              
              // call the kernel
              kernEfficientScanDownSweep << <((nNextHighestPowTwo + blockSize - 1) / blockSize), blockSize >> > (nNextHighestPowTwo, d, dev_efficientScanBuf, dev_efficientIndices);
              
              // flip flop the buffers
              int* temp = dev_efficientScanBuf;
              dev_efficientScanBuf = dev_efficientIndices;
              dev_efficientIndices = temp;
            }

            // ------- end of scan

            int sizeOfCompactedStream = 0;
            // memcpy the final value of indices to out so that we can get the total size of compacted stream
            cudaMemcpy(&sizeOfCompactedStream, dev_efficientIndices + (nNextHighestPowTwo - 1), 1 * sizeof(int), cudaMemcpyDeviceToHost);

            // run the stream compaction
            StreamCompaction::Common::kernScatter << <((nNextHighestPowTwo + blockSize - 1) / blockSize), blockSize >> > (nNextHighestPowTwo, dev_efficientScanBuf, dev_efficientIdata, dev_efficientBools, dev_efficientIndices);

            // memcpy to out
            cudaMemcpy(odata, dev_efficientScanBuf, sizeOfCompactedStream * sizeof(int), cudaMemcpyDeviceToHost);

            timer().endGpuTimer();

            // free all our stuff
            cudaFree(dev_efficientScanBuf);
            cudaFree(dev_efficientBools);
            cudaFree(dev_efficientIdata);
            cudaFree(dev_efficientIndices);

            // return the total size of the compacted stream
            return sizeOfCompactedStream;
        }
    }
}
