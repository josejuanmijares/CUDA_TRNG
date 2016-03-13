// filename: vadd.cu
// a simple CUDA kernel to add two vectors

extern "C"   // ensure function name to be exactly "vadd"
{
    __global__ void vadd(const float *a, const float *b, float *c)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        c[i] = a[i] + b[i];
    }
}
