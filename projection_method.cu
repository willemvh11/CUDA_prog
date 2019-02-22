#include <fstream>
#include <iostream>
#include <math.h>
#include <cmath>
#include <curand_kernel.h>
#include <cuda.h>
#include <string>
#include <time.h>


__device__ void rot( float *w, float *vec, const float dt)
{
    float mw = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    // Be very careful with this check if you change the code as mw has dimensions
    // i.e. technically this approximation is only valid if mw*t is very small but since
    // mw is much smaller than t in this code, this shouldn't be an issue
    if (mw<e-6)
    {
        float dot = w[0]*vec[0] + w[1]*vec[1] + w[2]*vec[2];
        float i1[3];
        i1[0] = w[0]*dot;
        i1[1] = w[1]*dot;
        i1[2] = w[2]*dot;
        float i3[3];
        i3[0] = w[1]*vec[2] - w[2]*vec[1];
        i3[1] = w[2]*vec[0] - w[0]*vec[2];
        i3[2] = w[0]*vec[1] - w[1]*vec[0];
        float pref2 = 1.0 - (mw*dt)*(mw*dt)/2.0;
        float pref1 = dt*dt/2.0;
        vec[0] = pref1*i1[0] + pref2*vec[0] + dt*i3[0];
        vec[1] = pref1*i1[1] + pref2*vec[1] + dt*i3[1];
        vec[2] = pref1*i1[2] + pref2*vec[2] + dt*i3[2];
    } else {
        float omega[3];
        float invmw = 1.0f/mw;
        omega[0] = w[0]*invmw;
        omega[1] = w[1]*invmw;
        omega[2] = w[2]*invmw;
        float dot = omega[0]*vec[0] + omega[1]*vec[1] + omega[2]*vec[2];
        float i1[3];
        i1[0] = omega[0]*dot;
        i1[1] = omega[1]*dot;
        i1[2] = omega[2]*dot;
        float i2[3];
        i2[0] = vec[0] - i1[0];
        i2[1] = vec[1] - i1[1];
        i2[2] = vec[2] - i1[2];
        float i3[3];
        i3[0] = omega[1]*vec[2] - omega[2]*vec[1];
        i3[1] = omega[2]*vec[0] - omega[0]*vec[2];
        i3[2] = omega[0]*vec[1] - omega[1]*vec[0];
        float cwt =cos(mw*dt);
        float swt =sin(mw*dt);
        vec[0] = i1[0] + i2[0]*cwt + i3[0]*swt;
        vec[1] = i1[1] + i2[1]*cwt + i3[1]*swt;
        vec[2] = i1[2] + i2[2]*cwt + i3[2]*swt;
    }
}

//-----------------------------------------------------------------------------

__global__ void precessnucspins(float *i, float *a, float *s, const int ni, const float dt)
{
    extern __shared__ float iloc[];
    int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int ggid1 = (blockIdx.y * blockDim.y) + threadIdx.y;
    int glid = threadIdx.x;
    int glid1 = threadIdx.y;
    int groupid = blockIdx.x;
    int nl = blockDim.x;
    float w[3];
    float store = 0;
    int sind = 3*nl*groupid + glid;
    for(int ii = 0; ii < 3 && sind < 3*ni; ++ii, sind += nl)
    {
        iloc[glid + ii*nl + 3*nl*glid1] = i[sind + 3*ni*ggid1];
    }
    __syncthreads();
    if (ggid < ni)
    {
        store = a[ggid];
     
        w[0] = store*s[3*ggid1];
        w[1] = store*s[1 + 3*ggid1];
        w[2] = store*s[2 + 3*ggid1];

        rot (w, iloc+(3*glid+3*nl*glid1), dt);
    }
    __syncthreads();
    int wind = 3*nl*groupid + glid;
    for(int ii = 0; ii < 3 && wind < 3*ni; ++ii, wind += nl)
    {
        i[wind + 3*ni*ggid1] = iloc[glid + 3*nl*glid1 + ii*nl];
    }
}


//----------------------------------------------------------------------------- 

__global__ void setup_rand(curandState *state, unsigned long seed, const int mcs)
{
    unsigned ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init(seed, ggid, 4*mcs*ggid, &state[ggid]);
}

//----------------------------------------------------------------------------- 

__global__ void vecbuilds(float *s, float *sinit, curandState *state)
{
    extern __shared__ float sloc[];
    int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int nl = blockDim.x;
    int glid = threadIdx.x;
    int groupid = blockIdx.x;
    float v = curand_uniform(&state[ggid]);
    float g = curand_uniform(&state[ggid]);
    float m = sqrt(3.0f/4.0f);
    float phi = 2.0*M_PI*v;
    float th = acos(cbrtf(2.0*g - 1.0));
    sloc[3*glid] = m*sin(th)*cos(phi);
    sloc[3*glid + 1] = m*sin(th)*sin(phi);
    sloc[3*glid + 2] = m*cos(th);
    __syncthreads();
    sinit[ggid] = sloc[3*glid + 2];
    int wind = 3*nl*groupid + glid;
    for(int ii = 0; ii < 3; ++ii, wind += nl)
    {
        s[wind] = sloc[glid + ii*nl];
    }
}

//----------------------------------------------------------------------------- 

__global__ void vecbuildi(float *i, curandState *state, const int ni, const int nindium)
{
    extern __shared__ float iloc[];
    int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int ggid1 = (blockIdx.y * blockDim.y) + threadIdx.y;
    int nl = blockDim.x;
    int glid = threadIdx.x;
    int glid1 = threadIdx.y;
    int groupid = blockIdx.x;
    int ng = nl*gridDim.x; 
    float m = 0;
    if (ggid < nindium){ 
        m = sqrt(99.0f/4.0f);
    } else {
        m = sqrt(15.0f/4.0f);
    }
    float v = curand_uniform(&state[ggid + ggid1*ng]);
    float phi = 2.0f*M_PI*v;
    float g = curand_uniform(&state[ggid + ggid1*ng]);
    float th = acos(2.0f*g - 1.0f);
    iloc[3*glid + 3*nl*glid1] = m*sin(th)*cos(phi);
    iloc[3*glid + 3*nl*glid1 + 1] = m*sin(th)*sin(phi);
    iloc[3*glid + 3*nl*glid1 + 2] = m*cos(th);
    __syncthreads();
    int wind = 3*nl*groupid + glid;
    for(int ii = 0; ii < 3 && wind < 3*ni; ++ii, wind += nl)
    {
        i[wind + 3*ni*ggid1] = iloc[glid + 3*nl*glid1 + ii*nl];
    }
}

//----------------------------------------------------------------------------- 

__global__ void reduce(float *i, float *w, const int n, const int a, float *hyp, 
const int ni, float *wout, const int count, const int size)
{
    extern __shared__ float locmem[];
    float* store = locmem;
    float* wtemp = store + blockDim.x;
    int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int ggid1 = (blockIdx.y * blockDim.y) + threadIdx.y;
    int glid = threadIdx.x;
    int glid1 = threadIdx.y;
    int nl = blockDim.x;
    int ng = nl*gridDim.x;
    int groupid = blockIdx.x;
    if (count == 0) 
    {
        if (ggid < ni)
        {
            store[glid] = hyp[ggid];
        }
        int sind = 3*nl*groupid + glid;
        for(int ii = 0; ii < 3 && sind < 3*ni; ++ii, sind += nl)
        {
            w[sind + 3*ng*ggid1] = i[sind + 3*ni*ggid1];
        }
        __syncthreads();

        wtemp[glid + 3*nl*glid1] = store[(glid - glid%3)/3]*w[3*nl*groupid + glid + 3*ng*ggid1];
        wtemp[glid + nl + 3*nl*glid1] = store[(glid + nl - (glid + nl)%3)/3]*w[3*nl*groupid + glid + nl + 3*ng*ggid1];
        wtemp[glid + 2*nl + 3*nl*glid1] = store[(glid + 2*nl - (glid + 2*nl)%3)/3]*w[3*nl*groupid + glid + 2*nl + 3*ng*ggid1];
    
    } else {
        wtemp[glid + 3*nl*glid1] = w[3*nl*groupid + glid + 3*ng*ggid1];
        wtemp[glid + nl + 3*nl*glid1] = w[3*nl*groupid + glid + nl + 3*ng*ggid1];
        wtemp[glid + 2*nl + 3*nl*glid1] = w[3*nl*groupid + glid + 2*nl + 3*ng*ggid1];
    }
    #pragma unroll
    for (int k=1; k < n; k++)
    {
        __syncthreads();
        int b = nl >> k;
        if (glid < b) 
        {
            wtemp[3*glid + 3*nl*glid1] += wtemp[3*(glid + b)+ 3*nl*glid1];
            wtemp[3*glid + 1 + 3*nl*glid1] += wtemp[3*(glid + b) + 1+ 3*nl*glid1];
            wtemp[3*glid + 2 + 3*nl*glid1] += wtemp[3*(glid + b) + 2+ 3*nl*glid1];
        }
    }
    __syncthreads();
    
    if (glid == 0) 
    {
        wout[(ggid >> n)*3 + 3*size*ggid1] = wtemp[3*nl*glid1] + wtemp[3 + 3*nl*glid1];
        wout[(ggid >> n)*3 + 1 + 3*size*ggid1] = wtemp[1 + 3*nl*glid1] + wtemp[4 + 3*nl*glid1];
        wout[(ggid >> n)*3 + 2 + 3*size*ggid1] = wtemp[2 + 3*nl*glid1] + wtemp[5 + 3*nl*glid1];
    }    
    int c = 0;
    if (a%nl == 0)
    {
        c = a/nl;
    }
    else
    {
        c = a/nl + 1;
    }
    __syncthreads();
    
    int wind = 3*nl*groupid + glid;
    for(int ii = 0; ii < 3 && wind < 3*size; ++ii, wind += nl)
    {
        if (wind >= 3*c)
        {   
            wout[wind + 3*size*ggid1] = 0;
        }
    }
}

//----------------------------------------------------------------------------- 

__global__ void precesselecspins(float *w, float *wi, float *s, const int size, const int x, 
float *sstore, const int a, const float dt, float *wfin)
{
    extern __shared__ float locmem[];
    float* sloc = locmem;
    float* wloc = sloc + 3*blockDim.x;
    float wtemp[3];
    int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int glid = threadIdx.x;
    int nl = blockDim.x;
    int groupid = blockIdx.x;
    int sind = 3*nl*groupid + glid;
    for(int ii = 0; ii < 3; ++ii, sind += nl)
    {
        sloc[glid + ii*nl] = s[sind];
    }
    __syncthreads();
    wloc[3*glid] = w[3*a*ggid];
    wloc[3*glid + 1] = w[3*a*ggid + 1];
    wloc[3*glid + 2] = w[3*a*ggid + 2];
    sstore[x + size*ggid] = sloc[3*glid + 2];
    wtemp[0] = wloc[3*glid] + wi[0];
    wtemp[1] = wloc[1 + 3*glid] + wi[1];
    wtemp[2] = wloc[2 + 3*glid] + wi[2];
    rot (wtemp, sloc+(3*glid), dt);
    float len = sqrt(wtemp[0]*wtemp[0] + wtemp[1]*wtemp[1] + wtemp[2]*wtemp[2]);
    wtemp[0] = wtemp[0]/len;
    wtemp[1] = wtemp[1]/len;
    wtemp[2] = wtemp[2]/len;
    float dot = wtemp[0]*sloc[3*glid] + wtemp[1]*sloc[3*glid + 1] + wtemp[2]*sloc[3*glid + 2];
    wloc[3*glid] = dot*wtemp[0];
    wloc[3*glid + 1] = dot*wtemp[1];
    wloc[3*glid + 2] = dot*wtemp[2];
    __syncthreads();
    int wind = 3*nl*groupid + glid;
    for(int ii = 0; ii < 3; ++ii, wind += nl)
    {
        s[wind] = sloc[glid + ii*nl];
    }
    wind = 3*nl*groupid + glid;
    for(int ii = 0; ii < 3; ++ii, wind += nl)
    {
        wfin[wind] = wloc[glid + ii*nl];
    }
}

//-----------------------------------------------------------------------------

__global__ void prep2(float *sstore, float *output, const int size, float *sinit)
{
    extern __shared__ float loc[];
    int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int ggid1 = (blockIdx.y * blockDim.y) + threadIdx.y;
    int glid = threadIdx.x;
    int glid1 = threadIdx.y;
    int nl = blockDim.x;
    int ng = nl*gridDim.x;
    float store = sinit[ggid1];
    loc[glid + nl*glid1] = sstore[ggid + size*ggid1];
    loc[glid + nl*glid1] = loc[glid + nl*glid1]/store;
    output[ggid + ng*ggid1] = loc[glid + nl*glid1];
} 

//-----------------------------------------------------------------------------

__global__ void reduce2(const int n, const int a, float *output, float *out)
{
    extern __shared__ float sstoretemp[];
    int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int ggid1 = (blockIdx.y * blockDim.y) + threadIdx.y;
    int glid = threadIdx.x;
    int glid1 = threadIdx.y;
    int nl1 = blockDim.y;
    int nl = blockDim.x;
    int ng = nl*gridDim.x;
    sstoretemp[glid + nl*glid1] = output[ggid + ng*ggid1];
    #pragma unroll
    for (int k=1; k < n; k++)
    {
        __syncthreads();
        int b = nl1 >> k;
        if (glid1 < b) 
        {
            sstoretemp[glid + nl*glid1] += sstoretemp[glid+ nl*(glid1+b)];
        }
    }
    __syncthreads();
    if (glid1 == 0) 
    {
        out[ggid + ng*(ggid1 >> n)] = sstoretemp[glid] + sstoretemp[glid + nl];
    }
    int c = 0;
    if (a%nl1 == 0)
    {
        c = a/nl1;
    }
    else
    {
        c = a/nl1 + 1;
    }
    if (ggid1 > c) 
    {
        out[ggid + ng*ggid1] = 0;
    }
}

//-----------------------------------------------------------------------------

__global__ void tensors(float *output, float *Rzz, const int size, const int j)
{
     int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
     if (ggid < size) 
     {
        Rzz[ggid + j*size] = Rzz[ggid + j*size] + output[ggid];
        
     }
     
}

//-----------------------------------------------------------------------------

__global__ void final(float *Rzz, const int mcs, const int xmax)
{
    int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
    float recipmcs = 1.0f/mcs;
    if (ggid < xmax)
    {
        Rzz[ggid] = (1.0/2.0)*Rzz[ggid]*recipmcs;
    }
}

//-----------------------------------------------------------------------------

__global__ void final_temp(float *Rzz, const int mcs, const int xmax, float *Rzztemp)
{
    int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
    float recipmcs = 1.0f/mcs;
    if (ggid < xmax)
    {
        Rzztemp[ggid] = (1.0/2.0)*Rzz[ggid]*recipmcs;
    }
}

//-----------------------------------------------------------------------------

__global__ void calcsproj(float *w, float *wi, float *s, const int a, float *wfin)
{
    extern __shared__ float locmem[];
    float* sloc = locmem;
    float* wloc = sloc + 3*blockDim.x;
    float wtemp[3];
    int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int glid = threadIdx.x;
    int nl = blockDim.x;
    int groupid = blockIdx.x;
    int sind = 3*nl*groupid + glid;
    for(int ii = 0; ii < 3; ++ii, sind += nl)
    {
        sloc[glid + ii*nl] = s[sind];
    }
    __syncthreads();
    wloc[3*glid] = w[3*a*ggid];
    wloc[3*glid + 1] = w[3*a*ggid + 1];
    wloc[3*glid + 2] = w[3*a*ggid + 2];
    wtemp[0] = wloc[3*glid] + wi[0];
    wtemp[1] = wloc[1 + 3*glid] + wi[1];
    wtemp[2] = wloc[2 + 3*glid] + wi[2];
    float len = sqrt(wtemp[0]*wtemp[0] + wtemp[1]*wtemp[1] + wtemp[2]*wtemp[2]);
    wtemp[0] = wtemp[0]/len;
    wtemp[1] = wtemp[1]/len;
    wtemp[2] = wtemp[2]/len;
    float dot = wtemp[0]*sloc[3*glid] + wtemp[1]*sloc[3*glid + 1] + wtemp[2]*sloc[3*glid + 2];
    wloc[3*glid] = dot*wtemp[0];
    wloc[3*glid + 1] = dot*wtemp[1];
    wloc[3*glid + 2] = dot*wtemp[2];
    __syncthreads();
    int wind = 3*nl*groupid + glid;
    for(int ii = 0; ii < 3; ++ii, wind += nl)
    {
        wfin[wind] = wloc[glid + ii*nl];
    }
}

//-----------------------------------------------------------------------------


int main(void)
{

    int nDevices;

    clock_t t;
    

    t = clock();

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
      printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

    int ni = 1160;
    
    int nindium = 553;
    
    
    
    int local_size1 = 32;
    int local_size2 = 32;
    
    int global_blocks1 = (ni + local_size1 - 1)/local_size1;
    int global_blocks2 = 1;
    
    int global_size1 = global_blocks1*local_size1;
    int global_size2 = global_blocks2*local_size2;
    
    
    
    // Set up timestep
    
    float dt = 2.6279419040294205*20;
    
    // Set up maxtime
    
    float tmax = 10000.0;
    
    // xmax - total number of timesteps
    
    int xmax = tmax/dt[0];
    
    // xmax must be a multiple of iterations
    
    int iterations = 1;
    
    int size = xmax/iterations;
    
    // Global size for final kernels
    
    int global_blocks_tensors = (size + local_size1 - 1)/local_size1;
    
    int global_sizetensors = global_blocks_tensors * local_size1;
    
    dim3 gridSizetensors = dim3 (global_blocks_tensors, global_blocks2);

    // Global size for odd reductions

    int global_blocks_odd = (global_blocks1 + local_size1 - 1)/local_size1;

    int global_size_odd = global_blocks_odd*local_size1;

    dim3 gridSizeodd = dim3 (global_blocks_odd, global_blocks2);

    // Global size for final step

    int global_blocks_final = (xmax + local_size1 - 1)/local_size1;
    
    // Set up monte carlo iterations
    
    int mcs = 5;
    
    // Set up 2D workgroups
    
    dim3 blockSize(local_size1, local_size2);
    
    dim3 gridSize = dim3 (global_blocks1, global_blocks2);
    
    // Set up electron spin and initial electron spin arrays
    float *s, *sinit;
    cudaMallocManaged(&s, 3*global_size2*sizeof(float));
    cudaMallocManaged(&sinit, global_size2*sizeof(float));

    float *sproj;

    cudaMallocManaged(&sproj, 3*global_size2*sizeof(float));
    
    // Set up external field
    float *wi;
    cudaMallocManaged(&wi, 3*sizeof(float));
    
    wi[0] = 0.0;
    wi[1] = 0.0;
    wi[2] = 0.0;
    
    
    
    
    // Set up nuclear spin vector arrays
    float *i;
    cudaMallocManaged(&i, 3*global_size2*ni*sizeof(float));
    
   
    
    // Set up state for random number generation
    curandState *state;
    

    cudaMallocManaged((void**)&state, global_size1*global_size2*sizeof(curandState));
    
    
    // Set up the hyperfine constants
    float *hyperfine;
    
    cudaMallocManaged(&hyperfine, ni*sizeof(float));
    
    std::ifstream hyp;
    
    hyp.open("hyp.txt");
    
    int p = 0;
    for(std::string line; std::getline(hyp, line); )
    {
        hyperfine[p]=std::atof(line.c_str());
        p += 1;
    }
    
    
    
    
    /*
    hyperfine[0] =-0.999985;
    hyperfine[1] =-0.7369246;
    hyperfine[2] =0.511210;
    hyperfine[3] =-0.0826998;
    
    hyperfine[4] =0.0655341;
    hyperfine[5] =-0.562082;
    hyperfine[6] =-0.905911;
    hyperfine[7] =0.357729;
    hyperfine[8] =0.358593;
    hyperfine[9] =0.869386;
    hyperfine[10] =-0.232996;
    hyperfine[11] =0.0388327;
    hyperfine[12] =0.661931;
    hyperfine[13] =-0.930856;
    hyperfine[14] =-0.893077;
    hyperfine[15] =0.0594001;
    */
    
    // Set up omega vector
    float *w;
    
    cudaMallocManaged(&w, 3*global_size1*global_size2*sizeof(float));

    // Set up output of omega vector

    float *wout;

    cudaMallocManaged(&wout, 3*global_size_odd*global_size2*sizeof(float));
    
    // Set up tensor vectors
    float *Rzz;

    cudaMallocManaged(&Rzz, xmax*sizeof(float));

    float *Rzztemp;
    
    cudaMallocManaged(&Rzztemp, xmax*sizeof(float));
    
    // Set up electron spin storage vector
    float *sstore;
    
    cudaMallocManaged(&sstore, size*global_size2*sizeof(float));
    
    // Set up output
    float *output;
    
    cudaMallocManaged(&output, global_sizetensors*global_size2*sizeof(float));

    float *out;
    
    cudaMallocManaged(&out, global_sizetensors*global_size2*sizeof(float));

    
    // Work out logs
    int n1 = log2f(local_size1);
    int n2 = log2f(local_size2);
    
    // Set up seed for random number generation
    unsigned long seed = 1; 

    int pmax = 0;

    float time = 0;

    int a = 0;
        
    //----------------------------------------------------------------------------- 
    // Kernel Calls
    
    // Call random number generation setup kernel
    setup_rand<<<global_blocks1*global_blocks2, local_size1*local_size2>>>(state,seed,mcs);
    
    for (int u = 0; u < mcs; ++u)
    {
        
        
        // Build electron spin vectors array
        vecbuilds<<<global_blocks2, local_size2, 3*local_size2*sizeof(float)>>>(s, sinit, state);
        
        
        
        
        // Build nuclear spin vector array
        vecbuildi<<<gridSize, blockSize, 3*local_size1*local_size2*sizeof(float)>>>(i, state, ni, nindium);

        a = global_size1;
        p = 0;
        
        while (a>1)
        {
            if (p%2 == 0)
            {
                reduce<<<gridSize, blockSize, (local_size1 + 3*local_size1*local_size2)*sizeof(float)>>>(i, w, n1, a, hyperfine, ni, wout, p, global_size_odd);
            } else{
                reduce<<<gridSizeodd, blockSize, (local_size1 + 3*local_size1*local_size2)*sizeof(float)>>>(i, wout, n1, a, hyperfine, ni, w, p, global_size1);
            }
            
            if (a%local_size1 == 0)
            {
                a = a/local_size1;
            } else {
                a = a/local_size1 + 1;
            }
            p = p + 1;
        }

        if (pmax%2 == 0)
        {
            calcsproj<<<global_blocks2,local_size2,2*3*local_size2*sizeof(float)>>>(w, wi, s, global_size1, sproj);
        } else {
            calcsproj<<<global_blocks2,local_size2,2*3*local_size2*sizeof(float)>>>(wout, wi, s, global_size_odd, sproj);
        }

        precessnucspins<<<gridSize, blockSize, 3*local_size1*local_size2*sizeof(float)>>>(i, hyperfine, sproj, ni, dt/2.0);
        
        for (int j = 0; j < iterations; ++j)
        {
            
            for (int x = 0; x < size; ++x)
            {    

                p = 0;
                a = global_size1;
                
                while (a>1)
                {
                    if (p%2 == 0)
                    {
                        reduce<<<gridSize, blockSize, (local_size1 + 3*local_size1*local_size2)*sizeof(float)>>>(i, w, n1, a, hyperfine, ni, wout, p, global_size_odd);
                    } else{
                        reduce<<<gridSizeodd, blockSize, (local_size1 + 3*local_size1*local_size2)*sizeof(float)>>>(i, wout, n1, a, hyperfine, ni, w, p, global_size1);
                    }
                    
                    if (a%local_size1 == 0)
                    {
                        a = a/local_size1;
                    } else {
                        a = a/local_size1 + 1;
                    }
                    p = p + 1;
                }
                pmax = p;

                


                
                if (pmax%2 == 0)
                {
                    precesselecspins<<<global_blocks2,local_size2,2*3*local_size2*sizeof(float)>>>(w, wi, s, size, x, sstore, global_size1, dt, sproj);
                } else {
                    precesselecspins<<<global_blocks2,local_size2,2*3*local_size2*sizeof(float)>>>(wout, wi, s, size, x, sstore, global_size_odd, dt, sproj);
                }
                /*
                for (int ii =0; ii<global_size2; ++ii)
                {
                    std::cout << x << " " << s[3*ii] << s[3*ii + 1] << s[3*ii + 2] << std::endl;
                    std::cout << x << " " << sproj[3*ii] << sproj[3*ii + 1] << sproj[3*ii + 2] << std::endl;
                }
                std::cout << x << " " << wout[0] << wout[1] << wout[2] << std::endl;
                */
                precessnucspins<<<gridSize, blockSize, 3*local_size1*local_size2*sizeof(float)>>>(i, hyperfine, sproj, ni, dt);
            }
            
            // Prepare sstore for Rxx, Rxy, Rzz calculation
            prep2<<<gridSizetensors, blockSize, local_size1*local_size2*sizeof(float)>>>(sstore, output, size, sinit);
            
            // Reset b between each monte carlo step
            int b = global_size2;
            int g = 0;
            // Reduction in the y direction (over different monte carlo steps running in parallel)
            // note that global size in the x direction is now related to xmax (no longer ni)
            
            while (b>1)
            {

                if (g%2 == 0)
                {
                    reduce2<<<gridSizetensors, blockSize, local_size1*local_size2*sizeof(float)>>>(n2, b, output, out);
                } else {
                    reduce2<<<gridSizetensors, blockSize, local_size1*local_size2*sizeof(float)>>>(n2, b, out, output);
                }
                
                if (b%local_size2 == 0)
                {
                    b = b/local_size2;
                }
                else
                {
                    b = b/local_size2 + 1;
                }
                g = g + 1;
            }
            
            // Sum Rxx, Rxy, Rzz over different monte carlo step iterations - note that this is
            // now a 1D workgroup size
            if (g%2 ==0)
            {
                tensors<<<global_blocks_tensors, local_size1>>>(output, Rzz, size, j);
            } else {
                tensors<<<global_blocks_tensors, local_size1>>>(out, Rzz, size, j);
            }
            
        }
        /*
        if (u%5 == 0 && u != 0)
        {
            final_temp<<<global_blocks_final, local_size1>>>(Rxx, Rxy, Rzz, u*global_size2, xmax, Rxxtemp, Rxytemp, Rzztemp);
            cudaDeviceSynchronize();
            if (u%2 == 0)
            {
                std::ofstream Rzztemp2txt;
        
                Rzztemp2txt.open("Rzz_w=0_93_temp2_1.txt");
                
                time = 0;
                
                for (int j = 0; j<xmax; ++j)
                {
                    Rzztemp2txt << time << " " << Rzztemp[j] << "\n";
                    time += dt[0];
                }
                Rzztemp2txt.close();
            } else {

                std::ofstream Rzztemp1txt;
        
                Rzztemp1txt.open("Rzz_w=0_93_temp1_1.txt");
                
                time = 0;
                
                for (int j = 0; j<xmax; ++j)
                {
                    Rzztemp1txt << time << " " << Rzztemp[j] << "\n";
                    time += dt[0];
                }
                Rzztemp1txt.close();
            }
        }
        */
        
    }
    
    int h = mcs*global_size2;
    
    
    
    final<<<global_blocks_final, local_size1>>>(Rzz, h, xmax);
    
    cudaDeviceSynchronize();

    t = clock() - t;
    //auto end = std::chrono::high_resolution_clock::now();

    //std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "us" << std::endl;
    std::cout << "Time: " << t << std::endl;
    
    
    
    std::ofstream Rzztxt;
    
    Rzztxt.open("Rzz_w=0_tstep=taumul20.txt");
    
    time = 0;
    
    for (int j = 0; j<xmax; ++j)
    {
        Rzztxt << time << " " << Rzz[j] << "\n";
        std::cout << time << " " << Rzz[j] << std::endl;
        time += dt;
    }
    Rzztxt.close();
    
    
    cudaFree(s);
    cudaFree(sinit);
    cudaFree(state);
    cudaFree(i);
    cudaFree(hyperfine);
    cudaFree(w);
    cudaFree(Rzz);
    cudaFree(sstore);
    cudaFree(output);
    cudaFree(wi);
    cudaFree(out);
    cudaFree(wout);
    cudaFree(Rzztemp);
    return 0;
    
    
    
}


