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

//----------------------------------------------------------------------------- 

__global__ void precessnucspins (float *i, float *s, const int ni, float* hyp, float* wout, const int n, const int size, const float dt)
{
    extern __shared__ float iloc[];

    int glid = threadIdx.x;
    int glid1 = threadIdx.y;
    int groupid = blockIdx.x;
    int nl = blockDim.x;
    int nl1 = blockDim.y;
    int ggid = (groupid * nl) + glid;
    int ggid1 = (blockIdx.y * nl1) + glid1;
    float w[3];
    float hyperfine = 0;
    int locsize = 3*nl*glid1;
    int locid = 3*glid + locsize;
    int globsize = 3*ni*ggid1;
    int sind = 3*nl*groupid + glid;
    for(int ii = 0; ii < 3; ++ii, sind += nl)
    {
        if (sind < 3*ni)
        {
            iloc[glid + ii*nl + locsize] = i[sind + globsize];
        } else {
            iloc[glid + ii*nl + locsize] = 0;
        }
        
    }
    __syncthreads();
    if (ggid < ni)
    {
// Idea! ADD CHECK IF GLID = 0 TO PREVENT MEMORY BANK CONFLICTS AND SAVE TO LOC MEM
// OR   write to code to prevent strided mem access
        hyperfine = hyp[ggid];

        w[0] = hyperfine*s[3*ggid1];
        w[1] = hyperfine*s[1 + 3*ggid1];
        w[2] = hyperfine*s[2 + 3*ggid1];

        rot (w, iloc+(locid), dt);
        
        
        
    }
    __syncthreads();
    int wind = 3*nl*groupid + glid;
    for(int ii = 0; ii < 3 && wind < 3*ni; ++ii, wind += nl)
    {
        i[wind + globsize] = iloc[glid + locsize + ii*nl];
    }
    __syncthreads();

    iloc[locid] = hyperfine*iloc[locid];
    iloc[locid + 1] = hyperfine*iloc[locid + 1];
    iloc[locid + 2] = hyperfine*iloc[locid + 2];

    #pragma unroll
    for (int k=1; k < n; k++)
    {
        __syncthreads();
        int b = nl >> k;
        if (glid < b) 
        {
            iloc[locid] += iloc[3*(glid + b)+ locsize];
            iloc[locid + 1] += iloc[3*(glid + b) + 1+ locsize];
            iloc[locid + 2] += iloc[3*(glid + b) + 2+ locsize];
        }
    }
    __syncthreads();
    
    if (glid == 0) 
    {
        wout[(ggid >> n)*3 + 3*size*ggid1] = iloc[locsize] + iloc[3 + locsize];
        wout[(ggid >> n)*3 + 1 + 3*size*ggid1] = iloc[1 + locsize] + iloc[4 + locsize];
        wout[(ggid >> n)*3 + 2 + 3*size*ggid1] = iloc[2 + locsize] + iloc[5 + locsize];
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

//----------------------------------------------------------------------------- 

__global__ void vecbuilds(float *s, float *sinit, curandState *state, const float len)
{
    extern __shared__ float sloc[];
    int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int nl = blockDim.x;
    int glid = threadIdx.x;
    int groupid = blockIdx.x;
    float v = curand_uniform(&state[ggid]);
    float g = curand_uniform(&state[ggid]);
    float phi = 2.0*M_PI*v;
    float th = acos(cbrtf(2.0*g - 1.0));
    sloc[3*glid] = len*sin(th)*cos(phi);
    sloc[3*glid + 1] = len*sin(th)*sin(phi);
    sloc[3*glid + 2] = len*cos(th);
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

__global__ void reduce(float *w, const int n, const int a, float *wout, const int size)
{
    extern __shared__ float wtemp[];
    int nl = blockDim.x;
    int groupid = blockIdx.x;
    int glid = threadIdx.x;
    int glid1 = threadIdx.y;
    int ggid = (groupid * nl) + glid;
    int ggid1 = (blockIdx.y * blockDim.y) + glid1;
    int ng = nl*gridDim.x;
    int locsize = 3*nl*glid1;
    int globsize = 3*ng*ggid1;
    int id = 3*glid + locsize;

    int wind = 3*nl*groupid + glid;
    
    wtemp[glid + locsize] = w[wind + globsize];
    wtemp[glid + nl + locsize] = w[wind + nl + globsize];
    wtemp[glid + 2*nl + locsize] = w[wind + 2*nl + globsize];
    
    #pragma unroll
    for (int k=1; k < n; k++)
    {
        __syncthreads();
        int b = nl >> k;
        if (glid < b) 
        {
            wtemp[id] += wtemp[3*(glid + b)+ locsize];
            wtemp[id + 1] += wtemp[3*(glid + b) + 1+ locsize];
            wtemp[id + 2] += wtemp[3*(glid + b) + 2+ locsize];
        }
    }
    __syncthreads();
    
    if (glid == 0) 
    {
        wout[(ggid >> n)*3 + 3*size*ggid1] = wtemp[locsize] + wtemp[3 + locsize];
        wout[(ggid >> n)*3 + 1 + 3*size*ggid1] = wtemp[1 + locsize] + wtemp[4 + locsize];
        wout[(ggid >> n)*3 + 2 + 3*size*ggid1] = wtemp[2 + locsize] + wtemp[5 + locsize];
    }    
    int c = (a + nl - 1)/nl;
    
    __syncthreads();
    
    
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
float *sstore, const int a, const float dt)
{
    extern __shared__ float sloc[];
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
    wtemp[0] = w[3*a*ggid];
    wtemp[1] = w[3*a*ggid + 1];
    wtemp[2] = w[3*a*ggid + 2];
    sstore[x + size*ggid] = sloc[3*glid + 2];
    wtemp[0] = wtemp[0] + wi[0];
    wtemp[1] = wtemp[1] + wi[1];
    wtemp[2] = wtemp[2] + wi[2];
    rot (wtemp, sloc+(3*glid), dt);
    __syncthreads();
    int wind = 3*nl*groupid + glid;
    for(int ii = 0; ii < 3; ++ii, wind += nl)
    {
        s[wind] = sloc[glid + ii*nl];
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
    if (ggid < size)
    {
        loc[glid + nl*glid1] = sstore[ggid + size*ggid1];
        loc[glid + nl*glid1] = loc[glid + nl*glid1]/sinit[ggid1];
    }
    output[ggid + ng*ggid1] = (1.0/2.0)*loc[glid + nl*glid1];
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

__global__ void tensors(float *output, float *Rzz, const int size, const int j, const float recipmcs)
{
     int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
     
     if (ggid < size) 
     {
        Rzz[ggid + j*size] = Rzz[ggid + j*size] + output[ggid]*recipmcs;
        
     }
     
}

//-----------------------------------------------------------------------------

__global__ void final(float *Rzz, const float recipmcs, const int xmax)
{
    int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ggid < xmax)
    {
        Rzz[ggid] = Rzz[ggid]*recipmcs;
    }
}

//-----------------------------------------------------------------------------

__global__ void final_temp(float *Rzz, const float recipmcs, const int xmax, float *Rzztemp)
{
    int ggid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ggid < xmax)
    {
        Rzztemp[ggid] = Rzz[ggid]*recipmcs;
    }
}

//-----------------------------------------------------------------------------


int main(void)
{

    // Code only works if ni <= local_size1**2 - haven't had time to figure out why

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

    int ni = 100;
    
    int nindium = 50;

    float len = sqrt(5.0/12.0);
    
    int local_size1 = 32;
    int local_size2 = 32;
    
    int global_blocks1 = (ni + local_size1 - 1)/local_size1;
    int global_blocks2 = 32;
    
    int global_size1 = global_blocks1*local_size1;
    int global_size2 = global_blocks2*local_size2;
    
    
    
    // Set up timestep
    
    float dt = 65.162413696574831;
    
    
    
    // Set up maxtime
    
    //float tmax = 30000000.0;
    
    // xmax - total number of timesteps
    
    int xmax = 1000;
    
    // xmax must be a multiple of iterations
    
    int iterations = xmax/1000;
    
    int size = xmax/iterations;
    
    // Global size for final kernels
    
    int global_blocks_tensors = (size + local_size1 - 1)/local_size1;
    
    int global_sizetensors = global_blocks_tensors * local_size1;
    
    dim3 gridSizetensors = dim3 (global_blocks_tensors, global_blocks2);

    // Global size for odd reductions

    int global_blocks_odd = (global_blocks1 + local_size1 - 1)/local_size1;

    int global_size_odd = global_blocks_odd*local_size1;

    dim3 gridSizeodd = dim3 (global_blocks_odd, global_blocks2);

    // Global size for 3rd reduction

    
    int global_blocks_red = (global_blocks_odd + local_size1 - 1)/local_size1;;

    int global_size_red = global_blocks_red*local_size1;

    dim3 gridSizered = dim3 (global_blocks_red, global_blocks2);

    // Global size for final step

    int global_blocks_final = (xmax + local_size1 - 1)/local_size1;
    
    // Set up monte carlo iterations
    
    int mcs = 1;
    
    // Set up 2D workgroups
    
    dim3 blockSize(local_size1, local_size2);
    
    dim3 gridSize = dim3 (global_blocks1, global_blocks2);
    
    // Set up electron spin and initial electron spin arrays
    float *s, *sinit;
    cudaMallocManaged(&s, 3*global_size2*sizeof(float));
    cudaMallocManaged(&sinit, global_size2*sizeof(float));
    
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
        
    // Kernel Calls
    
    // Call random number generation setup kernel
    setup_rand<<<global_blocks1*global_blocks2, local_size1*local_size2>>>(state,seed,mcs);


    
    
    for (int u = 0; u < mcs; ++u)
    {
        
        
        // Build electron spin vectors array
        vecbuilds<<<global_blocks2, local_size2, 3*local_size2*sizeof(float)>>>(s, sinit, state, len);
        
        
        
        
        // Build nuclear spin vector array
        vecbuildi<<<gridSize, blockSize, 3*local_size1*local_size2*sizeof(float)>>>(i, state, ni, nindium);

        // Precess the nuclear spins by dt/2 initially

        precessnucspins<<<gridSize, blockSize, 3*local_size1*local_size2*sizeof(float)>>>(i, s, ni, hyperfine, wout, n1, global_size_odd, dt/2.0);
        
        for (int j = 0; j < iterations; ++j)
        {
            
            for (int x = 0; x < size; ++x)
            {
            
                int p = 0;
                int a = global_blocks1;
                
                
                while (a>1)
                {
                    if (p%2 != 0)
                    {
                        reduce<<<gridSizered, blockSize, 3*local_size1*local_size2*sizeof(float)>>>(w, n1, a, wout, global_size_odd);
                    } else{
                        reduce<<<gridSizeodd, blockSize, 3*local_size1*local_size2*sizeof(float)>>>(wout, n1, a, w, global_size_red);
                    }
                    
                    a = (a + local_size1 - 1)/local_size1;
                    p = p + 1;
                }
                pmax = p;

                


                
                if (pmax%2 != 0)
                {
                    precesselecspins<<<global_blocks2,local_size2,3*local_size2*sizeof(float)>>>(w, wi, s, size, x, sstore, global_size_red, dt);
                } else {
                    precesselecspins<<<global_blocks2,local_size2,3*local_size2*sizeof(float)>>>(wout, wi, s, size, x, sstore, global_size_odd, dt);
                }
                
                
                precessnucspins<<<gridSize, blockSize, 3*local_size1*local_size2*sizeof(float)>>>(i, s, ni, hyperfine, wout, n1, global_size_odd, dt);
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
                
                b = (b + local_size2 - 1)/local_size2;
                g = g + 1;
            }
            
            // Sum Rxx, Rxy, Rzz over different monte carlo step iterations - note that this is
            // now a 1D workgroup size
            if (g%2 ==0)
            {
                tensors<<<global_blocks_tensors, local_size1>>>(output, Rzz, size, j, 1.0/global_size2);
            } else {
                tensors<<<global_blocks_tensors, local_size1>>>(out, Rzz, size, j, 1.0/global_size2);
            }
            
        }
        /*
        int o = 0;
        if (u != (mcs-1))
        {
            final_temp<<<global_blocks_final, local_size1>>>(Rzz, (u+1)*global_size2, xmax, Rzztemp);
            cudaDeviceSynchronize();
            
            std::ofstream Rzztemp2txt;
    
            Rzztemp2txt.open("Rzz_w=0_10071_temp1.txt");
            
            time = 0;
            
            for (int j = 0; j<xmax; ++j)
            {
                Rzztemp2txt << time << " " << Rzztemp[j] << "\n";
                time += dt;
            }
            Rzztemp2txt.close();
        }
        */
    }
    
    
    
    final<<<global_blocks_final, local_size1>>>(Rzz, 1.0/mcs, xmax);
    
    cudaDeviceSynchronize();

    t = clock() - t;
    //auto end = std::chrono::high_resolution_clock::now();

    //std::cout << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "us" << std::endl;
    std::cout << "Time: " << t << std::endl;
    
    
    
    std::ofstream Rzztxt;
    
    Rzztxt.open("root512_100_spins_1.txt");

    for (int j = 0; j<xmax; ++j)
    {
        Rzztxt << j << " " << Rzz[j] << "\n";
        //std::cout << time << " " << Rzz[j] << std::endl;
        time += dt;
    }
    Rzztxt.close();
    
    /*
    time = 0;

    float sum = 0;
    
    for (int j = 0; j<xmax; ++j)
    {
        
        if (j <= 1000)
        {
            Rzztxt << j << " " << Rzz[j] << "\n";
        }

        else if (j <= 10000)
        {
            sum = sum + Rzz[j];
            if (j%10 == 0)
            {
                Rzztxt << j - 5 << " " << sum/10.0 << "\n";
                sum = 0;
            }
        }

        else if (j <= 100000)
        {
            sum = sum + Rzz[j];
            if (j%100 == 0)
            {
                Rzztxt << j - 50 << " " << sum/100.0 << "\n";
                sum = 0;
            }
        }
        else if (j <= 1000000)
        {
            if (j%1000 == 0)
            {
                Rzztxt << j << " " << Rzz[j] << "\n";
            }
        }
        else if (j <= 10000000)
        {
            if (j%10000 == 0)
            {
                Rzztxt << j << " " << Rzz[j] << "\n";
            }
        }
        else if (j <= 100000000)
        {
            if (j%100000 == 0)
            {
                Rzztxt << j << " " << Rzz[j] << "\n";
            }
        }
    }

    */
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



