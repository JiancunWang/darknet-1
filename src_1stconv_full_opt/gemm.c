#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sds_lib.h>
#define MIN(X,Y) (((X) < (Y)) ? (X) : (Y))

#define array_size(M,N) M*N
#define CONST 600
#define NUM1 array_size(27,CONST)
#define NUM2 array_size(16,CONST)


/* return value of free-running 64-bit Zynq(TM) global counter */
unsigned long long sds_clock_counter(void);
unsigned long long total_run_time = 0;
unsigned int num_calls = 0;
unsigned long long count_val = 0;
#define sds_clock_start(){ \
    count_val = sds_clock_counter(); \
    num_calls++; \
}
#define sds_clock_stop() { \
    long long tmp = sds_clock_counter(); \
    total_run_time += (tmp - count_val); \
}
#define avg_cpu_cycles()(total_run_time / num_calls)


#pragma SDS data data_mover(B:AXIDMA_SG, C:AXIDMA_SG)
//#pragma SDS data dim (B[NUM1],   C[NUM2],   D[NUM2])
//#pragma SDS data zero_copy(B[0:NUM1], C[0:NUM2])
//#pragma SDS data access_pattern(B:SEQUENTIAL, C:SEQUENTIAL)
#pragma SDS data mem_attribute(B:CACHEABLE|NON_PHYSICAL_CONTIGUOUS,C:CACHEABLE|NON_PHYSICAL_CONTIGUOUS)
int gemm_nn_hw(float ALPHA, 
        float B[array_size(27,CONST)], 
        float C[array_size(16,CONST)]);

//float IFMAP[array_size(27,60)],OFMAP[array_size(16,60)],OFMAP_TMP[array_size(16,60)];

void init1_weights(float Weights[16][27]);

void gemm_nn_sw(int M, int N, int K, float ALPHA, 
        int lda, 
        float B[array_size(27,200704)], int ldb,
        float C[array_size(16,200704)], int ldc,
        float D[array_size(16,200704)], int N_K, int NN);

void init_weights(float Weights[432]);

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    printf ("GEMM NN SW\n");
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

int gemm_nn_hw(float ALPHA, 
        float B[array_size(27,CONST)], 
        float C[array_size(16,CONST)])
{
    int M = 16, N = CONST, K = 27;
    int i,j,k;
    float B_buf[27][CONST];
    float C_buf[16][CONST];
    float Weights[16][27];
    init1_weights(Weights);
    float temp, result;

#pragma HLS array_partition variable=B_buf   block factor=14 dim=1
#pragma HLS array_partition variable=Weights block factor=14 dim=2

    // store to buffer
    L1: for(i = 0; i < K; ++i){
      L1_1: for(j = 0; j < N; ++j){
#pragma HLS PIPELINE II=1
        B_buf[i][j] = B[i*N+j];
      }
    }

    L2: for(i = 0; i < M; ++i){
        L2_1: for(j = 0; j < N; ++j){
            result = 0;
#pragma HLS PIPELINE II=1
            L2_1_1: for(k = 0; k < K; ++k){
                temp = Weights[i][k] * B_buf[k][j];
                result += temp;
            }
            C_buf[i][j] = result;
        }
    }

    // store to buffer
    L3: for(i = 0; i < M; ++i){
      L3_1: for(j = 0; j < N; ++j){
#pragma HLS PIPELINE II=1
        C[i*N+j] = C_buf[i][j];
      }
    }

    return 1;

}

void gemm_nn_sw(int M, int N, int K, float ALPHA, 
        int lda, 
        float B[array_size(27,200704)], int ldb,
        float C[array_size(16,200704)], int ldc,
        float D[array_size(16,200704)], int N_K, int NN)
{
    int i,j,k;
    float Weights[432];
    init_weights(Weights);
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*Weights[i*lda+k];
            for(j = N_K; j < MIN(N_K+N, NN); ++j){
                D[i*ldc+j] = C[i*ldc+j] + A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

void gemm_cpu_hw(int TA, int TB, int M, int N, int K, float ALPHA, 
        float A[FIXED_SIZE], int lda, 
        float B[FIXED_SIZE], int ldb,
        float BETA,
        float C[FIXED_SIZE], int ldc)
{
    int i, j;
    int P = CONST;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
//    float *D = C;

    printf("GEMM NN HW function call \n");

    float *IFMAP, *IFMAP2, *IFMAP3, *IFMAP4;
    IFMAP=(float *) sds_alloc(27*CONST*sizeof(float));
    IFMAP2=(float *) sds_alloc(27*CONST*sizeof(float));
    IFMAP3=(float *) sds_alloc(27*CONST*sizeof(float));
    IFMAP4=(float *) sds_alloc(27*CONST*sizeof(float));

    float *OFMAP, *OFMAP2, *OFMAP3, *OFMAP4;
    OFMAP=(float *) sds_alloc(16*CONST*sizeof(float));
    OFMAP2=(float *) sds_alloc(16*CONST*sizeof(float));
    OFMAP3=(float *) sds_alloc(16*CONST*sizeof(float));
    OFMAP4=(float *) sds_alloc(16*CONST*sizeof(float));

    if (!IFMAP || !OFMAP || !IFMAP2 || !OFMAP2 || !IFMAP3 || !OFMAP3 || !IFMAP4 || !OFMAP4) {
       if (IFMAP)     sds_free(IFMAP);
       if (OFMAP)     sds_free(OFMAP);
       if (IFMAP2)    sds_free(IFMAP2);
       if (OFMAP2)    sds_free(OFMAP2);
       if (IFMAP3)    sds_free(IFMAP3);
       if (OFMAP3)    sds_free(OFMAP3);
       if (IFMAP4)    sds_free(IFMAP4);
       if (OFMAP4)    sds_free(OFMAP4);
       printf("Mem alloc failed");
    } 

/*
    float *IFMAP, *IFMAP2, *IFMAP3, *IFMAP4;
    IFMAP =(float *) malloc(27*CONST*sizeof(float));
    IFMAP2=(float *) malloc(27*CONST*sizeof(float));
    IFMAP3=(float *) malloc(27*CONST*sizeof(float));
    IFMAP4=(float *) malloc(27*CONST*sizeof(float));

    float *OFMAP, *OFMAP2, *OFMAP3, *OFMAP4;
    OFMAP =(float *) malloc(16*CONST*sizeof(float));
    OFMAP2=(float *) malloc(16*CONST*sizeof(float));
    OFMAP3=(float *) malloc(16*CONST*sizeof(float));
    OFMAP4=(float *) malloc(16*CONST*sizeof(float));

    if (!IFMAP || !OFMAP || !IFMAP2 || !OFMAP2 || !IFMAP3 || !OFMAP3 || !IFMAP4 || !OFMAP4 ) {
       if (IFMAP)     free(IFMAP);
       if (OFMAP)     free(OFMAP);
       if (IFMAP2)    free(IFMAP2);
       if (OFMAP2)    free(OFMAP2);
       if (IFMAP3)    free(IFMAP3);
       if (OFMAP3)    free(OFMAP3);
       if (IFMAP4)    free(IFMAP4);
       if (OFMAP4)    free(OFMAP4);
       printf("Mem alloc failed");
    }
*/

//    clock_t start = clock(), diff;


    printf("Allocation done \n");
    int x;
    for(x = 0; x < MIN(P+x,N); x = x+P) {

    // 1st iteration ...
      // Init arrays
      for(i = 0; i<K; ++i ) {
        for(j = x; j< MIN(P+x,N); ++j ) {
          IFMAP[i*P + (j-x)] = B[i*ldb+j]; 
        }
      }
      for(i = 0; i<M; ++i ) {
        for(j = x; j<MIN(P+x,N); ++j ) {
          OFMAP[i*P + (j-x)] = C[i*ldc+j]; 
        }
      }

//      #pragma SDS wait(1)
//      #pragma SDS async(1)

      sds_clock_start();      
      int success = gemm_nn_hw( ALPHA,IFMAP, OFMAP);
      sds_clock_stop();

      for(i = 0; i<M; ++i ) {
        for(j = x; j<MIN(P+x,N); ++j ) {
          C[i*ldc+j] = OFMAP[i*P+(j-x)];
        }
      }
    // End of 1st iteration ...
/*
    // 2nd iteration ...
      // Init arrays
      for(i = 0; i<K; ++i ) {
        for(j = x+P; j< MIN(2*P+x,N); ++j ) {
          IFMAP2[i*P + (j-x-P)] = B[i*ldb+j]; 
        }
      }
      for(i = 0; i<M; ++i ) {
        for(j = x+P; j<MIN(2*P+x,N); ++j ) {
          OFMAP2[i*P + (j-x-P)] = C[i*ldc+j]; 
        }
      }

      #pragma SDS wait(2)
      #pragma SDS async(2)
      gemm_nn_hw( ALPHA,IFMAP2, OFMAP2);

      for(i = 0; i<M; ++i ) {
        for(j = x+P; j<MIN(2*P+x,N); ++j ) {
          C[i*ldc+j] = OFMAP2[i*P+(j-x-P)];
        }
      }
    // End of 2nd iteration ...

    // 3rd iteration ...
      // Init arrays
      for(i = 0; i<K; ++i ) {
        for(j = x+2*P; j< MIN(3*P+x,N); ++j ) {
          IFMAP3[i*P + (j-x-2*P)] = B[i*ldb+j]; 
        }
      }
      for(i = 0; i<M; ++i ) {
        for(j = x+2*P; j<MIN(3*P+x,N); ++j ) {
          OFMAP3[i*P + (j-x-2*P)] = C[i*ldc+j]; 
        }
      }

      #pragma SDS wait(3)
      #pragma SDS async(3)
      gemm_nn_hw( ALPHA,IFMAP3, OFMAP3);

      for(i = 0; i<M; ++i ) {
        for(j = x+2*P; j<MIN(3*P+x,N); ++j ) {
          C[i*ldc+j] = OFMAP3[i*P+(j-x-2*P)];
        }
      }
    // End of 3rd iteration ...

    // 4th iteration ...
      // Init arrays
      for(i = 0; i<K; ++i ) {
        for(j = x+3*P; j< MIN(4*P+x,N); ++j ) {
          IFMAP4[i*P + (j-x-3*P)] = B[i*ldb+j]; 
        }
      }
      for(i = 0; i<M; ++i ) {
        for(j = x+3*P; j<MIN(4*P+x,N); ++j ) {
          OFMAP4[i*P + (j-x-3*P)] = C[i*ldc+j]; 
        }
      }

      #pragma SDS wait(4)
      #pragma SDS async(4)
      gemm_nn_hw( ALPHA,IFMAP4, OFMAP4);

      for(i = 0; i<M; ++i ) {
        for(j = x+3*P; j<MIN(4*P+x,N); ++j ) {
          C[i*ldc+j] = OFMAP4[i*P+(j-x-3*P)];
        }
      }
    // End of 4th iteration ...
*/    
    }

//    diff = clock() - start;
//    int msec = diff * 1000 / CLOCKS_PER_SEC;
//    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
    printf("Average cpu cycles gemm_nn_hw(): %ld\n", avg_cpu_cycles());
    printf("Total run time cycles gemm_nn_hw(): %lld\n", total_run_time);
    printf("Total function calls gemm_nn_hw(): %u\n", num_calls);


    sds_free(IFMAP);
    sds_free(OFMAP);
    sds_free(IFMAP2);
    sds_free(OFMAP2);
    sds_free(IFMAP3);
    sds_free(OFMAP3);
    sds_free(IFMAP4);
    sds_free(OFMAP4);

/*
    free(IFMAP);
    free(OFMAP);
    free(IFMAP2);
    free(OFMAP2);
    free(IFMAP3);
    free(OFMAP3);
    free(IFMAP4);
    free(OFMAP4);
*/

//    printf("Entering compute in sw mode! \n");
//    for(j = 0; j < N; j=j+P){
//      gemm_nn_sw(M, P, K, ALPHA,lda, B, ldb,C,ldc,D,j,N);
//    }
//    C = D;

}

#ifdef GPU

#include <math.h>

void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    float *A_gpu = cuda_make_array(A, (TA ? lda*K:lda*M));
    float *B_gpu = cuda_make_array(B, (TB ? ldb*N : ldb*K));
    float *C_gpu = cuda_make_array(C, ldc*M);

    gemm_ongpu(TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc);

    cuda_pull_array(C_gpu, C, ldc*M);
    cuda_free(A_gpu);
    cuda_free(B_gpu);
    cuda_free(C_gpu);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_ongpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_ongpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_ongpu(0,0,64,2916,363); 
       time_ongpu(0,0,64,2916,363); 
       time_ongpu(0,0,64,2916,363); 
       time_ongpu(0,0,192,729,1600); 
       time_ongpu(0,0,384,196,1728); 
       time_ongpu(0,0,256,196,3456); 
       time_ongpu(0,0,256,196,2304); 
       time_ongpu(0,0,128,4096,12544); 
       time_ongpu(0,0,128,4096,4096); 
     */
    time_ongpu(0,0,64,75,12544); 
    time_ongpu(0,0,64,75,12544); 
    time_ongpu(0,0,64,75,12544); 
    time_ongpu(0,0,64,576,12544); 
    time_ongpu(0,0,256,2304,784); 
    time_ongpu(1,1,2304,256,784); 
    time_ongpu(0,0,512,4608,196); 
    time_ongpu(1,1,4608,512,196); 

    return 0;
}
#endif

void init1_weights(float Weights[16][27]) {
  Weights[0][ 0]  = -0.064049;
  Weights[0][ 1]  = -0.045396;
  Weights[0][ 2]  = -0.037942;
  Weights[0][ 3]  = -0.110669;
  Weights[0][ 4]  = -0.142887;
  Weights[0][ 5]  = -0.096666;
  Weights[0][ 6]  = -0.059662;
  Weights[0][ 7]  = -0.140883;
  Weights[0][ 8]  = -0.077999;
  Weights[0][ 9]  = -0.029043;
  Weights[0][10]  = -0.015262;
  Weights[0][11]  = 0.016828;
  Weights[0][12]  = -0.086994;
  Weights[0][13]  = -0.098943;
  Weights[0][14]  = -0.057030;
  Weights[0][15]  = -0.095970;
  Weights[0][16]  = -0.142695;
  Weights[0][17]  = -0.085212;
  Weights[0][18]  = 0.035949;
  Weights[0][19]  = 0.021758;
  Weights[0][20]  = 0.069699;
  Weights[0][21]  = -0.016599;
  Weights[0][22]  = -0.039375;
  Weights[0][23]  = 0.016566;
  Weights[0][24]  = 0.009513;
  Weights[0][25]  = -0.060279;
  Weights[0][26]  = 0.011305;
//  Weights[0][27]  = 0.011305;
  Weights[1][ 0]  = -0.367249;
  Weights[1][ 1]  = -0.539303;
  Weights[1][ 2]  = -0.322591;
  Weights[1][ 3]  = 0.060934;
  Weights[1][ 4]  = -0.069708;
  Weights[1][ 5]  = 0.020492;
  Weights[1][ 6]  = 0.354283;
  Weights[1][ 7]  = 0.484615;
  Weights[1][ 8]  = 0.411248;
  Weights[1][ 9]  = -0.386754;
  Weights[1][10]  = -0.575147;
  Weights[1][11]  = -0.350489;
  Weights[1][12]  = 0.077713;
  Weights[1][13]  = -0.037349;
  Weights[1][14]  = 0.011587;
  Weights[1][15]  = 0.329123;
  Weights[1][16]  = 0.488097;
  Weights[1][17]  = 0.373668;
  Weights[1][18]  = -0.274666;
  Weights[1][19]  = -0.386286;
  Weights[1][20]  = -0.252450;
  Weights[1][21]  = 0.081470;
  Weights[1][22]  = -0.000253;
  Weights[1][23]  = 0.027131;
  Weights[1][24]  = 0.244731;
  Weights[1][25]  = 0.345035;
  Weights[1][26]  = 0.271151;
//  Weights[1][27]  = 0.271151;
  Weights[2][ 0]  = 0.048910;
  Weights[2][ 1]  = 0.044255;
  Weights[2][ 2]  = 0.013175;
  Weights[2][ 3]  = 0.072052;
  Weights[2][ 4]  = 0.100945;
  Weights[2][ 5]  = 0.040982;
  Weights[2][ 6]  = 0.061299;
  Weights[2][ 7]  = 0.056611;
  Weights[2][ 8]  = 0.007536;
  Weights[2][ 9]  = 0.065697;
  Weights[2][10]  = 0.074062;
  Weights[2][11]  = 0.066363;
  Weights[2][12]  = 0.096432;
  Weights[2][13]  = 0.133825;
  Weights[2][14]  = 0.110367;
  Weights[2][15]  = 0.079579;
  Weights[2][16]  = 0.077771;
  Weights[2][17]  = 0.051433;
  Weights[2][18]  = -0.079734;
  Weights[2][19]  = -0.112314;
  Weights[2][20]  = -0.098023;
  Weights[2][21]  = -0.100942;
  Weights[2][22]  = -0.130901;
  Weights[2][23]  = -0.110407;
  Weights[2][24]  = -0.144683;
  Weights[2][25]  = -0.186492;
  Weights[2][26]  = -0.160715;
//  Weights[2][27]  = -0.160715;
  Weights[3][ 0]  = 0.020491;
  Weights[3][ 1]  = 0.018107;
  Weights[3][ 2]  = 0.017737;
  Weights[3][ 3]  = 0.021935;
  Weights[3][ 4]  = 0.012123;
  Weights[3][ 5]  = 0.030554;
  Weights[3][ 6]  = 0.043495;
  Weights[3][ 7]  = 0.037545;
  Weights[3][ 8]  = 0.052784;
  Weights[3][ 9]  = 0.066645;
  Weights[3][10]  = 0.057874;
  Weights[3][11]  = 0.076482;
  Weights[3][12]  = 0.047373;
  Weights[3][13]  = 0.017640;
  Weights[3][14]  = 0.060144;
  Weights[3][15]  = 0.086085;
  Weights[3][16]  = 0.044519;
  Weights[3][17]  = 0.092652;
  Weights[3][18]  = 0.075497;
  Weights[3][19]  = 0.081775;
  Weights[3][20]  = 0.085064;
  Weights[3][21]  = 0.074896;
  Weights[3][22]  = 0.056466;
  Weights[3][23]  = 0.068583;
  Weights[3][24]  = 0.086938;
  Weights[3][25]  = 0.055582;
  Weights[3][26]  = 0.087979;
//  Weights[3][27]  = 0.087979;
  Weights[4][ 0] = -0.285478;
  Weights[4][ 1] = -0.019653;
  Weights[4][ 2] = 0.345278;
  Weights[4][ 3] = -0.438799;
  Weights[4][ 4] = -0.037588;
  Weights[4][ 5] = 0.436483;
  Weights[4][ 6] = -0.383369;
  Weights[4][ 7] = 0.051548;
  Weights[4][ 8] = 0.366270;
  Weights[4][ 9] = -0.329909;
  Weights[4][10] = -0.026821;
  Weights[4][11] = 0.334411;
  Weights[4][12] = -0.514832;
  Weights[4][13] = -0.039002;
  Weights[4][14] = 0.460419;
  Weights[4][15] = -0.402291;
  Weights[4][16] = 0.058740;
  Weights[4][17] = 0.363699;
  Weights[4][18] = -0.206056;
  Weights[4][19] = -0.005059;
  Weights[4][20] = 0.224419;
  Weights[4][21] = -0.321306;
  Weights[4][22] = -0.010215;
  Weights[4][23] = 0.298698;
  Weights[4][24] = -0.279343;
  Weights[4][25] = 0.070275;
  Weights[4][26] = 0.261606;
//  Weights[4][27] = 0.261606;
  Weights[5][ 0] = 0.242569;
  Weights[5][ 1] = 0.116060;
  Weights[5][ 2] = -0.013415;
  Weights[5][ 3] = -0.419785;
  Weights[5][ 4] = -0.326175;
  Weights[5][ 5] = -0.022160;
  Weights[5][ 6] = 0.160009;
  Weights[5][ 7] = 0.198714;
  Weights[5][ 8] = 0.038484;
  Weights[5][ 9] = 0.294787;
  Weights[5][10] = 0.137222;
  Weights[5][11] = -0.010351;
  Weights[5][12] = -0.501659;
  Weights[5][13] = -0.383250;
  Weights[5][14] = -0.023648;
  Weights[5][15] = 0.214822;
  Weights[5][16] = 0.259429;
  Weights[5][17] = 0.045957;
  Weights[5][18] = 0.189860;
  Weights[5][19] = 0.076730;
  Weights[5][20] = -0.005162;
  Weights[5][21] = -0.331359;
  Weights[5][22] = -0.249885;
  Weights[5][23] = -0.014541;
  Weights[5][24] = 0.131527;
  Weights[5][25] = 0.152464;
  Weights[5][26] = 0.013625;
//  Weights[5][27] = 0.013625;
  Weights[6][ 0] = -0.203777;
  Weights[6][ 1] = -0.108901;
  Weights[6][ 2] = -0.045687;
  Weights[6][ 3] = -0.096644;
  Weights[6][ 4] = 0.152983;
  Weights[6][ 5] = 0.159629;
  Weights[6][ 6] = -0.121730;
  Weights[6][ 7] = 0.062618;
  Weights[6][ 8] = 0.197030;
  Weights[6][ 9] = -0.240978;
  Weights[6][10] = -0.091014;
  Weights[6][11] = -0.010481;
  Weights[6][12] = -0.146597;
  Weights[6][13] = 0.182564;
  Weights[6][14] = 0.181673;
  Weights[6][15] = -0.146679;
  Weights[6][16] = 0.096083;
  Weights[6][17] = 0.218333;
  Weights[6][18] = -0.139365;
  Weights[6][19] = -0.060262;
  Weights[6][20] = -0.010551;
  Weights[6][21] = -0.104560;
  Weights[6][22] = 0.111869;
  Weights[6][23] = 0.103572;
  Weights[6][24] = -0.085335;
  Weights[6][25] = 0.025382;
  Weights[6][26] = 0.116080;
//  Weights[6][27] = 0.116080;
  Weights[7][ 0] = 0.097371;
  Weights[7][ 1] = 0.167597;
  Weights[7][ 2] = 0.110650;
  Weights[7][ 3] = -0.036876;
  Weights[7][ 4] = 0.017842;
  Weights[7][ 5] = 0.057932;
  Weights[7][ 6] = -0.150641;
  Weights[7][ 7] = -0.139692;
  Weights[7][ 8] = -0.046319;
  Weights[7][ 9] = 0.077428;
  Weights[7][10] = 0.060498;
  Weights[7][11] = 0.038227;
  Weights[7][12] = 0.105800;
  Weights[7][13] = 0.107418;
  Weights[7][14] = 0.083196;
  Weights[7][15] = 0.109122;
  Weights[7][16] = 0.110351;
  Weights[7][17] = 0.086962;
  Weights[7][18] = -0.098519;
  Weights[7][19] = -0.133139;
  Weights[7][20] = -0.061195;
  Weights[7][21] = 0.035622;
  Weights[7][22] = 0.010370;
  Weights[7][23] = -0.015952;
  Weights[7][24] = 0.099417;
  Weights[7][25] = 0.131931;
  Weights[7][26] = 0.025812;
//  Weights[7][27] = 0.025812;
  Weights[8][ 0] = 0.156707;
  Weights[8][ 1] = -0.356221;
  Weights[8][ 2] = 0.210613;
  Weights[8][ 3] = 0.246392;
  Weights[8][ 4] = -0.442155;
  Weights[8][ 5] = 0.194008;
  Weights[8][ 6] = 0.179819;
  Weights[8][ 7] = -0.257459;
  Weights[8][ 8] = 0.072929;
  Weights[8][ 9] = 0.171351;
  Weights[8][10] = -0.465199;
  Weights[8][11] = 0.268176;
  Weights[8][12] = 0.299898;
  Weights[8][13] = -0.559181;
  Weights[8][14] = 0.245532;
  Weights[8][15] = 0.214966;
  Weights[8][16] = -0.278101;
  Weights[8][17] = 0.091597;
  Weights[8][18] = 0.110905;
  Weights[8][19] = -0.271801;
  Weights[8][20] = 0.148567;
  Weights[8][21] = 0.172845;
  Weights[8][22] = -0.315583;
  Weights[8][23] = 0.144872;
  Weights[8][24] = 0.114207;
  Weights[8][25] = -0.175049;
  Weights[8][26] = 0.057914;
//  Weights[8][27] = 0.057914;
  Weights[9][ 0] = 0.031307;
  Weights[9][ 1] = 0.058963;
  Weights[9][ 2] = -0.007255;
  Weights[9][ 3] = 0.051301;
  Weights[9][ 4] = 0.100871;
  Weights[9][ 5] = 0.062653;
  Weights[9][ 6] = 0.035226;
  Weights[9][ 7] = 0.045940;
  Weights[9][ 8] = 0.023455;
  Weights[9][ 9] = -0.077770;
  Weights[9][10] = -0.054609;
  Weights[9][11] = -0.116076;
  Weights[9][12] = -0.141094;
  Weights[9][13] = -0.115833;
  Weights[9][14] = -0.117307;
  Weights[9][15] = -0.201489;
  Weights[9][16] = -0.226906;
  Weights[9][17] = -0.198016;
  Weights[9][18] = 0.119785;
  Weights[9][19] = 0.153772;
  Weights[9][20] = 0.078824;
  Weights[9][21] = 0.119333;
  Weights[9][22] = 0.160290;
  Weights[9][23] = 0.140711;
  Weights[9][24] = 0.087031;
  Weights[9][25] = 0.081911;
  Weights[9][26] = 0.076539;
//  Weights[9][27] = 0.076539;
  Weights[10][ 0] = 0.326242;
  Weights[10][ 1] = 0.040075;
  Weights[10][ 2] = -0.365727;
  Weights[10][ 3] = 0.450917;
  Weights[10][ 4] = -0.042975;
  Weights[10][ 5] = -0.464367;
  Weights[10][ 6] = 0.403586;
  Weights[10][ 7] = 0.024881;
  Weights[10][ 8] = -0.331582;
  Weights[10][ 9] = 0.347123;
  Weights[10][10] = 0.043263;
  Weights[10][11] = -0.422495;
  Weights[10][12] = 0.487444;
  Weights[10][13] = -0.046382;
  Weights[10][14] = -0.547478;
  Weights[10][15] = 0.385702;
  Weights[10][16] = 0.016303;
  Weights[10][17] = -0.349749;
  Weights[10][18] = 0.201352;
  Weights[10][19] = 0.042346;
  Weights[10][20] = -0.256398;
  Weights[10][21] = 0.289306;
  Weights[10][22] = -0.025776;
  Weights[10][23] = -0.322066;
  Weights[10][24] = 0.259447;
  Weights[10][25] = 0.036159;
  Weights[10][26] = -0.194817;
//  Weights[10][27] = -0.194817;
  Weights[11][ 0] = 0.060476;
  Weights[11][ 1] = -0.026369;
  Weights[11][ 2] = 0.094858;
  Weights[11][ 3] = -0.101062;
  Weights[11][ 4] = -0.200735;
  Weights[11][ 5] = 0.013529;
  Weights[11][ 6] = -0.054191;
  Weights[11][ 7] = -0.082036;
  Weights[11][ 8] = 0.157404;
  Weights[11][ 9] = 0.050235;
  Weights[11][10] = 0.007907;
  Weights[11][11] = 0.062751;
  Weights[11][12] = -0.059916;
  Weights[11][13] = -0.125095;
  Weights[11][14] = 0.017297;
  Weights[11][15] = -0.072896;
  Weights[11][16] = -0.077088;
  Weights[11][17] = 0.071014;
  Weights[11][18] = -0.018860;
  Weights[11][19] = -0.001331;
  Weights[11][20] = -0.013083;
  Weights[11][21] = -0.068675;
  Weights[11][22] = -0.065929;
  Weights[11][23] = -0.019279;
  Weights[11][24] = -0.097682;
  Weights[11][25] = -0.049194;
  Weights[11][26] = 0.012213;
//  Weights[11][27] = 0.012213;
  Weights[12][ 0] = -0.154086;
  Weights[12][ 1] = -0.151793;
  Weights[12][ 2] = -0.128748;
  Weights[12][ 3] = -0.159260;
  Weights[12][ 4] = -0.148493;
  Weights[12][ 5] = -0.094127;
  Weights[12][ 6] = -0.231925;
  Weights[12][ 7] = -0.180055;
  Weights[12][ 8] = -0.097392;
  Weights[12][ 9] = 0.149652;
  Weights[12][10] = 0.141412;
  Weights[12][11] = 0.113743;
  Weights[12][12] = 0.158771;
  Weights[12][13] = 0.143767;
  Weights[12][14] = 0.133898;
  Weights[12][15] = 0.089024;
  Weights[12][16] = 0.091712;
  Weights[12][17] = 0.092802;
  Weights[12][18] = 0.078120;
  Weights[12][19] = 0.068321;
  Weights[12][20] = -0.026740;
  Weights[12][21] = 0.106323;
  Weights[12][22] = 0.081434;
  Weights[12][23] = -0.012973;
  Weights[12][24] = 0.057619;
  Weights[12][25] = 0.042367;
  Weights[12][26] = -0.067043;
//  Weights[12][27] = -0.067043;
  Weights[13][ 0] = 0.380687;
  Weights[13][ 1] = 0.505032;
  Weights[13][ 2] = 0.367969;
  Weights[13][ 3] = -0.011850;
  Weights[13][ 4] = -0.056445;
  Weights[13][ 5] = 0.057177;
  Weights[13][ 6] = -0.352487;
  Weights[13][ 7] = -0.534827;
  Weights[13][ 8] = -0.354795;
  Weights[13][ 9] = 0.404564;
  Weights[13][10] = 0.551341;
  Weights[13][11] = 0.387640;
  Weights[13][12] = -0.003815;
  Weights[13][13] = -0.058395;
  Weights[13][14] = 0.061942;
  Weights[13][15] = -0.372085;
  Weights[13][16] = -0.604489;
  Weights[13][17] = -0.381647;
  Weights[13][18] = 0.275405;
  Weights[13][19] = 0.337972;
  Weights[13][20] = 0.239537;
  Weights[13][21] = 0.013591;
  Weights[13][22] = -0.053855;
  Weights[13][23] = 0.037433;
  Weights[13][24] = -0.231390;
  Weights[13][25] = -0.390020;
  Weights[13][26] = -0.257715;
//  Weights[13][27] = -0.257715;
  Weights[14][ 0] = 0.070536;
  Weights[14][ 1] = 0.060445;
  Weights[14][ 2] = 0.090112;
  Weights[14][ 3] = 0.033577;
  Weights[14][ 4] = 0.002713;
  Weights[14][ 5] = 0.080030;
  Weights[14][ 6] = 0.012857;
  Weights[14][ 7] = 0.036756;
  Weights[14][ 8] = 0.093639;
  Weights[14][ 9] = -0.076698;
  Weights[14][10] = -0.191530;
  Weights[14][11] = -0.169015;
  Weights[14][12] = -0.179441;
  Weights[14][13] = -0.306897;
  Weights[14][14] = -0.229474;
  Weights[14][15] = -0.160911;
  Weights[14][16] = -0.204887;
  Weights[14][17] = -0.154613;
  Weights[14][18] = 0.167764;
  Weights[14][19] = 0.104697;
  Weights[14][20] = 0.109872;
  Weights[14][21] = 0.112747;
  Weights[14][22] = 0.030831;
  Weights[14][23] = 0.090656;
  Weights[14][24] = 0.136328;
  Weights[14][25] = 0.122764;
  Weights[14][26] = 0.158473;
//  Weights[14][27] = 0.158473;
  Weights[15][ 0] = 0.181205;
  Weights[15][ 1] = 0.223681;
  Weights[15][ 2] = 0.170409;
  Weights[15][ 3] = 0.217128;
  Weights[15][ 4] = 0.264190;
  Weights[15][ 5] = 0.178473;
  Weights[15][ 6] = 0.176698;
  Weights[15][ 7] = 0.208407;
  Weights[15][ 8] = 0.133560;
  Weights[15][ 9] = -0.162913;
  Weights[15][10] = -0.188261;
  Weights[15][11] = -0.164639;
  Weights[15][12] = -0.188550;
  Weights[15][13] = -0.219027;
  Weights[15][14] = -0.176282;
  Weights[15][15] = -0.197434;
  Weights[15][16] = -0.204081;
  Weights[15][17] = -0.173185;
  Weights[15][18] = 0.003420;
  Weights[15][19] = -0.033567;
  Weights[15][20] = 0.003733;
  Weights[15][21] = -0.004463;
  Weights[15][22] = -0.050087;
  Weights[15][23] = 0.017009;
  Weights[15][24] = 0.001786;
  Weights[15][25] = -0.013393;
  Weights[15][26] = 0.030971;
//  Weights[15][27] = 0.030971;
}


void init_weights(float Weights[432]) {
  Weights[0]   = -0.064049;
  Weights[1]   = -0.045396;
  Weights[2]   = -0.037942;
  Weights[3]   = -0.110669;
  Weights[4]   = -0.142887;
  Weights[5]   = -0.096666;
  Weights[6]   = -0.059662;
  Weights[7]   = -0.140883;
  Weights[8]   = -0.077999;
  Weights[9]   = -0.029043;
  Weights[10]  = -0.015262;
  Weights[11]  = 0.016828;
  Weights[12]  = -0.086994;
  Weights[13]  = -0.098943;
  Weights[14]  = -0.057030;
  Weights[15]  = -0.095970;
  Weights[16]  = -0.142695;
  Weights[17]  = -0.085212;
  Weights[18]  = 0.035949;
  Weights[19]  = 0.021758;
  Weights[20]  = 0.069699;
  Weights[21]  = -0.016599;
  Weights[22]  = -0.039375;
  Weights[23]  = 0.016566;
  Weights[24]  = 0.009513;
  Weights[25]  = -0.060279;
  Weights[26]  = 0.011305;
  Weights[27]  = -0.367249;
  Weights[28]  = -0.539303;
  Weights[29]  = -0.322591;
  Weights[30]  = 0.060934;
  Weights[31]  = -0.069708;
  Weights[32]  = 0.020492;
  Weights[33]  = 0.354283;
  Weights[34]  = 0.484615;
  Weights[35]  = 0.411248;
  Weights[36]  = -0.386754;
  Weights[37]  = -0.575147;
  Weights[38]  = -0.350489;
  Weights[39]  = 0.077713;
  Weights[40]  = -0.037349;
  Weights[41]  = 0.011587;
  Weights[42]  = 0.329123;
  Weights[43]  = 0.488097;
  Weights[44]  = 0.373668;
  Weights[45]  = -0.274666;
  Weights[46]  = -0.386286;
  Weights[47]  = -0.252450;
  Weights[48]  = 0.081470;
  Weights[49]  = -0.000253;
  Weights[50]  = 0.027131;
  Weights[51]  = 0.244731;
  Weights[52]  = 0.345035;
  Weights[53]  = 0.271151;
  Weights[54]  = 0.048910;
  Weights[55]  = 0.044255;
  Weights[56]  = 0.013175;
  Weights[57]  = 0.072052;
  Weights[58]  = 0.100945;
  Weights[59]  = 0.040982;
  Weights[60]  = 0.061299;
  Weights[61]  = 0.056611;
  Weights[62]  = 0.007536;
  Weights[63]  = 0.065697;
  Weights[64]  = 0.074062;
  Weights[65]  = 0.066363;
  Weights[66]  = 0.096432;
  Weights[67]  = 0.133825;
  Weights[68]  = 0.110367;
  Weights[69]  = 0.079579;
  Weights[70]  = 0.077771;
  Weights[71]  = 0.051433;
  Weights[72]  = -0.079734;
  Weights[73]  = -0.112314;
  Weights[74]  = -0.098023;
  Weights[75]  = -0.100942;
  Weights[76]  = -0.130901;
  Weights[77]  = -0.110407;
  Weights[78]  = -0.144683;
  Weights[79]  = -0.186492;
  Weights[80]  = -0.160715;
  Weights[81]  = 0.020491;
  Weights[82]  = 0.018107;
  Weights[83]  = 0.017737;
  Weights[84]  = 0.021935;
  Weights[85]  = 0.012123;
  Weights[86]  = 0.030554;
  Weights[87]  = 0.043495;
  Weights[88]  = 0.037545;
  Weights[89]  = 0.052784;
  Weights[90]  = 0.066645;
  Weights[91]  = 0.057874;
  Weights[92]  = 0.076482;
  Weights[93]  = 0.047373;
  Weights[94]  = 0.017640;
  Weights[95]  = 0.060144;
  Weights[96]  = 0.086085;
  Weights[97]  = 0.044519;
  Weights[98]  = 0.092652;
  Weights[99]  = 0.075497;
  Weights[100] = 0.081775;
  Weights[101] = 0.085064;
  Weights[102] = 0.074896;
  Weights[103] = 0.056466;
  Weights[104] = 0.068583;
  Weights[105] = 0.086938;
  Weights[106] = 0.055582;
  Weights[107] = 0.087979;
  Weights[108] = -0.285478;
  Weights[109] = -0.019653;
  Weights[110] = 0.345278;
  Weights[111] = -0.438799;
  Weights[112] = -0.037588;
  Weights[113] = 0.436483;
  Weights[114] = -0.383369;
  Weights[115] = 0.051548;
  Weights[116] = 0.366270;
  Weights[117] = -0.329909;
  Weights[118] = -0.026821;
  Weights[119] = 0.334411;
  Weights[120] = -0.514832;
  Weights[121] = -0.039002;
  Weights[122] = 0.460419;
  Weights[123] = -0.402291;
  Weights[124] = 0.058740;
  Weights[125] = 0.363699;
  Weights[126] = -0.206056;
  Weights[127] = -0.005059;
  Weights[128] = 0.224419;
  Weights[129] = -0.321306;
  Weights[130] = -0.010215;
  Weights[131] = 0.298698;
  Weights[132] = -0.279343;
  Weights[133] = 0.070275;
  Weights[134] = 0.261606;
  Weights[135] = 0.242569;
  Weights[136] = 0.116060;
  Weights[137] = -0.013415;
  Weights[138] = -0.419785;
  Weights[139] = -0.326175;
  Weights[140] = -0.022160;
  Weights[141] = 0.160009;
  Weights[142] = 0.198714;
  Weights[143] = 0.038484;
  Weights[144] = 0.294787;
  Weights[145] = 0.137222;
  Weights[146] = -0.010351;
  Weights[147] = -0.501659;
  Weights[148] = -0.383250;
  Weights[149] = -0.023648;
  Weights[150] = 0.214822;
  Weights[151] = 0.259429;
  Weights[152] = 0.045957;
  Weights[153] = 0.189860;
  Weights[154] = 0.076730;
  Weights[155] = -0.005162;
  Weights[156] = -0.331359;
  Weights[157] = -0.249885;
  Weights[158] = -0.014541;
  Weights[159] = 0.131527;
  Weights[160] = 0.152464;
  Weights[161] = 0.013625;
  Weights[162] = -0.203777;
  Weights[163] = -0.108901;
  Weights[164] = -0.045687;
  Weights[165] = -0.096644;
  Weights[166] = 0.152983;
  Weights[167] = 0.159629;
  Weights[168] = -0.121730;
  Weights[169] = 0.062618;
  Weights[170] = 0.197030;
  Weights[171] = -0.240978;
  Weights[172] = -0.091014;
  Weights[173] = -0.010481;
  Weights[174] = -0.146597;
  Weights[175] = 0.182564;
  Weights[176] = 0.181673;
  Weights[177] = -0.146679;
  Weights[178] = 0.096083;
  Weights[179] = 0.218333;
  Weights[180] = -0.139365;
  Weights[181] = -0.060262;
  Weights[182] = -0.010551;
  Weights[183] = -0.104560;
  Weights[184] = 0.111869;
  Weights[185] = 0.103572;
  Weights[186] = -0.085335;
  Weights[187] = 0.025382;
  Weights[188] = 0.116080;
  Weights[189] = 0.097371;
  Weights[190] = 0.167597;
  Weights[191] = 0.110650;
  Weights[192] = -0.036876;
  Weights[193] = 0.017842;
  Weights[194] = 0.057932;
  Weights[195] = -0.150641;
  Weights[196] = -0.139692;
  Weights[197] = -0.046319;
  Weights[198] = 0.077428;
  Weights[199] = 0.060498;
  Weights[200] = 0.038227;
  Weights[201] = 0.105800;
  Weights[202] = 0.107418;
  Weights[203] = 0.083196;
  Weights[204] = 0.109122;
  Weights[205] = 0.110351;
  Weights[206] = 0.086962;
  Weights[207] = -0.098519;
  Weights[208] = -0.133139;
  Weights[209] = -0.061195;
  Weights[210] = 0.035622;
  Weights[211] = 0.010370;
  Weights[212] = -0.015952;
  Weights[213] = 0.099417;
  Weights[214] = 0.131931;
  Weights[215] = 0.025812;
  Weights[216] = 0.156707;
  Weights[217] = -0.356221;
  Weights[218] = 0.210613;
  Weights[219] = 0.246392;
  Weights[220] = -0.442155;
  Weights[221] = 0.194008;
  Weights[222] = 0.179819;
  Weights[223] = -0.257459;
  Weights[224] = 0.072929;
  Weights[225] = 0.171351;
  Weights[226] = -0.465199;
  Weights[227] = 0.268176;
  Weights[228] = 0.299898;
  Weights[229] = -0.559181;
  Weights[230] = 0.245532;
  Weights[231] = 0.214966;
  Weights[232] = -0.278101;
  Weights[233] = 0.091597;
  Weights[234] = 0.110905;
  Weights[235] = -0.271801;
  Weights[236] = 0.148567;
  Weights[237] = 0.172845;
  Weights[238] = -0.315583;
  Weights[239] = 0.144872;
  Weights[240] = 0.114207;
  Weights[241] = -0.175049;
  Weights[242] = 0.057914;
  Weights[243] = 0.031307;
  Weights[244] = 0.058963;
  Weights[245] = -0.007255;
  Weights[246] = 0.051301;
  Weights[247] = 0.100871;
  Weights[248] = 0.062653;
  Weights[249] = 0.035226;
  Weights[250] = 0.045940;
  Weights[251] = 0.023455;
  Weights[252] = -0.077770;
  Weights[253] = -0.054609;
  Weights[254] = -0.116076;
  Weights[255] = -0.141094;
  Weights[256] = -0.115833;
  Weights[257] = -0.117307;
  Weights[258] = -0.201489;
  Weights[259] = -0.226906;
  Weights[260] = -0.198016;
  Weights[261] = 0.119785;
  Weights[262] = 0.153772;
  Weights[263] = 0.078824;
  Weights[264] = 0.119333;
  Weights[265] = 0.160290;
  Weights[266] = 0.140711;
  Weights[267] = 0.087031;
  Weights[268] = 0.081911;
  Weights[269] = 0.076539;
  Weights[270] = 0.326242;
  Weights[271] = 0.040075;
  Weights[272] = -0.365727;
  Weights[273] = 0.450917;
  Weights[274] = -0.042975;
  Weights[275] = -0.464367;
  Weights[276] = 0.403586;
  Weights[277] = 0.024881;
  Weights[278] = -0.331582;
  Weights[279] = 0.347123;
  Weights[280] = 0.043263;
  Weights[281] = -0.422495;
  Weights[282] = 0.487444;
  Weights[283] = -0.046382;
  Weights[284] = -0.547478;
  Weights[285] = 0.385702;
  Weights[286] = 0.016303;
  Weights[287] = -0.349749;
  Weights[288] = 0.201352;
  Weights[289] = 0.042346;
  Weights[290] = -0.256398;
  Weights[291] = 0.289306;
  Weights[292] = -0.025776;
  Weights[293] = -0.322066;
  Weights[294] = 0.259447;
  Weights[295] = 0.036159;
  Weights[296] = -0.194817;
  Weights[297] = 0.060476;
  Weights[298] = -0.026369;
  Weights[299] = 0.094858;
  Weights[300] = -0.101062;
  Weights[301] = -0.200735;
  Weights[302] = 0.013529;
  Weights[303] = -0.054191;
  Weights[304] = -0.082036;
  Weights[305] = 0.157404;
  Weights[306] = 0.050235;
  Weights[307] = 0.007907;
  Weights[308] = 0.062751;
  Weights[309] = -0.059916;
  Weights[310] = -0.125095;
  Weights[311] = 0.017297;
  Weights[312] = -0.072896;
  Weights[313] = -0.077088;
  Weights[314] = 0.071014;
  Weights[315] = -0.018860;
  Weights[316] = -0.001331;
  Weights[317] = -0.013083;
  Weights[318] = -0.068675;
  Weights[319] = -0.065929;
  Weights[320] = -0.019279;
  Weights[321] = -0.097682;
  Weights[322] = -0.049194;
  Weights[323] = 0.012213;
  Weights[324] = -0.154086;
  Weights[325] = -0.151793;
  Weights[326] = -0.128748;
  Weights[327] = -0.159260;
  Weights[328] = -0.148493;
  Weights[329] = -0.094127;
  Weights[330] = -0.231925;
  Weights[331] = -0.180055;
  Weights[332] = -0.097392;
  Weights[333] = 0.149652;
  Weights[334] = 0.141412;
  Weights[335] = 0.113743;
  Weights[336] = 0.158771;
  Weights[337] = 0.143767;
  Weights[338] = 0.133898;
  Weights[339] = 0.089024;
  Weights[340] = 0.091712;
  Weights[341] = 0.092802;
  Weights[342] = 0.078120;
  Weights[343] = 0.068321;
  Weights[344] = -0.026740;
  Weights[345] = 0.106323;
  Weights[346] = 0.081434;
  Weights[347] = -0.012973;
  Weights[348] = 0.057619;
  Weights[349] = 0.042367;
  Weights[350] = -0.067043;
  Weights[351] = 0.380687;
  Weights[352] = 0.505032;
  Weights[353] = 0.367969;
  Weights[354] = -0.011850;
  Weights[355] = -0.056445;
  Weights[356] = 0.057177;
  Weights[357] = -0.352487;
  Weights[358] = -0.534827;
  Weights[359] = -0.354795;
  Weights[360] = 0.404564;
  Weights[361] = 0.551341;
  Weights[362] = 0.387640;
  Weights[363] = -0.003815;
  Weights[364] = -0.058395;
  Weights[365] = 0.061942;
  Weights[366] = -0.372085;
  Weights[367] = -0.604489;
  Weights[368] = -0.381647;
  Weights[369] = 0.275405;
  Weights[370] = 0.337972;
  Weights[371] = 0.239537;
  Weights[372] = 0.013591;
  Weights[373] = -0.053855;
  Weights[374] = 0.037433;
  Weights[375] = -0.231390;
  Weights[376] = -0.390020;
  Weights[377] = -0.257715;
  Weights[378] = 0.070536;
  Weights[379] = 0.060445;
  Weights[380] = 0.090112;
  Weights[381] = 0.033577;
  Weights[382] = 0.002713;
  Weights[383] = 0.080030;
  Weights[384] = 0.012857;
  Weights[385] = 0.036756;
  Weights[386] = 0.093639;
  Weights[387] = -0.076698;
  Weights[388] = -0.191530;
  Weights[389] = -0.169015;
  Weights[390] = -0.179441;
  Weights[391] = -0.306897;
  Weights[392] = -0.229474;
  Weights[393] = -0.160911;
  Weights[394] = -0.204887;
  Weights[395] = -0.154613;
  Weights[396] = 0.167764;
  Weights[397] = 0.104697;
  Weights[398] = 0.109872;
  Weights[399] = 0.112747;
  Weights[400] = 0.030831;
  Weights[401] = 0.090656;
  Weights[402] = 0.136328;
  Weights[403] = 0.122764;
  Weights[404] = 0.158473;
  Weights[405] = 0.181205;
  Weights[406] = 0.223681;
  Weights[407] = 0.170409;
  Weights[408] = 0.217128;
  Weights[409] = 0.264190;
  Weights[410] = 0.178473;
  Weights[411] = 0.176698;
  Weights[412] = 0.208407;
  Weights[413] = 0.133560;
  Weights[414] = -0.162913;
  Weights[415] = -0.188261;
  Weights[416] = -0.164639;
  Weights[417] = -0.188550;
  Weights[418] = -0.219027;
  Weights[419] = -0.176282;
  Weights[420] = -0.197434;
  Weights[421] = -0.204081;
  Weights[422] = -0.173185;
  Weights[423] = 0.003420;
  Weights[424] = -0.033567;
  Weights[425] = 0.003733;
  Weights[426] = -0.004463;
  Weights[427] = -0.050087;
  Weights[428] = 0.017009;
  Weights[429] = 0.001786;
  Weights[430] = -0.013393;
  Weights[431] = 0.030971;
}
