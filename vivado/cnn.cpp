#include "cnn.h"
#include <stdio.h>
using namespace std;
//----------------------------------------------------------
// Top function
//----------------------------------------------------------
//#include "sds_lib.h"
#define ALPHA 1	
#define array_size(x,y) x*y
#define M 16
#define K 27
#define N 600

void dut(
    hls::stream<float> &strm_in,
    hls::stream<float> &strm_out
)
{

 // int M = strm_in.read();
 // int N = strm_in.read();
 // int K = strm_in.read();
  // int M=16,K=27,N=600; 
  float B[array_size(M,N)], C[array_size(K,N)];
   
  //  for (int i = 0; i < M*K; ++i ) {
  //          if ( !strm_in.empty() ) {
  //          A[i]= strm_in.read();
  //        }
  //  }
 
     for (int i = 0; i < N*K; ++i ) {
            if ( !strm_in.empty() ) {
            B[i] = strm_in.read();
           }
     }

     for (int i =0; i<M*N; i++) {
      
             C[i]=0;
     } 
     cnn_compute(C,B,C);  

     for (int i =0; i<M*N; i++) {
      
             strm_out.write(C[i]);
     } 

 
}



void cnn_compute( float D[array_size(N,M)], float B[array_size(K,N)], float C[array_size(M,N)]) {
    

   // int M=16,K=27,N=600; 

    int lda = K, ldb= N , ldc = N; 


  //  #pragma HLS array_partition variable=D block factor=600 dim=2
    int i,j,k;
   // float Weights[432];
    float W[M][K];
    // float tmp[array_size(16,600)];
   // float tmp[M][N];
    // float B_hw[array_size(16,600)];

    init_weights_hw(W);
   
  float c_buf[M][N];
  float b_buf[K][N];

  // Transfer matrix A from multi-buffer into local RAM
  for(i=0; i< K; i++) {
    for(j=0; j< N; j++) {
   
     #pragma HLS PIPELINE II=1
      b_buf[i][j] = B[i * N + j];
    }
  }

   for(i=0; i< M; i++) {
    for(j=0; j< N; j++) {
   
     #pragma HLS PIPELINE II=1
      c_buf[i][j] = C[i * N + j];
    }
  }
  

  //  for(i = 0; i < M; ++i){
  //    for(j = 0; j < ldc; ++j){
  //       
  //         #pragma HLS PIPELINE II=1
  //      // #pragma HLS unroll
  //      // #pragma HLS ARRAY_PARTITION variable=tmp complete dim=0
  //         tmp[i][j] = D[i*ldc+j];
  //    }
  //  }
  
 // for(i=0; i< M; i++) {
 //   for(j=0; j< K; j++) {
 //  
 //    #pragma HLS PIPELINE II=1
 //     W[i][j] = Weights[i * K + j];
 //   }
 // }
   cnn_compute_kernel(W, b_buf, c_buf);           
 // for(i = 0; i < M; ++i){
 //     for(j = 0; j < N; ++j){
 // 	      c_buf[i][j] = tmp[i][j];
 //     }
 //  }
 for(i = 0; i < M; ++i){
     for(j = 0; j < N; ++j){
           #pragma HLS PIPELINE II=1
 	    C[i*N+j]  = c_buf[i][j];
     }
  }

}

void cnn_compute_kernel(float W[M][K], float b_buf[K][N], float c_buf[M][N])
{

  #pragma HLS INLINE self
 // #pragma HLS array_partition variable=b_buf block factor=9 dim=1
 //  #pragma HLS array_partition variable=W block factor=9 dim=2
 //  #pragma HLS array_partition variable=b_buf dim=0
 //  #pragma HLS array_partition variable=W dim=0
  #pragma HLS array_partition variable=b_buf cyclic factor=27 dim=1
 // #pragma HLS array_partition variable=c_buf block factor=14 dim=2
 #pragma HLS array_partition variable=W cyclic factor=27 dim=2
// int M=16,K=27,N=600; 

  int i,j,k;
  float temp;
  L1: for (i = 0; i < M; i++) {
  
                     // #pragma HLS pipeline
              L2: for(j = 0; j < N; ++j){
                       #pragma HLS pipeline
                       // #pragma HLS unroll
                      float result = 0;
                L3:      for(k = 0; k < K; ++k){
                       
                        // #pragma HLS unroll
                        // register float A_PART = ALPHA*W[i][k];
                        // tmp[i][j] += A_PART*b_buf[k][j];
                        temp = W[i][k]*b_buf[k][j];
                        result += temp;
            }
            c_buf[i][j] = result;
        }
   }

}

void init_weights_hw(float Weights[16][27]) {

  // #pragma HLS unroll
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

