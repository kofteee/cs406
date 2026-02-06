// g++ dot_simd.cpp -march[need to provide the app option] -O3[or none] -fopt-info-vec 

#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <chrono>
using namespace std;

#define N (1 << 26)
//#define INFO
float dot_norm(float* a, float* b) {
  float sum = 0;
  for (int i = 0; i < N; i++){
    sum += a[i]*b[i];
  }
  return sum;
}

void dot_128(const float* p1, const float* p2, float& sum ) {
  __m128 acc = _mm_setzero_ps();

  const float* const p1End = p1 + N;
  for( ; p1 < p1End; p1 += 4, p2 += 4 ) {

    // Load 2 vectors, 4 floats / each                                                                                                                                                                                
    const __m128 a = _mm_loadu_ps( p1 );
    const __m128 b = _mm_loadu_ps( p2 );

    // Compute dot product of them. The 0xFF constant means "use all 4 source lanes, and broadcast the result into all 4 lanes of the destination".                                                                   
    const __m128 dp = _mm_dp_ps( a, b, 0xFF );
    acc = _mm_add_ps( acc, dp );
  }
  #ifdef INFO
    float* temp = (float*)&acc;
    cout << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << endl;
  #endif
  sum = _mm_cvtss_f32( acc );

}

void dot_256(const float* p1, const float* p2, float& sum ) {
  __m256 acc = _mm256_setzero_ps();
  const float* const p1End = p1 + N;
  for( ; p1 < p1End; p1 += 8, p2 += 8 ) {
      // Load 2 vectors, 8 floats / each                                                                                                                                                                                
      const __m256 a = _mm256_loadu_ps( p1 );
      const __m256 b = _mm256_loadu_ps( p2 );
      
      // vdpps AVX instruction does not compute dot product of 8-wide vectors.                                                                                                                                          
      // Instead, that instruction computes 2 independent dot products of 4-wide vectors.                                                                                                                               
      const __m256 dp = _mm256_dp_ps( a, b, 0xFF );
      acc = _mm256_add_ps( acc, dp );
    }


  #ifdef INFO
    float* temp = (float*)&acc;
    cout << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << " " << temp[4] << " " << temp[5] << " " << temp[6] << " " << temp[7] << endl;
  #endif
  
  // Add the 2 results into a single float.                                                                                                                                                                                 
  const __m128 low = _mm256_castps256_ps128( acc );       //< Compiles into no instructions. The low half of a YMM register is directly accessible as an XMM register with the same number.                                 
  const __m128 high = _mm256_extractf128_ps( acc, 1 );    //< This one however does need to move data, from high half of a register into low half. vextractf128 instruction does that.                                      
  const __m128 result = _mm_add_ss( low, high );
  sum = _mm_cvtss_f32( result );
}

void dot_256_vertical(const float* p1, const float* p2, float& sum ) {
  __m256 acc = _mm256_setzero_ps();
  const float* const p1End = p1 + N;
  for( ; p1 < p1End; p1 += 8, p2 += 8 ) {
      // Load 2 vectors, 8 floats / each                                                                                                                                                                                
      const __m256 a = _mm256_loadu_ps( p1 );
      const __m256 b = _mm256_loadu_ps( p2 );

      __m256 temp = _mm256_mul_ps(a, b);
      acc = _mm256_add_ps(temp, acc);
    }
 
  float* temp = (float*)&acc; 
#ifdef INFO
  cout << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << " " << temp[4] << " " << temp[5] << " " << temp[6] << " " << temp[7] << endl;
#endif
  
  sum = 0;
  for(int i = 0; i < 8; i++) {
    sum += *temp++;
  }
}


int main() {
  cout << "A float is " << sizeof(float) << " bytes "<< endl;

  float* a = new float[N]; 
  float* b = new float[N]; 
  
  for(int i = 0; i < N; i++) {
    a[i] = 1.0f; 
    b[i] = 1.0f;
  }
  
  float sum;

  auto t1 = std::chrono::high_resolution_clock::now();
  sum = dot_norm(a, b);
  auto t2 = std::chrono::high_resolution_clock::now();
  cout << sum << " dot_norm " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << " milliseconds\n";

  t1 = std::chrono::high_resolution_clock::now();
  dot_128(a, b, sum);
  t2 = std::chrono::high_resolution_clock::now();
  cout << sum << " dot_128 " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << " milliseconds\n";

  t1 = std::chrono::high_resolution_clock::now();
  dot_256(a, b, sum);
  t2 = std::chrono::high_resolution_clock::now();
  cout << sum << " dot_256 " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << " milliseconds\n";

  t1 = std::chrono::high_resolution_clock::now();
  dot_256_vertical(a, b, sum);
  t2 = std::chrono::high_resolution_clock::now();
  cout << sum << " dot_256_vertical " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << " milliseconds\n";

  return 0;
}
