//compile with: g++ prefetch.cpp  -std=c++11 -O3 
//use -fopt-info-vec-optimized for vectorization info verbose
//run: ./a.out

#include <iostream>
#include <chrono>
using namespace std;

#define N 1024 * 1024 * 256

void sum_norm(int* a, int* b, int& sum) {
  sum = 0;
  for (int i = 0; i < N; i++){
    sum += a[i]*b[i];
  }
}

void prefetch(int* a, int* b, int& sum, int d) {
  sum = 0;
  
  for (int i = 0; i < N - d; i += 8){    
    //The first 0 is READ_ONLY
    //The second 1 is LOCALITY LOW
    __builtin_prefetch (&a[i+d], 0, 1);
    __builtin_prefetch (&b[i+d], 0, 1);

    for(int j = 0; j < 8; j++) {
      sum += a[i + j] * b[i + j]; 
    }
  }
  for (int i = N-d; i < N; i++)
    sum = sum + a[i]*b[i];
}


int main() {
  cout << "An integer is " << sizeof(int) << " bytes "<< endl;

  int* a = new int[N];
  int* b = new int[N];

  for(int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i;
  }
  cout << a << " " << b << endl;
  
  int sum;
  auto t1 = std::chrono::high_resolution_clock::now();
  sum_norm(a, b, sum);
  auto t2 = std::chrono::high_resolution_clock::now();
  cout << sum << " ns " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 50 << " milliseconds\n";

  cout << "-----------------------------------------" << endl;
  cout << "Starting prefetching: " << endl;

  for(int k = 0; k <= 1024; k += 8) { 
    t1 = std::chrono::high_resolution_clock::now();
    prefetch(a, b, sum, k);
    t2 = std::chrono::high_resolution_clock::now();
    cout << sum << " " << k << " " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << " milliseconds\n";
  }
  return 0;
}
