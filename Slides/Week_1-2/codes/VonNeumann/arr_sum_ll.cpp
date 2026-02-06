#include <iostream>
#include <chrono>
#include <algorithm>    // std::random_shuffle

using namespace std;
using namespace std::chrono;

typedef unsigned long long ull;

long long arrsum(ull* A, ull* B, int n) {  
  auto start = high_resolution_clock::now();  
  ull sum = 0;
  for(int i = 0; i < n; i++){
    sum += A[B[i]];
  }
  auto stop = high_resolution_clock::now();
  
  auto duration = duration_cast<milliseconds>(stop - start);  
  cout << "Time taken by function: "
       << duration.count() << " milliseconds" << endl;
  return sum;

}

int main(int argc, char** argv){
  int n = 1 << atoi(argv[2]);
  ull* A = new ull[n];
  ull* B = new ull[n];
  
  for(int i = 0; i < n; i++){
    A[i] = rand();
    B[i] = i;
  }

  long long sum;

#ifdef RANDOMIZE
  random_shuffle(B, B + n);
#endif
  cout << "Experiment starts: " << endl;
  for(int i = 0; i < atoi(argv[1]); i++) {
    sum = arrsum(A, B, n);
  }
  cout << "Sum is " << sum << endl;

  return 0;
}
