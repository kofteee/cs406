/*
  MPI_Reduce: Combines data from all processes using a specified operation 
  (e.g., sum, max) and sends the result to the root process.
*/ 
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Each process contributes its rank as the data
  int send_data = world_rank;

  int sum;
  //int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
  //               MPI_Op op, int root, MPI_Comm comm)
  MPI_Reduce(&send_data, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  // Root process prints the sum of all ranks
  if (world_rank == 0) {
    printf("Sum of ranks is %d\n", sum);
  }

  MPI_Finalize();
  return 0;
}
