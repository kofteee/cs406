/*
   MPI_Gather: Collects data from all processes and gathers it on the root process.
*/

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Each process contributes its rank
  int send_data = world_rank;

  int recv_data[world_size]; // Only meaningful on root process
  //int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
  //               void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
  
  MPI_Gather(&send_data, 1, MPI_INT, recv_data, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Root process prints gathered data
  if (world_rank == 0) {
    printf("Gathered data at root: ");
    for (int i = 0; i < world_size; i++) {
      printf("%d ", recv_data[i]);
    }
    printf("\n");
  }

  MPI_Finalize();
  return 0;
}
