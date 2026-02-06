/*
  MPI_Bcast: Sends data from one process to all other processes in the communicator.
*/

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int number;
  if (world_rank == 0) {
    // The root process initializes the number
    number = 100;
  }

  // Broadcast the number to all processes
  //int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
  MPI_Bcast(&number, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // All processes now have the number
  printf("Process %d received number %d\n", world_rank, number);

  MPI_Finalize();
  return 0;
}
