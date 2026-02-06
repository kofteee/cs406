/*
Key Concepts:
MPI_Send: Sends data to a specific process.
MPI_Recv: Receives data from a specific process.
MPI_STATUS_IGNORE: Ignores the status of the received message for simplicity.
*/

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  printf("Hello world from processor %s, rank %d out of %d processors\n",processor_name, world_rank, world_size);

  int number;
  if (world_rank == 0) {
    // Process 0 sends a number to Process 1
    for(int i = 1; i < world_size; i++) {
      number = 42 + i; // Data to be sent	  
      //int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
      MPI_Send(&number, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      printf("Process 0 sent number %d to process %d\n", number, i);
    }
  } else { 
    //A process receives the number from Process 0
    
    //int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status * status)      
    MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Process %d received number %d from process 0\n", world_rank, number);
  }

  MPI_Finalize();
  return 0;
}
