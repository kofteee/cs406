/*
Key Concepts:                                                                                                                        
MPI_Comm_size: Retrieves the total number of processes.
MPI_Comm_rank: Retrieves the rank of the calling process.
MPI_Get_processor_name: Gets the name of the processor.
                                                                                                                                     
tosun: sbatch --account=mdbf --partition=mid_mdbf --qos=mid_mdbf -N4 ./hello.slurm                                                     
gandalf: mpicc / mpirun -n [p] --mca btl ^openib ./a.out //the mca part is for disabling the infiniband (if/when necessary)                              
*/

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print a hello message from each process
  printf("Hello world from processor %s, rank %d out of %d processors\n",
	 processor_name, world_rank, world_size);

    MPI_Barrier(MPI_COMM_WORLD);

  // Finalize the MPI environment
  MPI_Finalize();
  return 0;
}



