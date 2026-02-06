/*
MPI_Comm_split:
- Splits the global communicator (MPI_COMM_WORLD) into sub-communicators based on the color value.
- Processes with the same color are grouped together.
- Each subgroup is assigned its own communicator (group_comm).
- The key argument ensures a consistent ordering of ranks within the new communicator.

Process Grouping:
- Processes are divided into two groups based on whether their rank is even (color=0) or odd (color=1).

Barrier Synchronization:
- MPI_Barrier(group_comm) ensures that all processes within each group reach the barrier before proceeding.

The barrier is only applied to processes in the same group.
- MPI_Comm_free:

Frees the newly created communicator to avoid resource leaks.
*/

#include <mpi.h>
#include <stdio.h>
#include<unistd.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_rank, world_size;

  // Get the rank and size in the global communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  printf("Before global barrier: %d/%d\n", world_rank, world_size);
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(2); //make the threads spent some time
  printf("After first global barrier: %d/%d\n", world_rank, world_size);
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(2); //make the threads spent some time
  printf("After second global barrier: %d/%d\n", world_rank, world_size);

  // Divide processes into two groups: Even and Odd ranks
  int color = world_rank % 2; // Define group color: 0 for even, 1 for odd
  MPI_Comm group_comm;

  // Split the communicator based on the color
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &group_comm);

  // Get rank and size in the new group communicator
  int group_rank, group_size;
  MPI_Comm_rank(group_comm, &group_rank);
  MPI_Comm_size(group_comm, &group_size);

  // Each process prints its group information
  printf("Global Rank: %d, Group: %d, Group Rank: %d, Group Size: %d\n",
	 world_rank, color, group_rank, group_size);

  // Barrier synchronization within each group
  printf("Global Rank: %d is waiting at the barrier in Group %d\n", world_rank, color);
  MPI_Barrier(group_comm);
  printf("Global Rank: %d passed the barrier in Group %d\n", world_rank, color);

  // Free the group communicator
  MPI_Comm_free(&group_comm);

  MPI_Finalize();
  return 0;
}

