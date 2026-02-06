#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) fprintf(stderr, "Run with exactly 2 MPI ranks.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int peer = 1 - rank;

    int send_val = (rank == 0) ? 123 : 456;
    int recv_val = -1;

    MPI_Request reqs[2];
    MPI_Status  stats[2];

    /*
     * (1) Why post the receive first?
     *
     * With nonblocking operations, *both* calls return immediately, but the
     * underlying MPI library still needs a matching receive posted somewhere
     * to complete a send (especially for larger messages).
     *
     * In short: "post receives early, then sends" is a common MPI best practice.
     */

    // Nonblocking receive: will accept 1 int from 'peer' with tag 0
    MPI_Irecv(&recv_val, 1, MPI_INT,
              /*source=*/peer, /*tag=*/0, MPI_COMM_WORLD,
              &reqs[0]);

    /*
     * Nonblocking send: sends 1 int to 'peer' with tag 0.
     *
     * IMPORTANT: 'send_val' must remain unchanged and in scope until the send
     * request completes (MPI_Wait/MPI_Test on reqs[1]). In this example it is
     * a stack variable that stays valid until after MPI_Waitall.
     */
    MPI_Isend(&send_val, 1, MPI_INT,
              /*dest=*/peer, /*tag=*/0, MPI_COMM_WORLD,
              &reqs[1]);

    /*
     * At this point, both operations may still be in-flight.
     * You could do useful computation here to overlap with communication.
     */

    // Wait for both nonblocking operations to complete and collect their statuses
    MPI_Waitall(2, reqs, stats);

    /*
     * (2) Using MPI_Status (the 'stats' array)
     *
     * MPI_Status tells you what actually happened for the completed operation:
     *   - stats[i].MPI_SOURCE : actual sender rank (useful if source = MPI_ANY_SOURCE)
     *   - stats[i].MPI_TAG    : actual tag (useful if tag = MPI_ANY_TAG)
     *   - error code (accessible via MPI_Error_string or stats[i].MPI_ERROR in some MPIs)
     *
     * Additionally, with receives you can query how many elements arrived using
     * MPI_Get_count(). This is especially useful with variable-size messages.
     *
     * In this example:
     *   - stats[0] corresponds to the Irecv (reqs[0])
     *   - stats[1] corresponds to the Isend (reqs[1])
     */

    int recv_count = -1;
    MPI_Get_count(&stats[0], MPI_INT, &recv_count);

    printf("Rank %d:\n", rank);
    printf("  Sent     %d to rank %d (send status: source=%d tag=%d)\n",
           send_val, peer, stats[1].MPI_SOURCE, stats[1].MPI_TAG);
    printf("  Received %d from rank %d (recv status: source=%d tag=%d, count=%d)\n",
           recv_val, peer, stats[0].MPI_SOURCE, stats[0].MPI_TAG, recv_count);

    MPI_Finalize();
    return 0;
}
