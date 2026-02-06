#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void p2p_linear_bcast_double(double *buf, int count, int root, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int tag = 4242;

    if (size == 1) return;

    if (rank == root) {
        // Root sends the same payload to every other rank (O(P) messages).
        MPI_Request *reqs = (MPI_Request*)malloc((size_t)(size - 1) * sizeof(MPI_Request));
        if (!reqs) {
            fprintf(stderr, "Root: malloc failed\n");
            MPI_Abort(comm, 1);
        }

        int k = 0;
        for (int r = 0; r < size; r++) {
            if (r == root) continue;
            MPI_Isend(buf, count, MPI_DOUBLE, r, tag, comm, &reqs[k++]);
        }
        MPI_Waitall(size - 1, reqs, MPI_STATUSES_IGNORE);
        free(reqs);
    } else {
        // Non-root receives from root.
        MPI_Request req;
        MPI_Irecv(buf, count, MPI_DOUBLE, root, tag, comm, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

static void fill_pattern(double *buf, int n, double seed)
{
    for (int i = 0; i < n; i++) buf[i] = seed + 0.001 * (double)i;
}

static int check_pattern(const double *buf, int n, double seed)
{
    // O(1) quick check at a few indices
    int idxs[3] = {0, n/2, n-1};
    for (int t = 0; t < 3; t++) {
        int i = idxs[t];
        if (i < 0 || i >= n) continue;
        double expected = seed + 0.001 * (double)i;
        double err = fabs(buf[i] - expected);
        if (err > 1e-12) return 0;
    }
    return 1;
}

static void reduce_time_stats(double local, int size, MPI_Comm comm,
                              double *tmin, double *tavg, double *tmax)
{
    double sum = 0.0;
    MPI_Reduce(&local, tmin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local, tmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (tavg) *tavg = sum / (double)size;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n      = 1 << 23;   // doubles
    int iters  = 100;
    int warmup = 10;
    int root   = 0;

    // args: -n elements  -i iters  -w warmup  -r root
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-n") && a + 1 < argc) n = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-i") && a + 1 < argc) iters = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-w") && a + 1 < argc) warmup = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-r") && a + 1 < argc) root = atoi(argv[++a]);
        else {
            if (rank == 0) {
                fprintf(stderr, "Usage: %s [-n elements] [-i iters] [-w warmup] [-r root]\n", argv[0]);
            }
            MPI_Finalize();
            return 2;
        }
    }

    if (root < 0 || root >= size) {
        if (rank == 0) fprintf(stderr, "Invalid root %d for size %d\n", root, size);
        MPI_Finalize();
        return 2;
    }

    double *buf = (double*)malloc((size_t)n * sizeof(double));
    if (!buf) {
        fprintf(stderr, "Rank %d: malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- Sanity check P2P ---
    double seed = 7.0;
    if (rank == root) fill_pattern(buf, n, seed);
    else memset(buf, 0, (size_t)n * sizeof(double));

    p2p_linear_bcast_double(buf, n, root, MPI_COMM_WORLD);

    int ok = check_pattern(buf, n, seed);
    int all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!all_ok) {
        if (rank == 0) fprintf(stderr, "P2P broadcast sanity check FAILED.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- Sanity check MPI_Bcast ---
    seed = 9.0;
    if (rank == root) fill_pattern(buf, n, seed);
    else memset(buf, 0, (size_t)n * sizeof(double));

    MPI_Bcast(buf, n, MPI_DOUBLE, root, MPI_COMM_WORLD);

    ok = check_pattern(buf, n, seed);
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!all_ok) {
        if (rank == 0) fprintf(stderr, "MPI_Bcast sanity check FAILED.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ---------------------------
    // Warmup and timing: P2P
    // ---------------------------
    for (int w = 0; w < warmup; w++) {
        if (rank == root) buf[0] = (double)w;
        p2p_linear_bcast_double(buf, n, root, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int it = 0; it < iters; it++) {
        if (rank == root) buf[0] = (double)it;
        p2p_linear_bcast_double(buf, n, root, MPI_COMM_WORLD);
    }
    double t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    double p2p_time = (t1 - t0) / (double)iters;

    // ---------------------------
    // Warmup and timing: MPI_Bcast
    // ---------------------------
    for (int w = 0; w < warmup; w++) {
        if (rank == root) buf[0] = (double)w;
        MPI_Bcast(buf, n, MPI_DOUBLE, root, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int it = 0; it < iters; it++) {
        if (rank == root) buf[0] = (double)it;
        MPI_Bcast(buf, n, MPI_DOUBLE, root, MPI_COMM_WORLD);
    }
    t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    double mpi_time = (t1 - t0) / (double)iters;

    // Timing stats across ranks
    double p2p_min=0, p2p_avg=0, p2p_max=0;
    double mpi_min=0, mpi_avg=0, mpi_max=0;

    reduce_time_stats(p2p_time, size, MPI_COMM_WORLD, &p2p_min, &p2p_avg, &p2p_max);
    reduce_time_stats(mpi_time, size, MPI_COMM_WORLD, &mpi_min, &mpi_avg, &mpi_max);

    if (rank == 0) {
        double bytes = (double)n * (double)sizeof(double);
        printf("Ranks: %d  Root: %d\n", size, root);
        printf("Message: %d doubles (%.3f MiB)\n", n, bytes / (1024.0 * 1024.0));
        printf("Iters: %d  Warmup: %d\n\n", iters, warmup);

        printf("Time per broadcast (seconds):\n");
        printf("  P2P linear (root Isend to all): min %.6e  avg %.6e  max %.6e\n",
               p2p_min, p2p_avg, p2p_max);
        printf("  MPI_Bcast:                    min %.6e  avg %.6e  max %.6e\n",
               mpi_min, mpi_avg, mpi_max);

        if (mpi_avg > 0.0) {
            printf("\nSpeedup (avg): P2P / MPI_Bcast = %.3fx  (>1 means MPI_Bcast faster)\n",
                   p2p_avg / mpi_avg);
        }
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
