#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

float stencil ( float v1, float v2, float v3, float v4 )
{
    return (v1 + v2 + v3 + v4) / 4;
}

void laplace_step ( float *in, float *out, int n, int m)
{
    int i, j;
    for ( i=1; i < n-1; i++ )
        for ( j=1; j < m-1; j++ )
        out[i*m+j]= stencil(in[i*m+j+1], in[i*m+j-1], in[(i-1)*m+j], in[(i+1)*m+j]);
}

float laplace_error ( float *old, float *new, int n, int m )
{
    int i, j;
    float error=0.0f;
    for ( i=1; i < n-1; i++ )
        for ( j=1; j < m-1; j++ )
        error = fmaxf( error, sqrtf( fabsf( old[i*m+j] - new[i*m+j] )));
    return error;
}

void laplace_copy ( float *in, float *out, int n, int m )
{
    int i, j;
    for ( i=1; i < n-1; i++ )
        for ( j=1; j < m-1; j++ )
        out[i*m+j]= in[i*m+j];
}


void laplace_init ( float *in, int n, int m )
{
    int i, j;
    const float pi  = 2.0f * asinf(1.0f);
    memset(in, 0, n*m*sizeof(float));
    for (j=0; j<m; j++)  in[    j    ] = 0.f;
    for (j=0; j<m; j++)  in[(n-1)*m+j] = 0.f;
    for (i=0; i<n; i++)  in[   i*m   ] = sinf(pi*i / (n-1));
    for (i=0; i<n; i++)  in[ i*m+m-1 ] = sinf(pi*i / (n-1))*expf(-pi);
}

int main(int argc, char** argv)
{
    // Init time
    double t1 = MPI_Wtime();

    // Init variables
    int n = 4096, m = 4096;
    const float pi  = 2.0f * asinf(1.0f);
    const float tol = 3.0e-3f;

    float error= 1.0f;

    int i, j, iter_max=100, iter=0;
    float *A, *Anew, *Aext, *Anewext, *row0, *rown, *prev, *post; 

    // get runtime arguments: n, m and iter_max
    if (argc>1) {  n        = atoi(argv[1]); }
    if (argc>2) {  m        = atoi(argv[2]); }
    if (argc>3) {  iter_max = atoi(argv[3]); }

    // INITIALIZE MPI (size = nproc)
    int nproc, rank, task;		
    MPI_Status s;    
    MPI_Request request;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Number of rows per process and of n distributed
    int split = n / nproc;
    int extension;
    if (rank == 0 || rank == nproc-1) {extension = 1;}
    else {extension = 2;}

    // Declare arrays
    A    = (float*) malloc(n * m * sizeof(float) );
    Anew = (float*) malloc(n * m * sizeof(float) );
    Aext = (float*) malloc(n * (split+extension) * sizeof(float));
    Anewext = (float*) malloc(n * (split+extension) * sizeof(float));
    row0 = (float*) malloc(m * sizeof(float));
    rown = (float*) malloc(m * sizeof(float));
    prev = (float*) malloc(m * sizeof(float));
    post = (float*) malloc(m * sizeof(float));
    
    // set boundary conditions
    if (rank == 0) {
        laplace_init (A, n, m);
        printf("\nJacobi relaxation Calculation: %d rows x %d columns mesh,"
            " maximum of %d iterations\n",
            n, m, iter_max );
    }

        // Main loop: iterate until error <= tol a maximum of iter_max iterations
    while ( error > tol && iter < iter_max ) {
        // Scatter A
        MPI_Scatter(A, split*m, MPI_FLOAT, A, split*m, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Send and recieve prev and post
        memcpy(row0, A, m*sizeof(float));
        memcpy(rown, A+(m*(split-1)), m*sizeof(float));
        // If not 0
        if (rank > 0) {
            MPI_Send(row0, m, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
            if (rank+1 < nproc) {
                MPI_Irecv(post, m, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &request);
                }            
        }
        // if not last
        if (rank < (nproc-1)) {
            MPI_Send(rown, m, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
            if (rank-1 > 0) {
                MPI_Irecv(prev, m, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &request);
                } 
        }
        MPI_Wait(&request, &s);

        // Extend A with prev and post
        if (rank == 0) {
            memcpy(Aext, A, split*m*sizeof(float));
            memcpy(Aext+(split*m), post, m*sizeof(float));
        } else if (rank == nproc-1) {
            memcpy(Aext, prev, m*sizeof(float));
            memcpy(Aext+m, A, split*m*sizeof(float));
        } else {
            memcpy(Aext, prev, m*sizeof(float));
            memcpy(Aext+m, A, split*m*sizeof(float));
            memcpy(Aext+(split*m), post, m*sizeof(float));
        }

        // Create Anewext as a copy of Aext
        memcpy(Anewext, Aext, (split+extension)*m*sizeof(float));

        // Compute new values using main matrix and writing into auxiliary matrix (with Aext and Anewext)
        laplace_step (Aext, Anewext, split+extension, m);

        // Create Anew by removing the extensions from Anewext
        if (rank == 0) {memcpy(Anew, Anewext, split*m*sizeof(float));}
        else           {memcpy(Anew, Anewext+m, split*m*sizeof(float));}

        // Gather Anew to 0
        MPI_Gather(Anew, split*m, MPI_FLOAT, Anew, split*m, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        if (rank == 0){
            // Compute error = maximum of the square root of the absolute differences (with Aext and Anewext)
            error = 0.0f;
            error = laplace_error (A, Anew, n, m);

            // Copy from auxiliary matrix to main matrix 
            laplace_copy (Anew, A, n, m);
            
            // if number of iterations is multiple of 10 then print error on the screen
            iter++;
            if (iter % (iter_max/10) == 0)
            printf("%5d, %0.6f\n", iter, error);
        }
        MPI_Bcast(&error, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        printf("Rank %d, error %f", rank, error);
    } // while

    free(A); free(Anew); free(Aext); free(Anewext);
    free(row0); free(rown); free(prev); free(post);

    if (rank == 0){
        double t2 = MPI_Wtime();
        printf("Finished calculation! Time: %fs\n", t2-t1);
    }
}
