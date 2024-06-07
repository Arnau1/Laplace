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
    int n = 8, m = 8;
    const float pi  = 2.0f * asinf(1.0f);
    const float tol = 3.0e-3f;

    float error= 1.0f;
    float max_error;

    int i, j, iter_max=1, iter=0;
    float *A, *Anew, *Aext, *Anewext, *prev, *post; 

    // get runtime arguments: n, m and iter_max
    if (argc>1) {  n        = atoi(argv[1]); }
    if (argc>2) {  m        = atoi(argv[2]); }
    if (argc>3) {  iter_max = atoi(argv[3]); }

    // INITIALIZE MPI
    int nproc, rank, task;		
    MPI_Status s;    
    MPI_Request request;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Number of rows per process and the extension
    int split = n / nproc;
    int extension;
    if (rank == 0 || rank == nproc-1) {extension = 1;}
    else {extension = 2;}

    // Declare arrays
    A    = (float*) malloc(n * m * sizeof(float) ); // 4096 * 4096
    Anew = (float*) malloc(n * m * sizeof(float) ); // 4096 * 4096
    Aext = (float*) malloc((split+extension) * m * sizeof(float)); // 1026 * 4096 (o 1025 * 4096)
    Anewext = (float*) malloc((split+extension) * m * sizeof(float)); // 1026 * 4096 (o 1025 * 4096)
    prev = (float*) malloc(m * sizeof(float)); // 4096
    post = (float*) malloc(m * sizeof(float)); // 4096
    
    // set boundary conditions
    if (rank == 0) {
        laplace_init (A, n, m);
        printf("\nJacobi relaxation Calculation: %d rows x %d columns mesh,"
            " maximum of %d iterations\n",
            n, m, iter_max );
    }

    // Scatter A
    MPI_Scatter(A, split*m, MPI_FLOAT, A, split*m, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Main loop: iterate until error <= tol a maximum of iter_max iterations
    while ( error > tol && iter < iter_max ) {
        if (rank == 2) {
            printf("Printing A at rank 2\n");
            for (int i=0; i<split; i++){
                for (int j=0; j<m; j++){
                    printf(" %f ", A[i*m + j]);
                }
                printf("\n");
            }
        }

        // Send and recieve prev and post
        if (rank > 0) {MPI_Send(&A[0], m, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);}
        if (rank < (nproc-1)) {MPI_Send(&A[(n-1)*m], m, MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD);}
        if (rank < (nproc-1)) {MPI_Recv(post, m, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &s);}
        if (rank > 0) {MPI_Recv(prev, m, MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD, &s);}

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

        if (rank == 2) {
            printf("Printing Aext at rank 2\n");
            for (int i=0; i<split+extension; i++){
                for (int j=0; j<m; j++){
                    printf(" %f ", Aext[i*m + j]);
                }
                printf("\n");
            }
        }

        // Compute new values using main matrix and writing into auxiliary matrix (with Aext and Anewext)
        laplace_step (Aext, Anewext, split+extension, m);
        
        // Compute error = maximum of the square root of the absolute differences (with Aext and Anewext)
        error = 0.0f;
        error = laplace_error (Aext, Anewext, n, m);

        // Create Anew by removing the extensions from Anewext
        if (rank == 0) {memcpy(Anew, Anewext, split*m*sizeof(float));}
        else           {memcpy(Anew, Anewext+m, split*m*sizeof(float));}

        // Copy from auxiliary matrix to main matrix 
        laplace_copy (Anew, A, n, m);

        if (rank == 2) {
            printf("Printing A at rank 2 later\n");
            for (int i=0; i<split; i++){
                for (int j=0; j<m; j++){
                    printf(" %f ", A[i*m + j]);
                }
                printf("\n");
            }
        }

        // Collect error
        MPI_Reduce(&error, &max_error, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Bcast(&max_error, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        error = max_error;
        
        iter++;
        if (rank == 0){
            // if number of iterations is multiple of 10 then print error on the screen
            if (iter % (iter_max/10) == 0)
                printf("Process: %d, Iteration: %d, Error: %0.6f\n", rank, iter, error);
        }
        else {
            // if number of iterations is multiple of 10 then print error on the screen
            if (iter % (iter_max/10) == 0)
                printf("                                  %0.6f\n",error);
        }
    } // while


    free(A); free(Anew); free(Aext); free(Anewext); free(prev); free(post);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0){
        double t2 = MPI_Wtime();
        printf("Finished calculation! Time: %fs\n", t2-t1);
    }
    
    MPI_Finalize();
    return 0;
}
