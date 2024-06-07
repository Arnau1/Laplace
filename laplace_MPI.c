#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

float stencil ( float v1, float v2, float v3, float v4 )
{
    return (v1 + v2 + v3 + v4) / 4;
}

// Adapted for parallelization
void laplace_step (float *in, float *out, int n, int m, float *previous, float *posterior, int rank, int size)
{
    int i, j, k, l;
    if (rank != 0) 
    {
        for (k=1; k < m-1; k++)
            out[k] = stencil(in[k+1], in[k-1], previous[k], in[m+k]);
    }

    for ( i=1; i < n-1; i++ )
        for ( j=1; j < m-1; j++ )
            out[i*m+j]= stencil(in[i*m+j+1], in[i*m+j-1], in[(i-1)*m+j], in[(i+1)*m+j]);
    
    if (rank != (size-1))
    {
      for(l=1; l < m-1; l++)
            out[(n-1)*m+l] = stencil(in[(n-1)*m+l+1], in[(n-1)*m+l-1], in[(n-2)*m+l], posterior[l]);
    }

}

// Adapted for parallelization
// float laplace_error ( float *old, float *new, int n, int m, int size, int rank )
// {
//     int i, j, skip0=0, skipn=0;
//     float error=0.0f;
//     if (rank == 0) {skip0 = 1;}
//     else if (rank == size-1) {skipn = 1;}
//     // don't skip the top and bottom edges for the central parts of the matrix
//     for ( i=skip0; i < n-skipn; i++ ){
//         for ( j=1; j < m-1; j++ ){
//             error = fmaxf( error, sqrtf( fabsf( old[i*m+j] - new[i*m+j] )));
//         }
//     }
//     return error;
// }

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
    // Initial time
    double t1 = MPI_Wtime();
    
    // INITIALIZE VARIABLES
    int n = 4096, m = 4096;
    const float pi  = 2.0f * asinf(1.0f);
    const float tol = 3.0e-3f;

    float error= 1.0f;
    float max_error = 0.0f;

    int i, j, iter_max=1000, iter=0;
    float *A, *Anew, *previous, *posterior;

    // get runtime arguments: n, m and iter_max
    if (argc>1) {  n        = atoi(argv[1]); }
    if (argc>2) {  m        = atoi(argv[2]); }
    if (argc>3) {  iter_max = atoi(argv[3]); }

    A    = (float*) malloc( n*m*sizeof(float) );
    Anew = (float*) malloc( n*m*sizeof(float) );
    previous = (float*) malloc(m*sizeof(float));
    posterior = (float*) malloc(m*sizeof(float));

    // INITIALIZE MPI (size = nproc)
    int size, rank;		
    MPI_Status s;   
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);	

    if (rank==0)
    {
        laplace_init (A, n, m);
        printf("Jacobi relaxation Calculation: %d rows x %d columns mesh,"
            " maximum of %d iterations\n", n, m, iter_max );    
    }

    MPI_Scatter(A, n*m/size, MPI_FLOAT, A, n*m/size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // MAIN LOOP: iterate until error <= tol a maximum of iter_max iterations
    while ( error > tol && iter < iter_max ) {
        // Send previous and posterior
        if (rank > 0) {MPI_Send(&A[0], m, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);}
        if (rank < (size-1)) {MPI_Send(&A[(n-1)*m], m, MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD);}
        if (rank < (size-1)) {MPI_Recv(posterior, m, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &s);}
        if (rank > 0) {MPI_Recv(previous, m, MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD, &s);}
        
        // Compute new values using main matrix and writing into auxiliary matrix
        laplace_step (A, Anew, n/size, m, previous, posterior, rank, size);

        // Compute error = maximum of the square root of the absolute differences
        error = 0.0f;
        // error = laplace_error (A, Anew, n/size, m, size, rank);
        error = laplace_error (A, Anew, n/size, m);

        // Copy from auxiliary matrix to main matrix
        laplace_copy (Anew, A, n/size, m);

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
    }

    free(A);
    free(Anew);  

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){           
        double t2 = MPI_Wtime();        
        printf("\nEXECUTION TIME: %fs\n", t2-t1);
    }
    MPI_Finalize();
    return 0;
}
