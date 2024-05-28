#include <stdio.h>
#include <math.h>

#include <string.h>

#define n 4096 
#define m 4096 


float A[n][m], Anew[n][m], y[n]; 

int main(int argc, char** argv) 
{ 
  const float pi  = 2.0f * asinf(1.0f); 
  const float tol = 3.0e-3f; 

  float error= 1.0f;; 

  int i, j, iter_max=100, iter=0; 

  // get iter_max from command line at execution time 
  if (argc>1) {  iter_max = atoi(argv[1]); } 

  // set all values in matrix as zero  
  memset(A, 0, n * m * sizeof(float)); 

  // set boundary conditions 
  for (j=0; j < m; j++) 
  { 
    A[0][j]   = 0.f;  
    A[n-1][j] = 0.f; 
  } 

  for (i=0; i < n; i++) 
  { 
    y[i] = sinf(pi * i / (n-1)); 
    A[i][0] = y[i]; 
    A[i][m-1] = y[i]*expf(-pi); 
  } 

  //int iter = 0; 
  
  // Main loop: iterate until error <= tol a maximum of iter_max iterations 

  while ( error > tol && iter < iter_max ) { 
    // Compute new values using main matrix and writing into auxiliary matrix 
    for( j=1; j < m-1; j++) 
      for( i=1; i < n-1; i++ ) 
         Anew[i][j]= (A[i][j+1]+A[i][j-1]+A[i-1][j]+A[i+1][j])/4; 

    // Compute error = maximum of the square root of the absolute differences 
    error = 0.0f; 

    for( j=1; j < m-1; j++) 
      for( i=1; i < n-1; i++ ) 
         error = fmaxf( error, sqrtf(fabsf(Anew[i][j]-A[i][j]))); 


    // Copy from auxiliary matrix to main matrix 
    for( j=1; j < m-1; j++) 
      for( i=1; i < n-1; i++ ) 
         A[i][j] = Anew[i][j]; 


    // if number of iterations is multiple of 10 then print error on the screen    
    iter++; 
    if (iter % (iter_max/10) == 0) 
       printf("%5d, %0.6f\n", iter, error);  
  } // while 

} 