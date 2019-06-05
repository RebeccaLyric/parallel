/******************************************************************************
** Program name: Functional Decomposition: Graindeer Simulation
** Author: Rebecca L. Taylor
** Date: 6 May 2019
** Description: This execution file includes the function definitions for
	squaring a number, generating a random float, and generating a random int.
******************************************************************************/

# include <stdlib.h>

// square a number
float SQR( float x) 
{
    return x * x;
}

// generate random numbers
float Ranf( unsigned int *seedp,  float low, float high )
{
        float r = (float) rand_r( seedp );              // 0 - RAND_MAX

        return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}

int Ranf( unsigned int *seedp, int ilow, int ihigh )
{
        float low =  (float)ilow;
        float high = (float)ihigh + 0.9999f;

        return (int)(  Ranf(seedp, low,high) );
}
