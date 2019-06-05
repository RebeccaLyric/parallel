/******************************************************************************
** Program name: Functional Decomposition: Graindeer Simulation
** Author: Rebecca L. Taylor
** Date: 6 May 2019
** Description: This header file includes the function declarations for
	squaring a number, generating a random float, and generating a random int.
******************************************************************************/

#ifndef RAND_HPP
#define RAND_HPP

// square a float 
float SQR( float x);

// generate random numbers
float Ranf( unsigned int *seedp,  float low, float high );
int Ranf( unsigned int *seedp, int ilow, int ihigh );

#endif
