/******************************************************************************
** Program name: Vectorized Array Multiplication and Reduction Using SSE
** Author: Rebecca L. Taylor
** Date: 13 May 2019
** Description: This header file declares the functions for squaring a number,
	generating random floats and ints, and seeding with the time of day. 
******************************************************************************/

#ifndef RAND_HPP
#define RAND_HPP

// square a float 
float SQR( float x);

// generate random numbers
float Ranf( float low, float high );
int Ranf( int ilow, int ihigh );

// seed time of day
void TimeOfDaySeed( );

#endif
