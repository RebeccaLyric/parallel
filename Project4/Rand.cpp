/******************************************************************************
** Program name: Vectorized Array Multiplication and Reduction Using SSE
** Author: Rebecca L. Taylor
** Date: 13 May 2019
** Description: This execution file defines the functions for squaring a number,
	generating random floats and ints, and seeding with the time of day. 
******************************************************************************/

# include <stdlib.h>
# include <time.h>
# include <math.h>

// square a number
float SQR( float x) 
{
    return x * x;
}

// generate random numbers
float Ranf( float low, float high )
{
        float r = (float) rand();               // 0 - RAND_MAX
        float t = r  /  (float) RAND_MAX;       // 0. - 1.

        return   low  +  t * ( high - low );
}

int Ranf( int ilow, int ihigh )
{
        float low = (float)ilow;
        float high = ceil( (float)ihigh );

        return (int) Ranf(low,high);
}

void TimeOfDaySeed( )
{
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time_t  timer;
	time( &timer );
	double seconds = difftime( timer, mktime(&y2k) );
	unsigned int seed = (unsigned int)( 1000.*seconds );    // milliseconds
	srand( seed );
}
