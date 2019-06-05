#include <omp.h>
#include <stdio.h>
#include <math.h>

//#define NUMT        4                //defined in shell script
//#define ARRAYSIZE   500000
#define NUMTRIES    1000

float A[ARRAYSIZE];
float B[ARRAYSIZE];
float C[ARRAYSIZE];

int main()
{
#ifndef _OPENMP
    fprintf(stderr, "OpenMP is not supported here -- sorry.\n");
    return 1;
#endif

omp_set_num_threads(NUMT);
    fprintf(stderr, "Using %d thread(s)\n", NUMT);

    double maxMegaMults = 0.;
    double sumMegaMults = 0.;
    double minTimeDiff = 999999;
    double sumTimeDiff = 0.;

    for (int t = 0; t < NUMTRIES; t++)
    {
        double time0 = omp_get_wtime();
        //printf("Time 0     = %lf seconds\n", time0);

        #pragma omp parallel for
        for (int i = 0; i < ARRAYSIZE; i++)
        {
            C[i] = A[i] * B[i];
        }

        double time1 = omp_get_wtime();
        //printf("Time 1     = %lf seconds\n", time1);
        double timeDiff = (time1-time0)*100000.;
        sumTimeDiff += timeDiff; 
        if (timeDiff < minTimeDiff)
        {
            minTimeDiff = timeDiff;
        }
        
        double megaMults = (double)ARRAYSIZE/(time1-time0)/1000000.;
        sumMegaMults += megaMults;
        if (megaMults > maxMegaMults)
        {
            maxMegaMults = megaMults;
        }
    }

    printf("Avg. performance = %8.2lf MegaMults/Sec\n", sumMegaMults/(double)NUMTRIES);
    printf("Peak performance = %8.2lf MegaMults/Sec\n", maxMegaMults);
    printf("Avg. time        = %8.2lf microseconds\n", sumTimeDiff/(double)NUMTRIES);
    printf("Min. time        = %8.2lf microseconds\n\n", minTimeDiff);

    return 0;
}
