#ifndef _INCLUDE_HMM_
#define _INCLUDE_HMM_

/*#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>*/
#include"type.h"


typedef struct model model_t;

double **d2array(int row, int column);
void d2free(double **array);
int **i2array(int row, int column);
void i2free(int **array);
void cbaumwelch(int dimension, int total_state, int total_step, double *pi, double **a, double **means, int **obs);
int *viterbi(int dimension, int total_state, int total_step, model_t model, int **obs);
double **get_rate(int total_step, double dt, int *state);


#endif
