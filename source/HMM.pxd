cdef extern from "cHMM.h":
    ctypedef struct model_t:
        double *pi
        double **a
        double **means

    double **d2array(int row, int column)
    void d2free(double **array)
    int **i2array(int, int)
    void i2free(int **array)
    void cbaumwelch(int dimension, int total_state, int total_step, double *pi, double **a, double **means, int **obs)
    int *viterbi(int dimension, int total_state, int total_step, model_t model, int **obs)
    double **get_rate(int total_step, double dt, int *state)
