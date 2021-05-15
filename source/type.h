#ifndef _INCLUDE_TYPE_
#define _INCLUDE_TYPE_
/*構造体の定義*/

struct model {
    double *pi;
    double **a;
    double **means;
};

struct rate {
    double realtime;
    int state;
};

/*構造体の定義終わり*/

#endif
