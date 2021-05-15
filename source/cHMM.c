#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include"type.h"

/*構造体の定義*/
typedef struct model model_t;
typedef struct rate rate_t;

/*構造体の定義終わり*/

/*基本計算*/
double **d2array(int row, int column){
    /*
    matrix[i][j]で参照できる行列(ポインタ)をreturnする関数
    初期化はしていない。メモリを連続的に当てている。
    この操作ではなんどもポインタを使っているので注意。
    matrix[0]もポインタ。
    printf("%p\n",matrix[0]);とかで確認できる
    使い方：
        double **mat;
        mat = d2array(n ,m, l);
    */
    double **matrix;
    matrix = (double**) malloc(row*sizeof(double*));
    if(matrix == NULL){
        printf("d2arrayメモリが確保できません\n");
        exit(1);
    }
    matrix[0] = (double *) malloc(row * column * sizeof(double));
    if(matrix[0] == NULL){
        printf("d2array[0]メモリが確保できません\n");
        exit(1);
    }
    for(int i=0; i<row; i++){
        matrix[i] = matrix[0] + i * column;
    }
    return matrix;
}
/*２次元配列の解放*/
void d2free(double **array){
    free(array[0]);
    array[0] = NULL;
    free(array);
    array = NULL;
}
int **i2array(int row, int column){
    /*
    matrix[i][j]で参照できる行列(ポインタ)をreturnする関数
    初期化はしていない。メモリを連続的に当てている。
    この操作ではなんどもポインタを使っているので注意。
    matrix[0]もポインタ。
    printf("%p\n",matrix[0]);とかで確認できる
    使い方：
        double **mat;
        mat = d2array(n ,m, l);
    */
    int **matrix;
    matrix = (int**) malloc(row*sizeof(int*));
    if(matrix == NULL){
        printf("i2arrayメモリが確保できません\n");
        exit(1);
    }
    matrix[0] = (int *) malloc(row * column * sizeof(int));
    if(matrix[0] == NULL){
        printf("i2array[0]メモリが確保できません\n");
        exit(1);
    }
    for(int i=0; i<row; i++){
        matrix[i] = matrix[0] + i * column;
    }
    return matrix;
}
/*２次元配列の解放*/
void i2free(int **array){
    free(array[0]);
    array[0] = NULL;
    free(array);
    array = NULL;
}
/*３次元配列の確保*/
double ***d3array(int row, int column, int depth){
    double ***matrix = (double***)malloc(row*sizeof(double**));
    if(matrix == NULL){
        printf("d3arrayメモリが確保できません\n");
        exit(1);
    }
    matrix[0] = (double**)malloc(row*column*sizeof(double*));
    if(matrix[0] == NULL){
        printf("d3array[0]メモリが確保できません\n");
        exit(1);
    }
    matrix[0][0] = (double*)malloc(row*column*depth*sizeof(double));
    if(matrix[0][0] == NULL){
        printf("d3array[0][0]=%d %d %dメモリが確保できません\n", row, column, depth);
        exit(1);
    }
    for(int i=0; i<row; i++){
        matrix[i] = matrix[0] + i*column;
        for (int j=0; j<column; j++){
            matrix[i][j] = matrix[0][0] + i*(column*depth) + j*depth;
        }
    }
    return matrix;
}
/*3次元配列の解放*/
void d3free(double ***array){
    free(array[0][0]);
    array[0][0] = NULL;
    free(array[0]);
    array[0] = NULL;
    free(array);
    array = NULL;
}
/*factorialの計算*/
double factorial(int n){
    if (n==0){
        return 1;
    }else{
        return n * factorial(n-1);
    }
}
/*poisson分布計算*/
double poisson(double mean, int n){
    /*
    0の所には小さい値eを入れる。
    ビタビアルゴリズムでの -inf発散を避けるため。
    */
    double p;
    double dn = (double) n;
    if (n < 100){
        p = pow(mean, dn)*exp(-mean)/factorial(n);
    }else{
        p = pow((mean*exp(1)/dn), dn)*exp(-mean);
    }
    if (p == 0){
        p = 1e-300;
    }
    //p = pow((mean*exp(1)/dn), dn)*exp(-mean);
    return p;
}
/* 一次元配列の要素のlogを取ってから足しあげる。math.hが必要 */
double sum_log(int length, double *array){
    int i;
    double sum_log = 0.;

    for (i=0; i<length; i++){
        sum_log += log(array[i]);
    }
    return sum_log;
}
double dmax(int length, double *array){
    /*
    配列の最大値を求める。
    */
    double max;
    max = array[0];
    for (int i=1; i<length; i++){
        if(max < array[i]){
            max = array[i];
        }
    }
    return max;
}
int argmax(int length, double *array){
    /*
    配列の最大値のindexをえる。
    */
    int i, index = 0;
    double max = array[0];
    for (i=1; i<length; i++){
        if(max < array[i]){
            max = array[i];
            index = i;
        }
    }
    return index;
}
/*基本計算 end*/


/*baumwelch */
void get_b(int dimension, int total_state, int total_step, int **obs, double **restrict means, double ***restrict b){
    /*
    bを計算する。
    スパイク数がポアソン分布に従うとした時。
    スパイク数, ポアソン分布のλが与えられている時の、その場合を得る確率がb.
    */
    int d, m, step;
    for(step=0; step<total_step; step++){
        for(m=0; m<total_state; m++){
            for(d=0; d<dimension; d++){
                b[step][m][d] = poisson(means[d][m], obs[step][d]);
            }
        }
    }
}
void get_prod_b(int dimension, int total_state, int total_step, double ***restrict b, double **restrict prod_b){
    /*
    stepごとの、それぞれの状態を得る確率。dは並列している時系列のこと
    */
    int step, i, d;
    for (step=0; step<total_step; step++){
        for (i=0; i<total_state; i++){
            prod_b[step][i] = 1;
            for (d=0; d<dimension; d++){
                prod_b[step][i] = prod_b[step][i] * b[step][i][d];
            }
        }
    }
}

void get_alpha_beta(int dimension, int total_state, int total_step, double *restrict pi, double **restrict a, double ***restrict b, double **restrict alpha, double **restrict beta, double *restrict coefficient){
    /* alpha, beta, coefficient のポインタを書き換える*/
    int i, j, step;
    double sum[total_state];

    double **prod_b = d2array(total_step, total_state);
    get_prod_b(dimension, total_state, total_step, b, prod_b);

    /*forward algorithm*/
    /*initialization*/
    for (i=0; i<total_state; i++){
        alpha[0][i] = pi[i] * prod_b[0][i];
    }
    /*induction*/
    for (step=0; step<total_step-1; step++){
        coefficient[step] = 0;
        for (i=0; i<total_state; i++){
            coefficient[step] += alpha[step][i];
        }
        for (i=0; i<total_state; i++){
            alpha[step][i] = alpha[step][i]/coefficient[step];
        }
        for (j=0; j<total_state; j++){
            sum[j] = 0;
            for (i=0; i<total_state; i++){
                sum[j] += alpha[step][i] * a[i][j];
            }
            alpha[step+1][j] = sum[j] * prod_b[step+1][j];
        }
    }
    coefficient[total_step-1] = 0;
    for (i=0; i<total_state; i++){
        coefficient[total_step-1] += alpha[total_step-1][i];
    }
    for (i=0; i<total_state; i++){
        alpha[total_step-1][i] = alpha[total_step-1][i]/coefficient[total_step-1];
    }

    /*backward algorithm*/
    /*initialization*/
    for (i=0; i<total_state; i++){
        beta[total_step-1][i] = 1/coefficient[total_step-1];
    }
    /*induction*/
    for (step=total_step-1; step>0; step=step-1){
        for (i=0; i<total_state; i++){
            beta[step-1][i] = 0;
            for (j=0; j<total_state; j++){
                beta[step-1][i] += a[i][j] * prod_b[step][j] * beta[step][j];
            }
            beta[step-1][i] = beta[step-1][i]/coefficient[step-1];
        }
    }
    d2free(prod_b);
}


void get_newmodel(int dimension, int total_state, int total_step, double *restrict pi, double **restrict a, double **restrict means, double ***restrict b, int **restrict obs, double **restrict alpha, double **restrict beta, double *restrict coefficient){
    double **gamma = d2array(total_step, total_state);
    double ***xi = d3array(total_step-1, total_state, total_state);
    double sum;
    int i, j, d, step;

    /*get gamma*/
    for (step=0; step<total_step; step++){
        for (i=0; i<total_state; i++){
            gamma[step][i] = alpha[step][i] * beta[step][i] * coefficient[step];
        }
    }
    /*get xi*/
    double **prod_b = d2array(total_step, total_state);
    get_prod_b(dimension, total_state, total_step, b, prod_b);

    for (step=0; step<total_step-1; step++){
        for (i=0; i<total_state; i++){
            for (j=0; j<total_state; j++){
                xi[step][i][j] = alpha[step][i] * a[i][j] * prod_b[step+1][j] * beta[step+1][j];
            }
        }
    }
    /*update model parameter*/
    /* pi */
    for (i=0; i<total_state; i++){
        pi[i] = gamma[0][i];
        if (pi[i] == 0){
            pi[i] = 1e-300;
        }
    }
    /* a */
    double sum_xi_a[total_state];
    //double sum_xi[total_state][total_state];
    double **sum_xi = d2array(total_state, total_state);
    //初期化
    for (i=0; i<total_state; i++){
        sum_xi_a[i] = 0;
        for (j=0; j<total_state; j++){
            sum_xi[i][j] = 0;
        }
    }
    for (step=0; step<total_step-1; step++){
        for (i=0; i<total_state; i++){
            for (j=0; j<total_state; j++){
                sum_xi_a[i] += xi[step][i][j];
                sum_xi[i][j] += xi[step][i][j];
            }
        }
    }
    for (i=0; i<total_state; i++){
        for (j=0; j<total_state; j++){
            if (sum_xi_a[i] != 0){
                a[i][j] = sum_xi[i][j] / sum_xi_a[i];
            }else{
                a[i][j] = 1e-300;
            }
        }
    }
    /* means */
    double total_gamma[total_state];
    for (i=0; i<total_state; i++){
        total_gamma[i] = 0;
    }
    for (step=0; step<total_step; step++){
        for (i=0; i<total_state; i++){
            total_gamma[i] += gamma[step][i];
        }
    }
    for (d=0; d<dimension; d++){
        for (i=0; i<total_state; i++){
            if (total_gamma[i] != 0){
                sum = 0;
                for (step=0; step<total_step; step++){
                    sum += obs[step][d] * gamma[step][i];
                }
                means[d][i] = sum / total_gamma[i];
            }else{
                means[d][i] = 1e-300;
            }
        }
    }
    /*free memory that is unused after*/
    d2free(sum_xi);
    d2free(gamma);
    d2free(prod_b);
    d3free(xi);

}
/*
バウムウェルチアルゴリズムを計算する
引数：
    dimension = int 並列するスパイクの数
    total_state = int 隠れ状態の数
    total_step = int 全ステップ数
    pi = double型の動的配列　初期状態の確率 (total_state)
    a = double型の動的配列の動的配列　状態遷移確率 (total_state * total_state)
    means = double型の動的配列の動的配列 ポアソン分布の平均、出力するスパイク個数の平均 (dimension * total_state)
    obs = 2次元の行列　観測されたスパイクの個数　(total_step, dimension)
出力：
    なし。
    void型でpi, a, meansのポインタ値は変えず、数値を書き換える。
*/
void cbaumwelch(int dimension, int total_state, int total_step, double *restrict pi, double **restrict a, double **restrict means, int **restrict obs){

    int loop = 0;
    double old_P_log, new_P_log;
    double ***b = d3array(total_step, total_state, dimension);
    if (b == NULL){
        printf("メモリが確保できません\n");
        exit(1);
    }

    get_b(dimension, total_state, total_step, obs ,means, b);

    double **alpha=d2array(total_step, total_state);
    double **beta=d2array(total_step, total_state);
    double *coefficient=(double*)malloc(total_step*sizeof(double));

    /* 初めのモデルの更新 */
    get_alpha_beta(dimension, total_state, total_step, pi, a, b, alpha, beta, coefficient);

    get_newmodel(dimension, total_state, total_step, pi, a, means, b, obs, alpha, beta, coefficient);
    // モデルが更新された

    /* 初めのモデルの対数尤度計算 */
    new_P_log = sum_log(total_step, coefficient);

    do {
        loop += 1;
        /* 前のモデルの尤度を古い尤度にする */
        old_P_log = new_P_log;

        get_b(dimension, total_state, total_step, obs, means, b);
        get_alpha_beta(dimension, total_state, total_step, pi, a, b, alpha, beta, coefficient);
        /* 前のモデルから更新されたモデルの尤度計算 */
        new_P_log = sum_log(total_step, coefficient);

        /* 更新されたモデルを使ってパラメータの更新 */
        get_newmodel(dimension, total_state, total_step, pi, a, means, b, obs, alpha, beta, coefficient);

        //printf("loop = %d,\tProb = %e\n", loop, new_P_log);

        if (loop>5000){
            printf("Now, loop is over the given number 5000\n");
            break;
        }

    } while (fabs(new_P_log - old_P_log) > 1e-5);
    /*
    前のモデルの尤度と更新されたモデルの尤度比較
    old 前のモデル
    new 前のモデルから更新されたモデル
    ここでは、最新のモデルの評価はしてないことに注意
    */


    printf("state %d, loop = %d\n", total_state, loop);

    /*使わないメモリの解放*/
    d2free(alpha);
    d2free(beta);
    free(coefficient);
    coefficient = NULL;
    d3free(b);
}
/* void baumwelch end*/


/* viterbi */
/*
引数：
    dimension = int 並列するスパイクの数
    total_state = int 隠れ状態の数
    total_step = int 全ステップ数
    model = model_t型　pi, a, meansを要素にもつ
    obs = 2次元の行列　観測されたスパイクの個数　(dimension, total_step)
出力：
    ビタビアルゴリズムで計算した状態列 state (1次元の動的配列)
*/
int *viterbi(int dimension, int total_state, int total_step, model_t model, int **obs){

    int i, j, step, d;
    /* 状態列を格納する動的配列　int型*/
    int *state = (int *)malloc(total_step*sizeof(int));
    if(state == NULL){
        printf("state取れません\n");
        exit(1);
    }
    /*
    ビタビアルゴリズムのスケーリングしたものを計算するときの変数
    deltaは対数尤度。psiは状態番号
    */
    double **delta = d2array(total_step, total_state);
    int **psi = i2array(total_step, total_state);

    double ***b = d3array(total_step, total_state, dimension);
    get_b(dimension, total_state, total_step, obs, model.means, b);

    /* log(np.prod(b))の計算 */
    double **log_prod_b = d2array(total_step, total_state);
    for (step=0; step<total_step; step++){
        for (i=0; i<total_state; i++){
            log_prod_b[step][i] = 0;
            for (d=0; d<dimension; d++){
                log_prod_b[step][i] += log(b[step][i][d]);
            }
        }
    }
    double *log_prod = (double*)malloc(total_state*sizeof(double));

    //initialization
    for (i=0; i<total_state; i++){
        delta[0][i] = log(model.pi[i]) + log_prod_b[0][i];
        psi[0][i] = 0;
        // printf("%lf, %lf, %lf\n", model.pi[i], log_prod_b[i][0], delta[i][0]);
    }
    //recursion
    for (step=0; step<total_step-1; step++){
        for (j=0; j<total_state; j++){
            for (i=0; i<total_state; i++){
                log_prod[i] = delta[step][i] + log(model.a[i][j]);
            }
            delta[step+1][j] = dmax(total_state, log_prod) + log_prod_b[step+1][j];
            psi[step+1][j] = argmax(total_state, log_prod);
        }
    }

    //termination
    double dammy[total_state];
    for (i=0; i<total_state; i++){
        dammy[i] = delta[total_step-1][i];
    }

    state[total_step-1] = argmax(total_state, dammy);

    //backtracking
    for (step=total_step-1; step>0; step=step-1){
        state[step-1] = psi[step][state[step]];
    }

    /*メモリ解放*/
    free(log_prod);
    d3free(b);
    d2free(delta);
    i2free(psi);

    return state;
}
/* viterbi end */


/* transrate state to state of real time */
double **get_rate(int total_step, double dt, int *state){
    int step;
    double **rate = d2array(total_step, 2);
    double norm = 1/dt;
    for (step=0; step<total_step; step++){
        rate[step][0] = step/norm;
        rate[step][1] = state[step];
    }
    return rate;
}
