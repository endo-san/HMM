import cython
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
# cimport HMM
import numpy as np
cimport numpy as cnp
import math



def get_obs(double onset, double offset, list spike_times, double dt):
    """
    引数：
    onset = スタート時刻
    offset = 終了時刻
    spike_set=初めのスパイク時刻を時刻0とした時のスパイク列
    dt=bin幅
    結果：
    観測列obs = (dimension, step)をえる
    """
    #--------------
    # 初めのスパイク時刻をonset = 0に設定することもできるが、
    # わざわざonsetを指定しているのは、訓練データとテストデータを扱うときに
    # データ間に隙間を作りたくないため
    #
    #--------------
    cdef:
        int d, step, dimension, total_step
        # int max_time, min_time
        double spike
        list spike_set
        cnp.ndarray obs

    dimension = len(spike_times)

    spike_set = [[spike - onset for spike in spike_times[d]] for d in range(dimension)]

    total_step = math.ceil((offset-onset)/dt)

    obs = np.zeros([total_step, dimension], dtype=int)

    for d in range(dimension):
        for x in spike_set[d]:
            step = math.floor(x/dt)
            obs[step, d] += 1

    return total_step, obs


def firstmodel(list spike_times, int seeed, double dt, int total_state, double total_step):
    """
    モデルの初期値を求める。
    ポアソン分布のパラメーターである平均は乱数を使って決める
    """
    cdef:
        int dimension
        double h
        cnp.ndarray pi, a, means, rate_mean

    dimension = len(spike_times)

    rate_mean = np.array([len(spike_times[d])/total_step for d in range(dimension)])
    # print("rate mean", rate_mean)
    np.random.seed(seeed)
    means = np.array([[h for x in range(total_state)] for h in rate_mean]) * np.random.rand(dimension, total_state) * 2

    pi = np.array([[1/<double>total_state for x in range(total_state)]])

    nonnormalize_a = (np.identity(total_state)+0.01)
    a = nonnormalize_a / np.sum(nonnormalize_a[0])

    return pi, a, means


@cython.boundscheck(False)
@cython.wraparound(False)
def baumwelch(cnp.ndarray pi, cnp.ndarray a, cnp.ndarray means, cnp.ndarray obs):
    cdef:
        int i, j
        int dimension, total_state, total_step
        double *p_pi
        double **p_a
        double **p_means
        int **p_obs

    dimension = len(obs[0])
    total_state = len(a)
    total_step = len(obs)

    p_pi = <double*>malloc(total_state*sizeof(double))
    p_a = d2array(total_state, total_state)
    p_means = d2array(dimension, total_state)

    for i in range(total_state):
        p_pi[i] = pi[0, i]
        for j in range(total_state):
            p_a[i][j] = a[i, j]
    for i in range(dimension):
        for j in range(total_state):
            p_means[i][j] = means[i, j]

    p_obs = i2array(total_step, dimension)
    # pinterではなく値の代入.pointer使いたい
    for i in range(total_step):
        for j in range(dimension):
            p_obs[i][j] = obs[i, j]

    cbaumwelch(dimension, total_state, total_step, p_pi, p_a, p_means, p_obs)

    for i in range(total_state):
        pi[0, i] = p_pi[i]
        for j in range(total_state):
            a[i, j] = p_a[i][j]
    for i in range(dimension):
        for j in range(total_state):
            means[i, j] = p_means[i][j]

    # メモリの解放
    if p_pi is not NULL:
        free(p_pi)
        p_pi = NULL
    if p_a is not NULL:
        d2free(p_a)
    if p_means is not NULL:
        d2free(p_means)
    if p_obs is not NULL:
        i2free(p_obs)

    return pi, a, means


@cython.boundscheck(False)
@cython.wraparound(False)
def viterbirate(double dt, cnp.ndarray pi, cnp.ndarray a, cnp.ndarray means, cnp.ndarray obs):
    """
    pi.shape = (1, total_state)に注意
    """
    cdef:
        int i, j
        int dimension, total_state, total_step
        double *p_pi
        double **p_a
        double **p_means
        int **p_obs
        int *p_state
        double **p_rate
        model_t model

    dimension = len(obs[0])
    total_state = len(a)
    total_step = len(obs)

    p_pi = <double*>malloc(total_state*sizeof(double))
    p_a = d2array(total_state, total_state)
    p_means = d2array(dimension, total_state)

    for i in range(total_state):
        p_pi[i] = pi[0, i]
        for j in range(total_state):
            p_a[i][j] = a[i, j]
    for i in range(dimension):
        for j in range(total_state):
            p_means[i][j] = means[i, j]

    p_obs = i2array(total_step, dimension)
    # pinterではなく値の代入.pointer使いたい
    for i in range(total_step):
        for j in range(dimension):
            p_obs[i][j] = obs[i, j]

    model.pi = p_pi
    model.a = p_a
    model.means = p_means

    p_state = viterbi(dimension, total_state, total_step, model, p_obs)

    p_rate = get_rate(total_step, dt, p_state)

    cdef cnp.ndarray rate = np.empty([total_step, 2])

    for i in range(total_step):
        for j in range(2):
            rate[i, j] = p_rate[i][j]

    # メモリの解放
    if p_pi is not NULL:
        free(p_pi)
        p_pi = NULL
    if p_a is not NULL:
        d2free(p_a)
    if p_means is not NULL:
        d2free(p_means)
    if p_obs is not NULL:
        i2free(p_obs)
    if p_state is not NULL:
        free(p_state)
        p_state = NULL
    if p_rate is not NULL:
        d2free(p_rate)

    return rate


class HMM_poisson():
    def __init__(self, list spike_times, int seeed, double dt, int total_state, double total_step):
        """
        モデルの初期値を求める。
        ポアソン分布のパラメーターである平均は乱数を使って決める
        """
        cdef:
            int dimension
            double h
            cnp.ndarray pi, a, means, rate_mean

        dimension = len(spike_times)
        rate_mean = np.array([len(spike_times[d])/total_step for d in range(dimension)])

        np.random.seed(seeed)
        means = np.array([[h for x in range(total_state)] for h in rate_mean]) * np.random.rand(dimension, total_state) * 2

        pi = np.array([[1/<double>total_state for x in range(total_state)]])

        nonnormalize_a = (np.identity(total_state)+0.01)
        a = nonnormalize_a / np.sum(nonnormalize_a[0])


        self.dt = dt
        self.pi = pi
        self.a = a
        self.means = means


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit(self, cnp.ndarray obs):
        """
        baumwelch
        obs.shape = (total_step, dimension)
        """
        cdef:
            int i, j
            int dimension, total_state, total_step
            double *p_pi
            double **p_a
            double **p_means
            int **p_obs

        dimension = len(obs[0])
        total_state = len(self.a)
        total_step = len(obs)

        p_pi = <double*>malloc(total_state*sizeof(double))
        p_a = d2array(total_state, total_state)
        p_means = d2array(dimension, total_state)

        for i in range(total_state):
            p_pi[i] = self.pi[0, i]
            for j in range(total_state):
                p_a[i][j] = self.a[i, j]
        for i in range(dimension):
            for j in range(total_state):
                p_means[i][j] = self.means[i, j]

        p_obs = i2array(total_step, dimension)
        # pinterではなく値の代入.pointer使いたい
        for i in range(total_step):
            for j in range(dimension):
                p_obs[i][j] = obs[i, j]

        cbaumwelch(dimension, total_state, total_step, p_pi, p_a, p_means, p_obs)

        for i in range(total_state):
            self.pi[0, i] = p_pi[i]
            for j in range(total_state):
                self.a[i, j] = p_a[i][j]
        for i in range(dimension):
            for j in range(total_state):
                self.means[i, j] = p_means[i][j]

        # メモリの解放
        if p_pi is not NULL:
            free(p_pi)
            p_pi = NULL
        if p_a is not NULL:
            d2free(p_a)
        if p_means is not NULL:
            d2free(p_means)
        if p_obs is not NULL:
            i2free(p_obs)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict(self, cnp.ndarray pi, cnp.ndarray obs):
        """
        pi.shape = (1, total_state)に注意
        """
        cdef:
            int i, j
            int dimension, total_state, total_step
            double *p_pi
            double **p_a
            double **p_means
            int **p_obs
            int *p_state
            double **p_rate
            model_t model

        dimension = len(obs[0])
        total_state = len(self.a)
        total_step = len(obs)

        p_pi = <double*>malloc(total_state*sizeof(double))
        p_a = d2array(total_state, total_state)
        p_means = d2array(dimension, total_state)

        for i in range(total_state):
            p_pi[i] = pi[0, i]
            for j in range(total_state):
                p_a[i][j] = self.a[i, j]
        for i in range(dimension):
            for j in range(total_state):
                p_means[i][j] = self.means[i, j]

        p_obs = i2array(total_step, dimension)
        # pinterではなく値の代入.pointer使いたい
        for i in range(total_step):
            for j in range(dimension):
                p_obs[i][j] = obs[i, j]

        model.pi = p_pi
        model.a = p_a
        model.means = p_means

        p_state = viterbi(dimension, total_state, total_step, model, p_obs)

        p_rate = get_rate(total_step, self.dt, p_state)

        cdef cnp.ndarray rate = np.empty([total_step, 2])

        for i in range(total_step):
            for j in range(2):
                rate[i, j] = p_rate[i][j]

        # メモリの解放
        if p_pi is not NULL:
            free(p_pi)
            p_pi = NULL
        if p_a is not NULL:
            d2free(p_a)
        if p_means is not NULL:
            d2free(p_means)
        if p_obs is not NULL:
            i2free(p_obs)
        if p_state is not NULL:
            free(p_state)
            p_state = NULL
        if p_rate is not NULL:
            d2free(p_rate)

        return rate
