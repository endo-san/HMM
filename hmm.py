import numpy as np
import glob
import matplotlib.pyplot as plt
import time
import HMM
# from joblib import Parallel, delayed


def hmm(spike_datas, m, dt, seed, spikestart, spikeend, plotFlag=False):
    """
    並列する時系列データを受け取って、隠れ状態の遷移を推定する。
    この場合の隠れ状態は、時系列の発生率（ポアソンモデルによる発生を仮定している）の状態空間の中のものである。
    例:複数のニューロンの時系列データから隠れ状態を推定する。
    引数：
        spike_datas：複数の並列する時刻データをlistにそれぞれ格納して、listでつないだもの。少し作るのは面倒だが、時刻データの数がそれぞれ違うので、わざわざlistに入れている。
            例：data = np.loadtxt -> data.tolist() -> spike_datas.append(data)みたいに作る
        m:状態数
        dt:時間ビン幅
        seed:初期値を決める際の乱数のタネ
        spikestart:スパイクが始まる時刻
        spikeend:スパイクが終わる
        plotFlag:隠れ状態の遷移をplotするかどうか。Trueなら書く。
    返り値：
        spike_hmm:予測した隠れ状態の遷移
        means:それぞれの状態の時系列データごとのrate
    """
    # t1 = time.time()
    total_step, spike_obs = HMM.get_obs(spikestart, spikeend, spike, dt)

    # 初期値
    pi, a, means = HMM.firstmodel(spike, seed, dt, m, total_step)
    # 計算
    pi, a, means = HMM.baumwelch(pi, a, means, spike_obs)
    # viterbiアルゴリズムの計算
    spike_hmm = HMM.viterbirate(dt, pi, a, means, spike_obs)

    # 図を描きたいならTrueにすればいい
    if plotFlag == True:
        plot_rate(m, dt, spike_hmm)

    # t2 = time.time()

    return spike_hmm, means


def plot_rate(m ,dt, rate):
    #--------------------
    # 隠れ状態の遷移の描画
    #--------------------
    #plt.style.use('ggplot')
    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.add_subplot(1,1,1, ylim=[-0.5, np.amax(rate[:, 1])+0.5])

    t = np.arange(0, len(rate))

    ax1.plot(rate[t, 0], rate[t, 1], "ro",markersize=1)
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("state")
    ax1.set_title('spike hidden state')

    title = "state "+str(m)+", dt = "+str(dt)+" sec"

    fig.suptitle(title)

    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    #plt.savefig("sample.eps", format="eps")
    #plt.close()
    plt.show()
