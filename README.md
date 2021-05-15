# HMM
This tools to analyze neural spike data with HMM in python, cython and C
## prepare
```
python setup.py build_ext --inplace
```
とすれば、そのディレクトリに、pythonのライブラリができる。
その後、
```
import HMM
hmm = HMM_poisson(spike, seed, dt, m, total_step)
```
で、使うことができる。

## How to use
# スパイク時刻を格納した時系列データを観測ベクトルに変換
spikestart: スパイクの発生時刻

spikeend: スパイクの終了時刻

spike: 実際のスパイク時刻を格納したデータ

dt: time window

```
total_step, spike_obs = HMM.get_obs(spikestart, spikeend, spike, dt)
```
# 学習
```
hmm = HMM_poisson(spike, seed, dt, m, total_step)
hmm.fit(spike, obs)
```
# viterbiアルゴリズムの計算
ステップごとの隠れ状態の変化を出力。
```
spike_hmm = hmm.predict(hmm.pi, spike_obs)
```
実時間に直したいときは、
```
spike_hmm = spike_hmm[0, :] * dt
```

# Reference
L.R.Rabiner, 1989, Proceedings of the IEEE, "A tutorial on hidden Markov models and selected applications in speech recognition"(https://ieeexplore.ieee.org/abstract/document/18626)
