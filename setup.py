#----------------------------------------------
# cythonを使うので,cythonをインストールしておく
# macOSでは、
# python setup.py build_ext -i
# で、カレントディレクトリにHMMのモジュールが作成される。
#----------------------------------------------
#
# 用いるデータは、スパイクが発生した時刻のデータ
# 使える関数はHMM.pyxに書いてある関数だけ
# 関数の効果：
# get_obs:指定したビン幅のカウント数のデータにする
# firstmodel:モデルパラメータの初期値を決める
# baumwelch:バウムウェルチアルゴリズムで最適なパラメーターを推定する
# viterbirate:ビタビアルゴリズムで最適な状態列を求めて、ビンのステップから
# 実際の時刻に変更する
# 詳しくはcHMM.c, HMM.pyxを見ればわかる。
#----------------------------------------------
from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

ext = Extension("HMM", sources=["source/HMM.pyx", "source/cHMM.c"], include_dirs=['.', get_include()])

setup(name="HMM", ext_modules=cythonize([ext]))
