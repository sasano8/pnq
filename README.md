# pnq
PNQ is a Python implementation like Language Integrated Query (LINQ).

# Requirement

- Python 3.7+

# Installation

``` shell
pip install pnq
```

# Getting started


# Setup


# 調査
Linqライクなライブラリで、type hintと親和性があるライブラリを調査した。
結論として、type hintを最大に活用したlinqライクなライブラリはないようだ。

## pyfunctional
- https://github.com/EntilZha/PyFunctional
- star: 1.9k

機能は多い。
型情報は伝搬しない。

## rx
pythonのreactive extensionの実装。
Linqとは少々異なる。

## pyLINQ star: 2
type hintが効かない。機能は少なくstarも少ない

## pinq
type hintが効かない。機能は少なくstarも少ない

## linqish
プロジェクトはもう死んでいる

## PYNQ
名前がLINQっぽいがPythonでFPGAを利用するためのフレームワーク。
