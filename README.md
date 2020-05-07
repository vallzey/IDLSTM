# IDLSTM

## Introduction

This project is the implementation of the paper "Time-aware Sequence Model for Next Item Recommendation".

## Requirments

The code is tested in the following envirenment.  
theano=0.9.0  
lasagne=0.2.dev1  
pandas=0.18.1  
cudnn=5.1  
cuda=8.0  

```bash
pip install -v theano==0.9.0
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
pip install -v pandas==0.18.1
```
## Train and Test
### IDLSTM

```bash
sh vtrain_game_dlstm_pos_500_1000.sh
```
### IDLSTM-EC

```bash
sh vtrain_game_dlstm_emcp_pos_500_1000.sh
```
