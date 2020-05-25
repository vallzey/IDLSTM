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

### RNN

```bash
sh vtrain_game_rnn_pos_500_1000.sh
```

### LSTM

```bash
sh vtrain_game_lstm_pos_500_1000.sh
```

### LSTM with time

```bash
sh vtrain_game_lstm_t_pos_500_1000.sh
```

### Time-LSTM

```bash
sh vtrain_game_tlstm2_pos_500_1000.sh
```

### IDLSTM

```bash
sh vtrain_game_dlstm_pos_500_1000.sh
```

### IDLSTM-EC

```bash
sh vtrain_game_dlstm_emcp_pos_500_1000.sh
```

## NOTE

If you want to use this code, please process the data into the same format as the example.

The data is placed in preprocess/data.
