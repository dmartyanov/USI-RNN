## Modeling Event Sequences with Unevenly Spaced Intervals with RNN

This project is a demo example of how sequences with unevenly spaced 
intervals can be modeled for processing via Neural Network
 
Tensorflow with Keras API is used to train and evaluate models 

### Prerequisites

- tensorflow r1.4
- keras 2.0.4
- numpy 1.12.1

## Getting Started
### Installation
- Install tensorflow from https://github.com/tensorflow/tensorflow
- Clone this repo:
```bash
git clone https://github.com/dmartyanov/usi-rnn
cd usi-rnn
```
- Install requirements:
```bash
pip install -r requirements.txt
```

### Train

Add your dataset to `./data/`

P - padding length for sequences, default is 200

M - number of examples

A - alphabet size, default is 252


- `./data/X.npy` - (M, P) numpy array with events padded to PSL with 0 value, max value is less than A
- `./data/Y.npy` - (M, ) numpy array with binary target values
- `./data/L.npy` - (M, ) numpy array with the lengths of the sequences before padding
- `./data/D.npy` - (M, P) numpy array with time labels padded to PSL with 0 value, non-decreasing sequences
- `./data/tr_idx.npy` - (TRAIN_LENGTH, ) numpy array with train indices
- `./data/cv_idx.npy` - (VALIDATION_LENGTH, ) numpy array with validation indices

Run:
```bash
python main.py
```

## References
[Slides](https://www.slideshare.net/DmitryMartyanov/modeling-sequences-with-unevenly-spaced-intervals-via-rnn-97408883)



