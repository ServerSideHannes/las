# Listen, attend and spell
Minimal tf 2.0 implementation of Listen, attend and spell (https://arxiv.org/abs/1508.01211)

#### Done:
+ Model architecture looks right to me. If you find an error in the code please dont hesitate to open an issue 😊

#### ToDo:
+ Implement data handing for easier training of model.
+ Train on LibriSpeech 100h

#### Usage
The file model.py contains the architecture of the model. Example usage below.

```python
"""
def LAS(dim, f_1, no_tokens):
  dim: Number of hidden neurons for most LSTM's.
  No_tokens: Number of unique tokens for input and output vector.
  f_1: pBLSTM takes (Batch, timesteps, f_1) as input, f_1 is number of features of the mel spectrogram per timestep. 
  Timestep is the width of the spectrogram.
"""

model = LAS(256, 512, 16)
model.compile(loss="mse", optimizer="adam")
```
Example of model.predict()
```python
...
model = LAS(256, 512, 16)
model.compile(loss="mse", optimizer="adam")

# x_1 should have shape (Batch-size, timesteps, f_1)
x_1 = mel_specdata
# x_2 should have shape (Batch-size, no_prev_tokens, No_tokens). 
x_2 = one_hot_encoded_vector

model.predict([x_1, x_2])
```
