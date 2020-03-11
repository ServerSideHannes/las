# Listen, attend and spell
Minimal tf 2.0 implementation of Listen, attend and spell (https://arxiv.org/abs/1508.01211).
To get a better understanding of the naming of the models variables please see the paper above.

#### Done:
+ [x] Model architecture looks right to me. If you find an error in the code please dont hesitate to open an issue 😊

#### ToDo:
+ [ ] Implement data handing for easier training of model.
+ [ ] Train on LibriSpeech 100h
+ [ ] Implement specAugment features (prev SOTA LibriSpeech) (https://arxiv.org/abs/1904.08779)

#### Usage
The file model.py contains the architecture of the model. Example usage below.

```python
"""
def LAS(dim, f_1, no_tokens):
  dim: Number of hidden neurons for most LSTM's.
  f_1: pBLSTM takes (Batch, timesteps, f_1) as input, f_1 is number of features of the mel spectrogram 
       per timestep. Timestep is the width of the spectrogram.
  No_tokens: Number of unique tokens for input and output vector.
"""

model = LAS(256, 256, 16)
model.compile(loss="mse", optimizer="adam")

# x_1 should have shape (Batch-size, timesteps, f_1)
x_1 = np.random.random((1, 550, 256))

# x_2 should have shape (Batch-size, no_prev_tokens, No_tokens). 
x_2 = np.random.random((1, 12, 16))

# By passing x_1 and x_2 the model will predict the 12th token 
# given by the spectogram and the prev predicted tokens

model(x=[x_1, x_2])
```
