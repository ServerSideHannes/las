# Listen, attend and spell
tf 2.0 implementation of Listen, attend and spell (Arxiv https://arxiv.org/abs/1508.01211)

#### Done:
Model architecture looks right to me. If you find an error in the code please dont hesitate to open an issue 😊

#### ToDo:
Implement data handing for easier training of model.
Train on LibriSpeech 100h

#### Usage
The file model.py contains the architecture of the model. Example usage below.

```python
"""
def LAS(dim, f_1, no_tokens):
  dim: hidden neurons for all LSTM's.
  No_tokens: the length of the one hot encoded vector for token inputs.
  f_1: pBLSTM takes (Batch, timesteps, f_1) as input, f_1 is number of features of the mel spectrogram per timestep.
"""

model = LAS(256, 512, 16)
model.compile(loss="mse", optimizer="adam")
```
