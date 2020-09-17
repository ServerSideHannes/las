import tensorflow as tf
import numpy as np

class attention(tf.keras.layers.Layer):
  def __init__(self, dim):
    super(attention, self).__init__()
    
    self.dim      = dim
    self.dense_s  = tf.keras.layers.Dense(self.dim)
    self.dense_h  = tf.keras.layers.Dense(self.dim)
    
  def call(self, inputs):
    # Split inputs into attentions vectors and inputs from the LSTM output
    s     = inputs[0] # (..., s_depth)
    h     = inputs[1] # (..., seq_len, h_depth)
    
    # Linear FC
    s_fi   = self.dense_s(s) # (..., F)
    h_psi  = self.dense_h(h) # (..., seq_len, F)
    
    # Linear blendning < φ(s_i), ψ(h_u) >
    e = tf.matmul(s_fi*h_psi, transpose_b=True) # (..., 1, B)
    
    # Softmax vector
    alpha = tf.nn.softmax(e) # (..., 1, N)

    # Context vector
    c = tf.matmul(alpha*h) # (..., 1, F)
    c = tf.squeeze(c, 1) # (..., F)
    
    return c

class att_rnn( tf.keras.layers.Layer):
  def __init__(self, units,):
    super(att_rnn, self).__init__()
    self.units      = units
    self.state_size = [self.units, self.units]
    
    self.attention_context  = attention(self.units)
    self.rnn                = tf.keras.layers.LSTMCell(self.units)
    self.rnn2               = tf.keras.layers.LSTMCell(self.units)
    
  def call(self, inputs, states, constants):
    h       = tf.squeeze(constants, axis=0)

    s       = self.rnn(inputs=inputs, states=states)
    s       = self.rnn2(inputs=s[0], states=s[1])[1]

    c       = self.attention_context([s[0], h]) # (..., F)
    out     = tf.keras.layers.concatenate([s[0], c], axis=-1) # (..., F*2)
    
    return out, [s[0], c]

class pBLSTM(tf.keras.layers.Layer):
  def __init__(self, dim):
    super(pBLSTM, self).__init__()
    
    self.dim        = dim
    self.LSTM       = tf.keras.layers.LSTM(self.dim, return_sequences=True)
    self.bidi_LSTM  = tf.keras.layers.Bidirectional(self.LSTM)
    
  @tf.function
  def call(self, inputs):
    y = self.bidi_LSTM(inputs) # (..., seq_len, F)
    
    if( tf.shape(inputs)[1] % 2 == 1):
      y = tf.keras.layers.ZeroPadding1D(padding=(0, 1))(y)

    y = tf.keras.layers.Reshape(target_shape=(-1, int(self.dim*4)))(y) # (..., seq_len//2, F*2)
    return y

def LAS(dim, f_1, no_tokens):
  input_1 = tf.keras.Input(shape=(None, f_1))
  input_2 = tf.keras.Input(shape=(None, no_tokens))
  
  #Listen; Lower resoultion by 8x
  x = pBLSTM( dim//2 )(input_1)
  x = pBLSTM( dim//2 )(x)
  x = pBLSTM( dim//2 )(x)
  
  #Attend
  x = tf.keras.layers.RNN(att_rnn(dim), return_sequences=True)(input_2, constants=x)
  
  #Spell
  x = tf.keras.layers.Dense(dim, activation="relu")(x)
  x = tf.keras.layers.Dense(no_tokens, activation="softmax")(x)

  model = tf.keras.Model(inputs=[input_1,input_2], outputs=x)
  return model

model = LAS(256, 256, 16)
model.compile(loss="mse", optimizer="adam")
