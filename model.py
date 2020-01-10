import tensorflow as tf
import numpy as np

class attention(tf.keras.layers.Layer):
  def __init__(self, dim):
    super(attention, self).__init__()
    
    self.dim = dim
    self.dense_s = tf.keras.layers.Dense(self.dim)
    
  def call(self, inputs):
    #Split inputs into attentions vectors and inputs from the LSTM output
    lstm_out      = inputs[0]
    attention_vec = inputs[1]
    
    #Linear FC
    lstm_out = self.dense_s(lstm_out)
    
    #Linear blendning
    alpha = tf.keras.backend.expand_dims(lstm_out)
    alpha = tf.matmul(attention_vec, alpha)
    alpha = tf.keras.backend.squeeze(alpha, axis=-1)

    #softmax_vector
    softmaxed = tf.nn.softmax(alpha, axis=-1)
    softmaxed = tf.keras.backend.expand_dims(softmaxed, axis=-2)

    #Wheighted vector fetures
    y = tf.matmul(softmaxed, attention_vec)
    y = tf.keras.backend.squeeze(y, axis=-2)

    return y

class att_rnn( tf.keras.layers.Layer):
  def __init__(self, units,):
    super(att_rnn, self).__init__()
    self.units      = units
    self.state_size = [self.units, self.units]
    
    self.attention_cell = attention(self.units)
    self.rnn            = tf.keras.layers.LSTMCell(self.units)
    self.rnn2           = tf.keras.layers.LSTMCell(self.units)
    
  def call(self, inputs, states, constants):
    constants = tf.squeeze(constants, axis=0)

    r         = self.rnn(inputs=inputs, states=states)[1]
    r         = self.rnn2(inputs=r[0], states=[r[0], r[1]])[1]
    
    c         = self.attention_cell([r[0], constants])
    c         = tf.add(r[1], c)
    
    return r[0], [r[0], c]

class pBLSTM(tf.keras.layers.Layer):
  def __init__(self, dim):
    super(pBLSTM, self).__init__()
    
    self.dim        = dim
    self.LSTM       = tf.keras.layers.LSTM(self.dim, return_sequences=True)
    self.bidi_LSTM  = tf.keras.layers.Bidirectional(self.LSTM)
    
  @tf.function
  def call(self, inputs):
    y = self.bidi_LSTM(inputs)
    
    if( int(tf.shape(inputs)[1]) % 2 == 1):
      y = tf.keras.layers.ZeroPadding1D(padding=(0, 1))(y)

    y = tf.keras.layers.Reshape(target_shape=(-1, int(self.dim*4)))(y)
    return y

def LAS(dim, f_1, no_tokens):
  input_1 = tf.keras.Input(shape=(None, f_1))
  input_2 = tf.keras.Input(shape=(None, no_tokens))
  
  #Lower resoultion by 8x
  #dim/2 is used since blstm use concat and therefore would return dim*2. It just looked cleaner to me.
  
  x = pBLSTM( int(dim/2) )(input_1)
  x = pBLSTM( int(dim/2) )(x)
  x = pBLSTM( int(dim/2) )(x)
  
  x = tf.keras.layers.Dense(dim,)(x)
  x = tf.keras.layers.RNN(att_rnn(dim), return_state=True)(input_2, constants=x)

  x = tf.keras.layers.concatenate([x[1], x[2]], axis=-1)
  x = tf.keras.layers.Dense(dim, activation="relu")(x)
  x = tf.keras.layers.Dense(no_tokens, activation="softmax")(x)

  model = tf.keras.Model(inputs=[input_1,input_2] , outputs=x)
  return model

model = LAS(256, 512, 16)
model.compile(loss="mse", optimizer="adam")
