import tensorflow as tf
from models.attention import CrossAttention

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, units):
        super(Decoder, self).__init__()
        
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=units,
            mask_zero=True
        )
        
        self.pre_attention_rnn = tf.keras.layers.LSTM(
            units=units,
            return_sequences=True,
            return_state=True
        )
        
        self.attention = CrossAttention(units)
        
        self.post_attention_rnn = tf.keras.layers.LSTM(
            units=units,
            return_sequences=True
        )
        
        self.output_layer = tf.keras.layers.Dense(
            units=vocab_size,
            activation=tf.nn.log_softmax
        )
    
    def call(self, context, target, state=None, return_state=False):
        x = self.embedding(target)
        x, hidden_state, cell_state = self.pre_attention_rnn(x, initial_state=state)
        x = self.attention(context, x)
        x = self.post_attention_rnn(x)
        logits = self.output_layer(x)
        
        if return_state:
            return logits, [hidden_state, cell_state]
        
        return logits