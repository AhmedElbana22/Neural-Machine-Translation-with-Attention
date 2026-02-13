import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, units):
        super(Encoder, self).__init__()
        
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=units,
            mask_zero=True
        )
        
        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode="sum",
            layer=tf.keras.layers.LSTM(
                units=units,
                return_sequences=True
            ),
        )
    
    def call(self, context):
        x = self.embedding(context)
        x = self.rnn(x)
        return x