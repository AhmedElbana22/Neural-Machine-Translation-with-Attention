import tensorflow as tf

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        
        self.mha = tf.keras.layers.MultiHeadAttention(
            key_dim=units,
            num_heads=1
        )
        
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
    
    def call(self, context, target):
        attn_output = self.mha(query=target, value=context)
        x = self.add([target, attn_output])
        x = self.layernorm(x)
        return x