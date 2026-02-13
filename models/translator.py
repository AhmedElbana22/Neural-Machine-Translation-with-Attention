import tensorflow as tf
from models.encoder import Encoder
from models.decoder import Decoder

class Translator(tf.keras.Model):
    def __init__(self, vocab_size, units):
        super().__init__()
        self.encoder = Encoder(vocab_size, units)
        self.decoder = Decoder(vocab_size, units)
    
    def call(self, inputs):
        context, target = inputs
        encoded_context = self.encoder(context)
        logits = self.decoder(encoded_context, target)
        return logits