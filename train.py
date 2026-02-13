import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from models.translator import Translator
from utils.data_loader import prepare_datasets, MAX_VOCAB_SIZE
from utils.metrics import masked_loss, masked_acc

UNITS = 256
EPOCHS = 20
STEPS_PER_EPOCH = 500

def train():
    print("Loading and preparing datasets...")
    train_data, val_data, english_vectorizer, portuguese_vectorizer = prepare_datasets()
    
    print("Building model...")
    translator = Translator(MAX_VOCAB_SIZE, UNITS)
    
    translator.compile(
        optimizer="adam",
        loss=masked_loss,
        metrics=[masked_acc, masked_loss]
    )
    
    print("Training model...")
    history = translator.fit(
        train_data.repeat(),
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_data,
        validation_steps=50,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
    )
    
    print("Saving model...")
    translator.save_weights('checkpoints/translator_weights.h5')
    
    return translator, history

if __name__ == "__main__":
    translator, history = train()