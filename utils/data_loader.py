import os
import numpy as np
import tensorflow as tf
import pathlib
import re
import unicodedata

np.random.seed(1234)
tf.random.set_seed(1234)

BUFFER_SIZE = None
BATCH_SIZE = 64
MAX_VOCAB_SIZE = 12000

def load_data(path):
    """Load Portuguese-English translation pairs from file"""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    pairs = [line.split("\t") for line in lines]
    
    context = np.array([context for target, context, _ in pairs])
    target = np.array([target for target, context, _ in pairs])
    
    return context, target

class CustomStandardization(tf.keras.layers.Layer):
    """Custom text standardization layer (replaces tf_text.normalize_utf8)"""
    
    def call(self, inputs):
        # Lowercase
        text = tf.strings.lower(inputs)
        # Remove characters except letters, spaces, and punctuation
        text = tf.strings.regex_replace(text, "[^ a-z.?!,¿]", "")
        # Add spaces around punctuation
        text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")
        # Strip whitespace
        text = tf.strings.strip(text)
        # Add SOS and EOS tokens
        text = tf.strings.join(["[SOS]", text, "[EOS]"], separator=" ")
        return text

def prepare_datasets(data_path=None):
    """Prepare training and validation datasets"""
    global BUFFER_SIZE
    
    # Auto-detect data path if not provided
    if data_path is None:
        current_file = pathlib.Path(__file__).resolve()
        project_root = current_file.parent.parent
        path_to_file = project_root / "data" / "por-eng" / "por.txt"
    else:
        path_to_file = pathlib.Path(data_path)
    
    # Check if file exists
    if not path_to_file.exists():
        raise FileNotFoundError(
            f"Data file not found at: {path_to_file}\n"
            f"Expected location: {path_to_file.resolve()}\n"
            f"Current working directory: {os.getcwd()}"
        )
    
    print(f"Loading data from: {path_to_file}")
    
    # Load data
    portuguese_sentences, english_sentences = load_data(path_to_file)
    
    BUFFER_SIZE = len(english_sentences)
    
    # Split train/val
    is_train = np.random.uniform(size=(len(portuguese_sentences),)) < 0.8
    
    train_raw = (
        tf.data.Dataset.from_tensor_slices(
            (english_sentences[is_train], portuguese_sentences[is_train])
        )
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )
    
    val_raw = (
        tf.data.Dataset.from_tensor_slices(
            (english_sentences[~is_train], portuguese_sentences[~is_train])
        )
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )
    
    # Create custom standardization
    custom_standardization = CustomStandardization()
    
    # Create vectorizers
    english_vectorizer = tf.keras.layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=MAX_VOCAB_SIZE,
        ragged=True
    )
    english_vectorizer.adapt(train_raw.map(lambda context, target: context))
    
    portuguese_vectorizer = tf.keras.layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=MAX_VOCAB_SIZE,
        ragged=True
    )
    portuguese_vectorizer.adapt(train_raw.map(lambda context, target: target))
    
    # Process text function (matches original)
    def process_text(context, target):
        context = english_vectorizer(context).to_tensor()
        target = portuguese_vectorizer(target)
        targ_in = target[:, :-1].to_tensor()
        targ_out = target[:, 1:].to_tensor()
        return (context, targ_in), targ_out
    
    train_data = train_raw.map(process_text, tf.data.AUTOTUNE)
    val_data = val_raw.map(process_text, tf.data.AUTOTUNE)
    
    return train_data, val_data, english_vectorizer, portuguese_vectorizer

def tokens_to_text(tokens, id_to_word):
    """Convert token IDs to text"""
    words = id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=" ")
    return result