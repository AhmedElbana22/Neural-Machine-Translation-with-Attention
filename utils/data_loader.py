import pathlib
import numpy as np
import tensorflow as tf
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

def unicode_to_ascii(text):
    """Normalize unicode characters to ASCII"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def preprocess_sentence(sentence):
    """Preprocess a single sentence"""
    sentence = unicode_to_ascii(sentence.lower().strip())
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = sentence.strip()
    sentence = '[SOS] ' + sentence + ' [EOS]'
    return sentence

class CustomStandardization(tf.keras.layers.Layer):
    """Custom text standardization layer to replace tensorflow-text"""
    
    def call(self, inputs):
        lowercase = tf.strings.lower(inputs)
        cleaned = tf.strings.regex_replace(lowercase, "[^ a-z.?!,¿]", "")
        spaced = tf.strings.regex_replace(cleaned, "[.?!,¿]", r" \0 ")
        stripped = tf.strings.strip(spaced)
        result = tf.strings.join(["[SOS]", stripped, "[EOS]"], separator=" ")
        return result

def prepare_datasets(data_path="data/por-eng/por.txt"):
    """Prepare training and validation datasets"""
    global BUFFER_SIZE
    
    path_to_file = pathlib.Path(data_path)
    portuguese_sentences, english_sentences = load_data(path_to_file)
    
    BUFFER_SIZE = len(english_sentences)
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
    
    custom_standardization = CustomStandardization()
    
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
    
    def process_text(context, target):
        context = english_vectorizer(context).to_tensor()
        target = portuguese_vectorizer(target).to_tensor()
        target_input = target[:, :-1]
        target_output = target[:, 1:]
        return (context, target_input), target_output
    
    train_data = train_raw.map(process_text, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_data = val_raw.map(process_text, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    return train_data, val_data, english_vectorizer, portuguese_vectorizer

def tokens_to_text(tokens, id_to_word):
    """Convert token IDs to text"""
    words = id_to_word(tokens)
    return tf.strings.reduce_join(words, axis=-1, separator=' ')