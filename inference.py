import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from models.translator import Translator
from utils.data_loader import prepare_datasets, MAX_VOCAB_SIZE
from utils.metrics import rouge1_similarity, jaccard_similarity

UNITS = 256

_, _, english_vectorizer, portuguese_vectorizer = prepare_datasets()

word_to_id = tf.keras.layers.StringLookup(
    vocabulary=portuguese_vectorizer.get_vocabulary(),
    mask_token="",
    oov_token="[UNK]"
)

id_to_word = tf.keras.layers.StringLookup(
    vocabulary=portuguese_vectorizer.get_vocabulary(),
    mask_token="",
    oov_token="[UNK]",
    invert=True
)

sos_id = word_to_id("[SOS]")
eos_id = word_to_id("[EOS]")

def tokens_to_text(tokens):
    words = id_to_word(tokens)
    return tf.strings.reduce_join(words, axis=-1, separator=' ')

def generate_next_token(decoder, context, next_token, done, state, temperature=0.0):
    logits, state = decoder(context, next_token, state=state, return_state=True)
    logits = logits[:, -1, :]
    
    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)
    else:
        logits = logits / temperature
        next_token = tf.random.categorical(logits, num_samples=1)
    
    logits = tf.squeeze(logits)
    next_token = tf.squeeze(next_token)
    logit = logits[next_token].numpy()
    next_token = tf.reshape(next_token, shape=(1,1))
    
    if next_token == eos_id:
        done = True
    
    return next_token, logit, state, done

def translate(model, text, max_length=50, temperature=0.0):
    tokens, logits = [], []
    
    text = tf.constant(text)[tf.newaxis]
    context = english_vectorizer(text).to_tensor()
    context = model.encoder(context)
    
    next_token = tf.fill((1, 1), sos_id)
    units = context.shape[-1]
    state = [tf.zeros((1, units)), tf.zeros((1, units))]
    done = False
    
    for _ in range(max_length):
        next_token, logit, state, done = generate_next_token(
            decoder=model.decoder,
            context=context,
            next_token=next_token,
            done=done,
            state=state,
            temperature=temperature
        )
        
        if done:
            break
        
        tokens.append(next_token)
        logits.append(logit)
    
    tokens = tf.concat(tokens, axis=-1)
    translation = tf.squeeze(tokens_to_text(tokens))
    translation = translation.numpy().decode()
    
    return translation, logits[-1] if logits else 0, tokens

def generate_samples(model, text, n_samples=4, temperature=0.6):
    samples, log_probs = [], []
    
    for _ in range(n_samples):
        _, logp, sample = translate(model, text, temperature=temperature)
        samples.append(np.squeeze(sample.numpy()).tolist())
        log_probs.append(logp)
    
    return samples, log_probs

def weighted_avg_overlap(samples, log_probs, similarity_fn):
    scores = {}
    
    for idx_candidate, candidate in enumerate(samples):
        overlap, weight_sum = 0.0, 0.0
        
        for idx_sample, (sample, logp) in enumerate(zip(samples, log_probs)):
            if idx_candidate == idx_sample:
                continue
            
            sample_p = float(np.exp(logp))
            weight_sum += sample_p
            overlap += sample_p * similarity_fn(candidate, sample)
        
        scores[idx_candidate] = round(overlap / weight_sum, 3) if weight_sum > 0 else 0
    
    return scores

def mbr_decode(model, text, n_samples=10, temperature=0.6, similarity_fn=rouge1_similarity):
    samples, log_probs = generate_samples(model, text, n_samples, temperature)
    scores = weighted_avg_overlap(samples, log_probs, similarity_fn)
    
    decoded = [tokens_to_text(s).numpy().decode('utf-8') for s in samples]
    best_idx = max(scores, key=lambda k: scores[k])
    
    return decoded[best_idx], decoded

if __name__ == "__main__":
    translator = Translator(MAX_VOCAB_SIZE, UNITS)
    translator.load_weights('checkpoints/translator_weights.h5')
    
    test_sentence = "I love languages"
    
    translation, _ = translate(translator, test_sentence, temperature=0.0)
    print(f"Greedy: {translation}")
    
    translation, candidates = mbr_decode(translator, test_sentence)
    print(f"MBR: {translation}")