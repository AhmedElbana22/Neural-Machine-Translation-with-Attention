import tensorflow as tf
from collections import Counter

def masked_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    loss = loss_fn(y_true, y_pred)
    
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask
    
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    
    return tf.reduce_sum(match) / tf.reduce_sum(mask)

def jaccard_similarity(candidate, reference):
    candidate_set = set(candidate)
    reference_set = set(reference)
    common_tokens = candidate_set.intersection(reference_set)
    all_tokens = candidate_set.union(reference_set)
    return len(common_tokens) / len(all_tokens) if all_tokens else 0

def rouge1_similarity(candidate, reference):
    candidate_counts = Counter(candidate)
    reference_counts = Counter(reference)
    
    overlap = sum(min(candidate_counts[token], reference_counts.get(token, 0)) 
                  for token in candidate_counts.keys())
    
    precision = overlap / len(candidate) if candidate else 0
    recall = overlap / len(reference) if reference else 0
    
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0