#coding=utf-8

import tensorflow as tf
import numpy as np

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
    
        self.warmup_steps = warmup_steps
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
def create_padding_mask(seq,padId):
    seq = tf.cast(tf.math.equal(seq, padId), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
def create_masks(src, tar,padId):
    enc_src_padding_mask = create_padding_mask(src,padId)#[batch,1,1,seq_len]
    dec_src_padding_mask = create_padding_mask(src,padId)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])#[seq_len,seq_len]
    dec_target_padding_mask = create_padding_mask(tar,padId)#[batch,1,1,seq_len]
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_src_padding_mask, combined_mask, dec_src_padding_mask

def get_angles(pos, i, d_model):
    """
    parameter:
        pos:(position * 1)
        i:(1 * d)
    return:(position * d)
    """
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
def positional_encoding(position, d_model):
#     print(np.arange(position)[:, np.newaxis])
#     print(np.arange(d_model)[np.newaxis, :])
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], q.dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        mask = tf.cast(mask,scaled_attention_logits.dtype)
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),# (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)])  # (batch_size, seq_len, d_model)