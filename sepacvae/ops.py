
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk import sent_tokenize, word_tokenize, ngrams
from tensorflow.python.ops import rnn_cell_impl as rnn_cell


def get_bleu_stats(ref, hyps):
    scores = []
    for hyp in hyps:
        try:
            scores.append(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7,
                                        weights=[1./3, 1./3,1./3]))
        except:
            scores.append(0.0)
    return np.max(scores), np.mean(scores)


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
                               - tf.div(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
                               - tf.div(tf.exp(recog_logvar), tf.exp(prior_logvar)), reduction_indices=1)
    return kld


def norm_log_liklihood(x, mu, logvar):
    return -0.5*tf.reduce_sum(tf.log(2*np.pi) + logvar + tf.div(tf.pow((x-mu), 2), tf.exp(logvar)), reduction_indices=1)


def sample_gaussian(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
    std = tf.exp(0.5 * logvar)
    z= mu + tf.multiply(std, epsilon)
    return z


def get_bow(embedding, avg=False):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    """
    embedding_size = embedding.get_shape()[2].value
    if avg:
        return tf.reduce_mean(embedding, reduction_indices=[1]), embedding_size
    else:
        return tf.reduce_sum(embedding, reduction_indices=[1]), embedding_size


def get_rnn_encode(embedding, cell, length_mask=None, scope=None, reuse=None):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    The padding should have zero
    """
    with tf.variable_scope(scope, 'RnnEncoding', reuse=reuse):
        if length_mask is None:
            length_mask = tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(embedding), reduction_indices=2)),reduction_indices=1)
            length_mask = tf.to_int32(length_mask)
        _, encoded_input = tf.nn.dynamic_rnn(cell, embedding, sequence_length=length_mask, dtype=tf.float32)
        return encoded_input, cell.state_size


def get_bi_rnn_encode(embedding, f_cell, b_cell, length_mask=None, scope=None, reuse=None):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    The padding should have zero
    """
    with tf.variable_scope(scope, 'RnnEncoding', reuse=reuse):
        if length_mask is None:
            length_mask = tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(embedding), reduction_indices=2)),reduction_indices=1)
            length_mask = tf.to_int32(length_mask)
        _, encoded_input = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, embedding, sequence_length=length_mask, dtype=tf.float32)
        encoded_input = tf.concat(encoded_input, 1)
        return encoded_input, f_cell.state_size+b_cell.state_size

def ppl_layer(logits, labels, mask, seq_lens):
    # 先求每个样本的交叉熵
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    #logits = tf.nn.softmax(logits)
    #label_one_hot = tf.one_hot(labels, tf.shape(logits)[-1])
    #entropy = -tf.log(tf.reduce_max(logits*label_one_hot, 2)+1e-7)
    # 去掉pad的影响
    entropy = entropy * mask
    # 对每个样本的交叉熵求和 sum(-logp)
    entropy = tf.reduce_sum(entropy, 1)
    # 求每句话的ppl
    ppl = tf.exp(entropy / tf.cast(seq_lens, tf.float32))
    # 求所有样本的ppl均值
    ppl = tf.reduce_mean(ppl)
    return ppl


def expand_direction(tensor, direction_num, shape, shape_dims):
    tensor = tf.expand_dims(tensor, 1)
    tensor = tf.concat([tensor for _ in range(direction_num)], 1)
    if shape_dims == 3:
        tensor = tf.reshape(tensor, [shape[0]*direction_num, shape[1], shape[2]])
    elif shape_dims == 2:
        tensor = tf.reshape(tensor, [shape[0]*direction_num, -1])
    elif shape_dims == 1:
        tensor = tf.reshape(tensor, [shape[0]*direction_num])
    return tensor


def build_train_mask(entiry_losses, config):
    print(entiry_losses)
    avg_loss = tf.reshape(entiry_losses, [config.batch_size, config.direction_num])
    best_ids = tf.argmin(avg_loss, 1, output_type=tf.int32)
    print(best_ids)
    
#    lens = int(np.array(tf.shape(entiry_losses))[1])
    
    one_tensor = tf.ones([1, 1], dtype=tf.float32)
    zero_tensor = tf.zeros([1, 1], dtype=tf.float32)
#    zero_tensor = -0.00000001 * tf.ones([1, 1], dtype=tf.float32)
    
    mask = tf.cond(tf.equal(best_ids[0],tf.constant([0], dtype=tf.int32))[0], lambda: one_tensor, lambda:zero_tensor)
    for i in range(config.batch_size):
        for j in range(config.direction_num):
            if i==0 and j==0:
                continue
            mask = tf.concat([
                    mask,
                    tf.cond(tf.equal(best_ids[i],tf.constant([j], dtype=tf.int32))[0], lambda: one_tensor, lambda:zero_tensor),
                    ], 0)
            
    return mask


def offset_conv1d(input_, name, config, reuse=True):
    if reuse:
        w = tf.get_variable(name)
        w2 = tf.get_variable(name + '_ff_weights')
    else:
        w = tf.get_variable(name, (config.offset_cnn_h, config.offset_cnn_w, 1, config.offset_cnn_p),
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        w2 = tf.get_variable(name + '_ff_weights', (config.offset_cnn_p-config.offset_cnn_h+1, config.offset_cnn_p),
                             initializer=tf.truncated_normal_initializer(stddev=0.01))
        
    input_ = tf.expand_dims(input_, 3)

    offset_conv = tf.transpose(tf.nn.conv2d(input_, w, strides=(1, 1, 1, 1), padding='VALID'),
                             [0, 3, 1, 2])
    offset_conv = tf.squeeze(offset_conv, 3)
    offset_logits = tf.nn.softmax(tf.einsum('ibn,nd->ibd', offset_conv, w2))

    # conv [batch_size, token_size]
    # 离散化
    offset_hard = tf.cast(tf.equal(offset_logits, tf.reduce_max(offset_logits, 1, keep_dims=True)), offset_logits.dtype)
    offset = tf.stop_gradient(offset_hard - offset_logits) + offset_logits

    # 修改原始数据
    output_ = tf.matmul(offset, tf.squeeze(input_, 3))

    return output_, tf.argmax(offset, 2)


def offset_rnn(input_, name, config, reuse=True):
    if config.offset_rnn_cell_type == "gru":
        cell = rnn_cell.GRUCell(config.offset_rnn_cell_size)
    else:
        cell = rnn_cell.LSTMCell(config.offset_rnn_cell_size, use_peepholes=False, forget_bias=1.0)
        
    length_mask = tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(input_), reduction_indices=2)),reduction_indices=1)
    length_mask = tf.to_int32(length_mask)
    rnn_output, _ = tf.nn.dynamic_rnn(cell, input_, sequence_length=length_mask, dtype=tf.float32)
    
    K_ = rnn_output # shape [batch_size, max_time, cell.output_size]
    Q_ = tf.transpose(rnn_output, (0,2,1))
    V_ = rnn_output
    
    offset_prob = tf.matmul(K_, Q_)
    offset_hard = tf.cast(tf.equal(offset_prob, tf.reduce_max(offset_prob, 1, keep_dims=True)), offset_prob.dtype)
    
    offset_prob_hard = tf.stop_gradient(offset_hard - offset_prob) + offset_prob
    
    output_ = tf.matmul(offset_prob_hard, V_)
    
    return output_, tf.argmax(offset_prob_hard, 2)


def offset_birnn(input_, name, config, reuse=True):
    if config.offset_rnn_cell_type == "gru":
        f_cell = rnn_cell.GRUCell(config.offset_rnn_cell_size)
        b_cell = rnn_cell.GRUCell(config.offset_rnn_cell_size)
    else:
        f_cell = rnn_cell.LSTMCell(config.offset_rnn_cell_size, 
                                   use_peepholes=False, forget_bias=1.0)
        b_cell = rnn_cell.LSTMCell(config.offset_rnn_cell_size, 
                                   use_peepholes=False, forget_bias=1.0)
        
    length_mask = tf.reduce_sum(
            tf.sign(tf.reduce_max(
                    tf.abs(input_), reduction_indices=2)),
                    reduction_indices=1)
    length_mask = tf.to_int32(length_mask)
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(
            f_cell, b_cell, input_, 
            sequence_length=length_mask, 
            dtype=tf.float32)
    rnn_output_fw, rnn_output_bw = rnn_output
    rnn_output = tf.add(rnn_output_fw, rnn_output_bw)
    
    K_ = rnn_output # shape [batch_size, max_time, cell.output_size]
    Q_ = tf.transpose(rnn_output, (0,2,1))
    V_ = rnn_output
    
    offset_prob = tf.matmul(K_, Q_)
    offset_hard = tf.cast(tf.equal(offset_prob, tf.reduce_max(offset_prob, 1, keep_dims=True)), offset_prob.dtype)
    
    offset_prob_hard = tf.stop_gradient(offset_hard - offset_prob) + offset_prob
    
    output_ = tf.matmul(offset_prob_hard, V_)
    
    return output_, tf.argmax(offset_prob_hard, 2)


def offset_selfattention(input_, name, config, reuse=True):
    K_ = input_ # shape [b, word_len, embedding]
    Q_ = tf.transpose(input_, (0,2,1))
    V_ = input_
    
    offset_prob = tf.matmul(K_, Q_)
    offset_hard = tf.cast(tf.equal(offset_prob, tf.reduce_max(offset_prob, 1, keep_dims=True)), offset_prob.dtype)
    
    offset_prob_hard = tf.stop_gradient(offset_hard - offset_prob) + offset_prob
    
    output_ = tf.matmul(offset_prob_hard, V_)
    
    return output_, tf.argmax(offset_prob_hard, 2)
    

def _response_tokenize(response):
    """
    Function: 将每个response进行tokenize
    Return: [token1, token2, ......]
    """
    #    response_tokens = []
    ##        vocab=self._get_vocab()
    #    for sentence in sent_tokenize(response):
    #        for token in word_tokenize(sentence):
    #           # if token in vocab:
    #            response_tokens.append(token)
    response_tokens = response.split()

    return response_tokens


def get_dp_gan_metrics(gen_responses):
    """
    Function：计算所有true_responses、gen_responses的
              token_gram、unigram、bigram、trigram、sent_gram的数量
    Return：token_gram、unigram、bigram、trigram、sent_gram的数量
    """
    responses = gen_responses

    token_gram = []
    unigram = []
    bigram = []
    trigram = []
    sent_gram = []

    for response in responses:
        tokens = _response_tokenize(response)
        token_gram.extend(tokens)
        unigram.extend([element for element in ngrams(tokens, 1)])
        bigram.extend([element for element in ngrams(tokens, 2)])
        trigram.extend([element for element in ngrams(tokens, 3)])
        sent_gram.append(response)

    return len(token_gram), len(set(unigram)), len(set(bigram)), \
           len(set(trigram)), len(set(sent_gram))


def get_distinct(gen_responses, n):
    """
    Function: 计算所有true_responses、gen_responses的ngrams的type-token ratio
    Return: ngrams-based type-token ratio
    """
    ngrams_list = []
    token_gram = []
    responses = gen_responses

    for response in responses:
        tokens = _response_tokenize(response)
        ngrams_list.extend([element for element in ngrams(tokens, n)])

    if len(ngrams_list) == 0:
        return 0
    else:
        return len(set(ngrams_list)) / len(ngrams_list)


def get_response_length(gen_responses):
    """ Reference:
         1. paper : Iulian V. Serban,et al. A Deep Reinforcement Learning Chatbot
    """
    response_lengths = []
    for gen_response in gen_responses:
        response_lengths.append(len(_response_tokenize(gen_response)))

    if len(response_lengths) == 0:
        return 0
    else:
        return sum(response_lengths) / len(response_lengths)


def get_bleu(true_responses, gen_responses, n_gram):
    """
    Function: 计算所有true_responses、gen_responses的ngrams的bleu

    parameters:
        n_gram : calculate BLEU-n,
                 calculate the cumulative 4-gram BLEU score, also called BLEU-4.
                 The weights for the BLEU-4 are 1/4 (25%) or 0.25 for each of the 1-gram, 2-gram, 3-gram and 4-gram scores.

    Reference:
        1. https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
        2. https://cloud.tencent.com/developer/article/1042161

    Return: bleu score BLEU-n
    """
    weights = {1: (1.0, 0.0, 0.0, 0.0),
               2: (1 / 2, 1 / 2, 0.0, 0.0),
               3: (1 / 3, 1 / 3, 1 / 3, 0.0),
               4: (1 / 4, 1 / 4, 1 / 4, 1 / 4)}
    total_score = []
    for true_response, gen_response in zip(true_responses, gen_responses):
        if len(_response_tokenize(gen_response)) <= 1:
            total_score.append(0)
            continue
        score = sentence_bleu(
            [_response_tokenize(true_response)],
            _response_tokenize(gen_response),
            weights[n_gram],
            smoothing_function=SmoothingFunction().method7)
        total_score.append(score)

    if len(total_score) == 0:
        return 0
    else:
        return sum(total_score) / len(total_score)