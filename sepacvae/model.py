import os
import re
import sys
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import OutputProjectionWrapper
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn_cell_impl as rnn_cell
from tensorflow.python.ops import variable_scope

import decoder_fn_lib as decoder_fn_lib
from seq2seq import dynamic_rnn_decoder
from ops import gaussian_kld
from ops import get_bi_rnn_encode
from ops import get_bow
from ops import get_rnn_encode
from ops import norm_log_liklihood
from ops import sample_gaussian
from ops import build_train_mask
from ops import expand_direction

class BaseTFModel(object):
    global_t = tf.placeholder(dtype=tf.int32, name="global_t")
    learning_rate = None
    scope = None

    @staticmethod
    def print_model_stats(tvars):
        total_parameters = 0
        for variable in tvars:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            print("Trainable %s with %d parameters" % (variable.name, variable_parametes))
            total_parameters += variable_parametes
        print("Total number of trainable parameters is %d" % total_parameters)

    @staticmethod
    def get_rnncell(cell_type, cell_size, keep_prob, num_layer):
        # thanks for this solution from @dimeldo
        cells = []
        for _ in range(num_layer):
            if cell_type == "gru":
                cell = rnn_cell.GRUCell(cell_size)
            else:
                cell = rnn_cell.LSTMCell(cell_size, use_peepholes=False, forget_bias=1.0)

            if keep_prob < 1.0:
                cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

            cells.append(cell)

        if num_layer > 1:
            cell = rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        else:
            cell = cells[0]

        return cell

    @staticmethod
    def print_loss(prefix, loss_names, losses, postfix):
        template = "%s "
        for name in loss_names:
            template += "%s " % name
            template += " %f "
        template += "%s"
        template = re.sub(' +', ' ', template)
        avg_losses = []
        values = [prefix]

        for loss in losses:
            values.append(np.mean(loss))
            avg_losses.append(np.mean(loss))
        values.append(postfix)

        print(template % tuple(values))
        return avg_losses

    def train(self, global_t, sess, train_feed):
        raise NotImplementedError("Train function needs to be implemented")

    def valid(self, *args, **kwargs):
        raise NotImplementedError("Valid function needs to be implemented")

    def batch_2_feed(self, *args, **kwargs):
        raise NotImplementedError("Implement how to unpack the back")

    def optimize(self, sess, config, loss, log_dir):
        if log_dir is None:
            return
        # optimization
        if self.scope is None:
            tvars = tf.trainable_variables()
        else:
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        grads = tf.gradients(loss, tvars)
        if config.grad_clip is not None:
            grads, _ = tf.clip_by_global_norm(grads, tf.constant(config.grad_clip))
        # add gradient noise
        if config.grad_noise > 0:
            grad_std = tf.sqrt(config.grad_noise / tf.pow(1.0 + tf.to_float(self.global_t), 0.55))
            grads = [g + tf.truncated_normal(tf.shape(g), mean=0.0, stddev=grad_std) for g in grads]

        if config.op == "adam":
            print("Use Adam")
            optimizer = tf.train.AdamOptimizer(config.init_lr)
        elif config.op == "rmsprop":
            print("Use RMSProp")
            optimizer = tf.train.RMSPropOptimizer(config.init_lr)
        else:
            print("Use SGD")
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.train_ops = optimizer.apply_gradients(zip(grads, tvars))
        self.print_model_stats(tvars)
        train_log_dir = os.path.join(log_dir, "checkpoints")
        print("Save summary to %s" % log_dir)
        self.train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)


class CVAE(BaseTFModel):

    def __init__(self, sess, config, api, log_dir, forward, scope=None):
        self.vocab = api.id2token
        self.rev_vocab = api.token2idx
        self.vocab_size = len(self.vocab)
        self.topic_vocab = api.topic_vocab
        self.topic_vocab_size = 0
        self.da_vocab = api.dialog_act_vocab
        self.da_vocab_size = 0
        self.sess = sess
        self.scope = scope
        self.maxlen1 = config.maxlen1
        self.maxlen2 = config.maxlen2
        self.go_id = self.rev_vocab["<s>"]
        self.eos_id = self.rev_vocab["</s>"]
        self.context_cell_size = config.cxt_cell_size
        self.sent_cell_size = config.sent_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.direction_num = config.direction_num
        self.dot_loss_weight = config.dot_loss_weight

        with tf.name_scope("io"):
            # all dialog context and known attributes
            self.input_contexts = tf.placeholder(dtype=tf.int32, shape=(None, None, self.maxlen1), name="dialog_context")
            self.floors = tf.placeholder(dtype=tf.int32, shape=(None, None), name="floor")
            self.context_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="context_lens")
            self.topics = tf.placeholder(dtype=tf.int32, shape=(None,), name="topics")
            self.my_profile = tf.placeholder(dtype=tf.float32, shape=(None, ), name="my_profile")
            self.ot_profile = tf.placeholder(dtype=tf.float32, shape=(None, ), name="ot_profile")

            # target response given the dialog context
            self.output_tokens = tf.placeholder(dtype=tf.int32, shape=(None, None), name="output_token")
            self.output_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="output_lens")
            self.output_das = tf.placeholder(dtype=tf.int32, shape=(None,), name="output_dialog_acts")

            # optimization related variables
            self.learning_rate = tf.Variable(float(config.init_lr), trainable=False, name="learning_rate")
            self.learning_rate_decay_op = self.learning_rate.assign(tf.multiply(self.learning_rate, config.lr_decay))
            self.global_t = tf.placeholder(dtype=tf.int32, name="global_t")
            self.use_prior = tf.placeholder(dtype=tf.bool, name="use_prior")

        if config.if_multi_direction:
            input_contexts = expand_direction(self.input_contexts, config.direction_num, [config.batch_size, 1, self.maxlen1], 3)
            floors = expand_direction(self.floors, config.direction_num, [config.batch_size, 1], 2)
            context_lens = expand_direction(self.context_lens, config.direction_num, [config.batch_size], 1)
            topics = expand_direction(self.topics, config.direction_num, [config.batch_size], 1)
            my_profile = expand_direction(self.my_profile, config.direction_num, [config.batch_size], 1)
            ot_profile = expand_direction(self.ot_profile, config.direction_num, [config.batch_size], 1)
            output_tokens = expand_direction(self.output_tokens, config.direction_num, [config.batch_size, 1], 2)
            output_lens = expand_direction(self.output_lens, config.direction_num, [config.batch_size, 1], 1)
        else:
            input_contexts = self.input_contexts
            floors = self.floors
            context_lens = self.context_lens
            topics = self.topics
            my_profile = self.my_profile
            ot_profile = self.ot_profile
            output_tokens = self.output_tokens
            output_lens = self.output_lens


        max_dialog_len = array_ops.shape(input_contexts)[1]
        max_out_len = array_ops.shape(output_tokens)[1]
        batch_size = array_ops.shape(input_contexts)[0]

#        with variable_scope.variable_scope("topicEmbedding"):
#            t_embedding = tf.get_variable("embedding", [self.topic_vocab_size, config.topic_embed_size], dtype=tf.float32)
#            topic_embedding = embedding_ops.embedding_lookup(t_embedding, self.topics)

#        if config.use_hcf:
#            with variable_scope.variable_scope("dialogActEmbedding"):
#                d_embedding = tf.get_variable("embedding", [self.da_vocab_size, config.da_embed_size], dtype=tf.float32)
#                da_embedding = embedding_ops.embedding_lookup(d_embedding, self.output_das)

        with variable_scope.variable_scope("wordEmbedding"):
            if api.word2vec is None:
                self.embedding = tf.get_variable("encoder_embedding", [self.vocab_size, config.embed_size],
                                                 dtype=tf.float32)
                print('generate variable')
            else:
                self.embedding = tf.constant(api.word2vec, dtype=tf.float32, shape=[self.vocab_size, config.embed_size])
                print('read from file')
            embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)], dtype=tf.float32,
                                         shape=[self.vocab_size, 1])
            embedding = self.embedding * embedding_mask

            input_embedding = embedding_ops.embedding_lookup(embedding, tf.reshape(input_contexts, [-1]))
            input_embedding = tf.reshape(input_embedding, [-1, self.maxlen1, config.embed_size])
            output_embedding = embedding_ops.embedding_lookup(embedding, output_tokens)

            if config.if_multi_direction:
                one_tensor = tf.ones([1, self.maxlen1, int(config.embed_size/self.direction_num)], dtype=tf.float32)
                zero_tensor = tf.zeros([1, self.maxlen1, int(config.embed_size/self.direction_num)], dtype=tf.float32)
                pad_tensor = tf.zeros([1, self.maxlen1, int(config.embed_size%self.direction_num)], dtype=tf.float32)

                one_item_list = []

                for i in range(self.direction_num):
                    ith_ = []
                    for j in range(self.direction_num):
                        if i==j:
                            ith_.append(one_tensor)
                        else:
                            ith_.append(zero_tensor)
                    if int(config.embed_size%self.direction_num) != 0:
                        ith_.append(pad_tensor)

                    one_item_list.append(tf.concat(ith_, 2))
                one_item = tf.concat(one_item_list, 0)
                one_batch = tf.concat([one_item for _ in range(config.batch_size)], 0)

                input_embedding = tf.add(input_embedding, one_batch)

            if config.sent_type == "bow":
                input_embedding, sent_size = get_bow(input_embedding)
                output_embedding, _ = get_bow(output_embedding)

            elif config.sent_type == "rnn":
                sent_cell = self.get_rnncell("gru", self.sent_cell_size, config.keep_prob, 1)
                input_embedding, sent_size = get_rnn_encode(input_embedding, sent_cell, scope="sent_rnn")
                output_embedding, _ = get_rnn_encode(output_embedding, sent_cell, output_lens,
                                                     scope="sent_rnn", reuse=True)
            elif config.sent_type == "bi_rnn":
                fwd_sent_cell = self.get_rnncell("gru", self.sent_cell_size, keep_prob=1.0, num_layer=1)
                bwd_sent_cell = self.get_rnncell("gru", self.sent_cell_size, keep_prob=1.0, num_layer=1)
                input_embedding, sent_size = get_bi_rnn_encode(input_embedding, fwd_sent_cell, bwd_sent_cell, scope="sent_bi_rnn")
                output_embedding, _ = get_bi_rnn_encode(output_embedding, fwd_sent_cell, bwd_sent_cell, output_lens, scope="sent_bi_rnn", reuse=True)
            else:
                raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")

            if config.sent_type != "bi_rnn":
                input_embedding = tf.reshape(input_embedding, [-1, max_dialog_len, config.sent_cell_size])
            else:
                input_embedding = tf.reshape(input_embedding, [-1, max_dialog_len, config.sent_cell_size*2])
            if config.keep_prob < 1.0:
                input_embedding = tf.nn.dropout(input_embedding, config.keep_prob)

#            # convert floors into 1 hot
#            floor_one_hot = tf.one_hot(tf.reshape(self.floors, [-1]), depth=2, dtype=tf.float32)
#            floor_one_hot = tf.reshape(floor_one_hot, [-1, max_dialog_len, 2])
#
#            joint_embedding = tf.concat([input_embedding, floor_one_hot], 2, "joint_embedding")

        with variable_scope.variable_scope("contextRNN"):
            enc_last_state = tf.reshape(input_embedding, [-1, config.sent_cell_size*2])
            '''
            enc_cell = self.get_rnncell(config.cell_type, self.context_cell_size, keep_prob=1.0, num_layer=config.num_layer)
            # and enc_last_state will be same as the true last state
            _, enc_last_state = tf.nn.dynamic_rnn(
                enc_cell,
                joint_embedding,
                dtype=tf.float32,
                sequence_length=self.context_lens)


            if config.num_layer > 1:
                if config.cell_type == 'lstm':
                    enc_last_state = [temp.h for temp in enc_last_state]

                enc_last_state = tf.concat(enc_last_state, 1)
            else:
                if config.cell_type == 'lstm':
                    enc_last_state = enc_last_state.h
            '''

        # combine with other attributes
#        if config.use_hcf:
#            attribute_embedding = da_embedding
#            attribute_fc1 = layers.fully_connected(attribute_embedding, 30, activation_fn=tf.tanh, scope="attribute_fc1")

#        cond_list = [topic_embedding, tf.expand_dims(self.my_profile, 1), tf.expand_dims(self.ot_profile, 1), enc_last_state]
#        cond_embedding = tf.concat(cond_list, 1)
        cond_embedding = enc_last_state

        with variable_scope.variable_scope("recognitionNetwork"):
#            if config.use_hcf:
#                recog_input = tf.concat([cond_embedding, output_embedding, attribute_fc1], 1)
#            else:
#                recog_input = tf.concat([cond_embedding, output_embedding], 1)
            recog_input = tf.concat([cond_embedding, output_embedding], 1)
            self.recog_mulogvar = recog_mulogvar = layers.fully_connected(recog_input, config.latent_size * 2, activation_fn=None, scope="muvar")
            recog_mu, recog_logvar = tf.split(recog_mulogvar, 2, axis=1)

        with variable_scope.variable_scope("priorNetwork"):
            # P(XYZ)=P(Z|X)P(X)P(Y|X,Z)
            prior_fc1 = layers.fully_connected(cond_embedding, int(max(config.latent_size * 2, 100)),
                                               activation_fn=tf.tanh, scope="fc1")
            prior_mulogvar = layers.fully_connected(prior_fc1, config.latent_size * 2, activation_fn=None,
                                                    scope="muvar")
            prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=1)

            # use sampled Z or posterior Z
            latent_sample = tf.cond(self.use_prior,
                                    lambda: sample_gaussian(prior_mu, prior_logvar),
                                    lambda: sample_gaussian(recog_mu, recog_logvar))

        with variable_scope.variable_scope("generationNetwork"):
            gen_inputs = tf.concat([cond_embedding, latent_sample], 1)

#            gen_inputs_copy = tf.concat([gen_inputs for _ in range(self.direction_num)], 0)
#
#            add_direction_size = config.topic_embed_size + 1 + 1 + config.sent_cell_size*2 + config.latent_size
#
#            one_tensor = tf.ones([config.batch_size,
#                                  int(add_direction_size/self.direction_num)], dtype=tf.float32)
#            zero_tensor = tf.zeros([config.batch_size,
#                                    int(add_direction_size/self.direction_num)], dtype=tf.float32)
#            pad_tensor = tf.zeros([config.batch_size,
#                                   int(add_direction_size%self.direction_num)], dtype=tf.float32)
#
#            one_batch_list = []
#
#            for i in range(self.direction_num):
#                ith_ = []
#                for j in range(self.direction_num):
#                    if i==j:
#                        ith_.append(one_tensor)
#                    else:
#                        ith_.append(zero_tensor)
#                if int(config.embed_size%self.direction_num) != 0:
#                    ith_.append(pad_tensor)
#
#                one_batch_list.append(tf.concat(ith_, 1))
#            one_batch = tf.concat(one_batch_list, 0)
#
#            gen_inputs_copy = tf.add(gen_inputs_copy, one_batch)
#            gen_inputs_copy = tf.add(one_batch, gen_inputs_copy)

            # Y loss
            if config.use_hcf:
                pass
#                meta_fc1 = layers.fully_connected(gen_inputs, 400, activation_fn=tf.tanh, scope="meta_fc1")
#                if config.keep_prob <1.0:
#                    meta_fc1 = tf.nn.dropout(meta_fc1, config.keep_prob)
#                self.da_logits = layers.fully_connected(meta_fc1, self.da_vocab_size, scope="da_project")
#                da_prob = tf.nn.softmax(self.da_logits)
#                pred_attribute_embedding = tf.matmul(da_prob, d_embedding)
#                if forward:
#                    selected_attribute_embedding = pred_attribute_embedding
#                else:
#                    selected_attribute_embedding = attribute_embedding
#                dec_inputs = tf.concat([gen_inputs, selected_attribute_embedding], 1)
            else:
                self.da_logits = tf.zeros((config.batch_size*self.direction_num, self.da_vocab_size))
                dec_inputs = gen_inputs
                selected_attribute_embedding = None

            # Decoder
            if config.num_layer > 1:
                dec_init_state = []
                for i in range(config.num_layer):
                    temp_init = layers.fully_connected(dec_inputs, self.dec_cell_size, activation_fn=None,
                                                        scope="init_state-%d" % i)
                    if config.cell_type == 'lstm':
                        temp_init = rnn_cell.LSTMStateTuple(temp_init, temp_init)

                    dec_init_state.append(temp_init)

                dec_init_state = tuple(dec_init_state)
            else:
                dec_init_state = layers.fully_connected(dec_inputs, self.dec_cell_size, activation_fn=None,
                                                        scope="init_state")
                if config.cell_type == 'lstm':
                    dec_init_state = rnn_cell.LSTMStateTuple(dec_init_state, dec_init_state)

        with variable_scope.variable_scope("decoder"):
            dec_cell = self.get_rnncell(config.cell_type, self.dec_cell_size, config.keep_prob, config.num_layer)
            dec_cell = OutputProjectionWrapper(dec_cell, self.vocab_size)

            if forward:
                loop_func = decoder_fn_lib.context_decoder_fn_inference(None, dec_init_state, embedding,
                                                                        start_of_sequence_id=self.go_id,
                                                                        end_of_sequence_id=self.eos_id,
                                                                        maximum_length=self.maxlen2,
                                                                        num_decoder_symbols=self.vocab_size,
                                                                        context_vector=selected_attribute_embedding)
                dec_input_embedding = None
                dec_seq_lens = None
            else:
                loop_func = decoder_fn_lib.context_decoder_fn_train(dec_init_state, selected_attribute_embedding)
                dec_input_embedding = embedding_ops.embedding_lookup(embedding, output_tokens)
                dec_input_embedding = dec_input_embedding[:, 0:-1, :]
                dec_seq_lens = output_lens - 1

                if config.keep_prob < 1.0:
                    dec_input_embedding = tf.nn.dropout(dec_input_embedding, config.keep_prob)

                # apply word dropping. Set dropped word to 0
                if config.dec_keep_prob < 1.0:
                    keep_mask = tf.less_equal(tf.random_uniform((batch_size, max_out_len-1), minval=0.0, maxval=1.0),
                                              config.dec_keep_prob)
                    keep_mask = tf.expand_dims(tf.to_float(keep_mask), 2)
                    dec_input_embedding = dec_input_embedding * keep_mask
                    dec_input_embedding = tf.reshape(dec_input_embedding, [-1, max_out_len-1, config.embed_size])

            dec_outs, _, final_context_state = dynamic_rnn_decoder(dec_cell, loop_func,
                                                                   inputs=dec_input_embedding,
                                                                   sequence_length=dec_seq_lens)
            if final_context_state is not None:
                final_context_state = final_context_state[:, 0:array_ops.shape(dec_outs)[1]]
                mask = tf.to_int32(tf.sign(tf.reduce_max(dec_outs, axis=2)))
                self.dec_out_words = tf.multiply(tf.reverse(final_context_state, axis=[1]), mask)
            else:
                self.dec_out_words = tf.argmax(dec_outs, 2)

            if config.if_multi_direction:
                test_out_emb = embedding_ops.embedding_lookup(embedding, self.dec_out_words)
                test_inp_emb = embedding_ops.embedding_lookup(embedding, tf.reshape(input_contexts, [-1]))
                test_inp_emb = tf.reshape(test_inp_emb, [-1, self.maxlen1, config.embed_size])

                if config.test_trick == "embedding":
                    test_out_emb = tf.reduce_mean(test_out_emb, 1)
                    test_inp_emb = tf.reduce_mean(test_inp_emb, 1)
                elif config.test_trick == "encoder":
                    if config.sent_type == "bow":
                        test_inp_emb, sent_size = get_bow(test_inp_emb)
                        test_out_emb, _ = get_bow(test_out_emb)

                    elif config.sent_type == "rnn":
                        test_inp_emb, sent_size = get_rnn_encode(test_inp_emb, sent_cell, scope="sent_rnn", reuse=True)
                        test_out_emb, _ = get_rnn_encode(test_out_emb, sent_cell, scope="sent_rnn", reuse=True)
                    elif config.sent_type == "bi_rnn":
                        test_inp_emb, sent_size = get_bi_rnn_encode(test_inp_emb, fwd_sent_cell, bwd_sent_cell, scope="sent_bi_rnn", reuse=True)
                        test_out_emb, _ = get_bi_rnn_encode(test_out_emb, fwd_sent_cell, bwd_sent_cell, scope="sent_bi_rnn", reuse=True)
                    else:
                        raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")
                else:
                    raise ValueError("Unknown test_trick. Must be one of [embedding, encoder]")

                sim_io = -1 * tf.reduce_sum(tf.multiply(test_out_emb, test_inp_emb), 1)

                self.test_samples_mask = tf.reduce_sum(build_train_mask(sim_io, config), 1)


        if not forward:
            with variable_scope.variable_scope("loss"):
                labels = output_tokens[:, 1:]

                label_mask = tf.to_float(tf.sign(labels))

                rc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_outs, labels=labels)

                rc_loss = tf.reduce_sum(rc_loss * label_mask, reduction_indices=1)

                # reconstruct the meta info about X
                if config.use_hcf:
                    da_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.da_logits, labels=self.output_das)
                    self.avg_da_loss = tf.reduce_mean(da_loss)
                else:
                    self.avg_da_loss = 0.0

                kld = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)

                if config.if_multi_direction:
                    all_elbo_loss = rc_loss + kld

                    loss_mask = build_train_mask(all_elbo_loss, config)
                    print(loss_mask)
                    self.sample_mask = tf.reduce_sum(loss_mask, 1)

                    forward_loss_mask = tf.reduce_sum(loss_mask, 1)
                    reverse_loss_mask = (1.0 - forward_loss_mask)/(self.direction_num - 1.0)

                    # scene loss
                    dec_outs_emb = tf.einsum('btv,ve->bte', tf.stop_gradient(dec_outs), embedding)
                    dec_outs_sent_emb, _ = get_bi_rnn_encode(dec_outs_emb, fwd_sent_cell, bwd_sent_cell, output_lens-1, scope='sent_bi_rnn', reuse=True)
                    dec_outs_sent_emb = tf.reshape(dec_outs_sent_emb, [-1, config.sent_cell_size*2])* tf.stop_gradient(loss_mask)
                    dec_outs_sent_emb_T = tf.transpose(dec_outs_sent_emb, [1,0])
                    dot_loss = tf.matmul(dec_outs_sent_emb, dec_outs_sent_emb_T)
                    # normalization
                    dot_loss = 5*(dot_loss - tf.reduce_min(dot_loss))/(tf.reduce_max(dot_loss)-tf.reduce_min(dot_loss))

                    dot_loss = tf.reshape(dot_loss, [config.batch_size*self.direction_num, config.batch_size, self.direction_num])

                    #full_loss_mask = reverse_loss_mask - forward_loss_mask
                    #full_loss_mask = expand_direction(
                    #        tf.reshape(full_loss_mask, [config.batch_size, self.direction_num]),
                    #        config.direction_num,[config.batch_size, 1],2)

                    positive_mask = expand_direction(
                            tf.reshape(forward_loss_mask, [config.batch_size, self.direction_num]),
                            config.direction_num,[config.batch_size, 1],2)
                    negative_mask = expand_direction(
                            tf.reshape(reverse_loss_mask, [config.batch_size, self.direction_num]),
                            config.direction_num,[config.batch_size, 1],2)

                    p_dot_loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(dot_loss, 1), tf.stop_gradient(positive_mask)), 1)
                    n_dot_loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(dot_loss, 1), tf.stop_gradient(negative_mask)), 1)
                    #dot_loss = tf.reduce_sum(tf.multiply(tf.reduce_sum(dot_loss, 1), tf.stop_gradient(full_loss_mask)), 1)

                    contrastive_loss = -1 * tf.log(tf.exp(p_dot_loss)/(tf.exp(p_dot_loss)+tf.exp(n_dot_loss)))

                    dir_labels = [dir_no%self.direction_num for dir_no in range(config.batch_size*self.direction_num)]
                    dir_labels_tensor = tf.constant(dir_labels, dtype = tf.int32)
                    self.da_logits = layers.fully_connected(dec_outs_sent_emb, self.direction_num, scope='dir_project')
                    dir_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.da_logits, labels=dir_labels_tensor)* tf.stop_gradient(forward_loss_mask)

                    self.avg_dir_loss = tf.reduce_sum(dir_loss)/config.batch_size
                    self.strength_loss = tf.reduce_sum(contrastive_loss)/config.batch_size
                    #self.strength_loss = tf.reduce_sum(dot_loss)/config.batch_size


                    self.avg_rc_loss = tf.reduce_sum(rc_loss*tf.stop_gradient(forward_loss_mask))/tf.reduce_sum(tf.stop_gradient(forward_loss_mask))
                    # used only for perpliexty calculation. Not used for optimzation
                    self.rc_ppl = tf.exp(tf.reduce_sum(rc_loss*tf.stop_gradient(forward_loss_mask)) / tf.reduce_sum(tf.reduce_sum(label_mask,1)*tf.stop_gradient(forward_loss_mask)))
    #                self.my_ppl = ppl_layer(dec_outs, labels,label_mask, self.output_lens)
                    self.avg_kld = tf.reduce_mean(kld*tf.stop_gradient(forward_loss_mask))

#                    self.reverse_avg_rc_loss = tf.reduce_mean(rc_loss)
#                    self.reverse_avg_bow_loss  = tf.reduce_mean(bow_loss*tf.stop_gradient(reverse_loss_mask))
#                    self.reverse_avg_kld = tf.reduce_mean(kld*tf.stop_gradient(reverse_loss_mask))
                else:
                    self.avg_rc_loss = tf.reduce_mean(rc_loss)
                    # used only for perpliexty calculation. Not used for optimzation
                    self.rc_ppl = tf.exp(tf.reduce_sum(rc_loss) / tf.reduce_sum(label_mask))
    #                self.my_ppl = ppl_layer(dec_outs, labels,label_mask, self.output_lens)
                    self.avg_kld = tf.reduce_mean(kld)

                if log_dir is not None:
                    kl_weights = tf.minimum(tf.to_float(self.global_t)/config.full_kl_step, 1.0)
                else:
                    kl_weights = tf.constant(1.0)

                self.kl_w = kl_weights
                
                self.elbo = self.avg_rc_loss + kl_weights*self.avg_kld
                
                if config.if_multi_direction:
                    #self.elbo = self.avg_rc_loss + kl_weights*self.avg_kld + 0.001*self.strength_loss + self.avg_dir_loss
                    aug_elbo = kl_weights*self.strength_loss + self.avg_dir_loss + self.elbo
                else:
                    aug_elbo = self.avg_da_loss + self.elbo

#                if config.if_multi_direction:
#                    reverse_elbo = self.reverse_avg_rc_loss
##                    reverse_elbo = self.reverse_avg_rc_loss + kl_weights * self.reverse_avg_kld
##                    reverse_aug_elbo = self.reverse_avg_bow_loss + reverse_elbo
#                    reverse_aug_elbo = reverse_elbo
#
#                    aug_elbo = aug_elbo - 0.01*reverse_aug_elbo


                tf.summary.scalar("strength_loss", self.strength_loss)
                tf.summary.scalar("avg_dir_loss", self.avg_dir_loss)
                tf.summary.scalar("rc_loss", self.avg_rc_loss)
                tf.summary.scalar("elbo", self.elbo)
                tf.summary.scalar("kld", self.avg_kld)

                self.summary_op = tf.summary.merge_all()

                self.log_p_z = norm_log_liklihood(latent_sample, prior_mu, prior_logvar)
                self.log_q_z_xy = norm_log_liklihood(latent_sample, recog_mu, recog_logvar)
                self.est_marginal = tf.reduce_mean(rc_loss - self.log_p_z + self.log_q_z_xy)

            self.optimize(sess, config, aug_elbo, log_dir)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

    def batch_2_feed(self, batch, global_t, use_prior, repeat=1):
        context, context_lens, floors, topics, my_profiles, ot_profiles, outputs, output_lens, output_das = batch
        feed_dict = {self.input_contexts: context, self.context_lens:context_lens,
                     self.floors: floors, self.topics:topics, self.my_profile: my_profiles,
                     self.ot_profile: ot_profiles, self.output_tokens: outputs,
                     self.output_das: output_das, self.output_lens: output_lens,
                     self.use_prior: use_prior}
        if repeat > 1:
            tiled_feed_dict = {}
            for key, val in feed_dict.items():
                if key is self.use_prior:
                    tiled_feed_dict[key] = val
                    continue
                multipliers = [1]*len(val.shape)
                multipliers[0] = repeat
                tiled_feed_dict[key] = np.tile(val, multipliers)
            feed_dict = tiled_feed_dict

        if global_t is not None:
            feed_dict[self.global_t] = global_t

        return feed_dict

    def train(self, global_t, sess, train_feed, update_limit=5000):
        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        kl_losses = []
        cl_losses = []
        dir_losses = []
        local_t = 0
        start_time = time.time()
        loss_names =  ["elbo_loss", "rc_loss", "rc_peplexity", "kl_loss", "cl_loss", "dir_loss"]
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break

            feed_dict = self.batch_2_feed(batch, global_t, use_prior=False)
            _, sum_op, elbo_loss, rc_loss, rc_ppl, kl_loss, cl_loss, dir_loss = sess.run([self.train_ops, self.summary_op,
                                                                         self.elbo,
                                                                         self.avg_rc_loss, self.rc_ppl, self.avg_kld,
                                                                         self.strength_loss, self.avg_dir_loss],
                                                                         feed_dict)
            self.train_summary_writer.add_summary(sum_op, global_t)
            elbo_losses.append(elbo_loss)
            rc_ppls.append(rc_ppl)
            rc_losses.append(rc_loss)
            kl_losses.append(kl_loss)
            cl_losses.append(cl_loss)
            dir_losses.append(dir_loss)

            global_t += 1
            local_t += 1
            if local_t % (train_feed.num_batch / 10) == 0:
                kl_w = sess.run(self.kl_w, {self.global_t: global_t})
                self.print_loss("%.2f" % (train_feed.ptr / float(train_feed.num_batch)),
                                loss_names, [elbo_losses, rc_losses, rc_ppls, kl_losses, cl_losses, dir_losses], "kl_w %f" % kl_w)

            if update_limit is not None and local_t >= update_limit:
                break

        # finish epoch!
        epoch_time = time.time() - start_time
        avg_losses = self.print_loss("Epoch Done", loss_names,
                                     [elbo_losses, rc_losses, rc_ppls, kl_losses, cl_losses, dir_losses],
                                     "step time %.4f" % (epoch_time / min(update_limit+1, train_feed.num_batch)))

        return global_t, avg_losses[0]

    '''
    def valid(self, name, sess, valid_feed):
        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        bow_losses = []
        kl_losses = []

        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=False, repeat=1)

            elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss = sess.run(
                [self.elbo, self.avg_bow_loss, self.avg_rc_loss,
                 self.rc_ppl, self.avg_kld], feed_dict)
            elbo_losses.append(elbo_loss)
            rc_losses.append(rc_loss)
            rc_ppls.append(rc_ppl)
            bow_losses.append(bow_loss)
            kl_losses.append(kl_loss)

        avg_losses = self.print_loss(name, ["elbo_loss", "bow_loss", "rc_loss", "rc_peplexity", "kl_loss"],
                                     [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses], "")
        return avg_losses[0]
    '''

    def valid(self, name, sess, if_multi_direction, valid_feed):
        ppls = []
        sample_masks = []

        while True:
            batch = valid_feed.next_batch()

            if batch is None:
                break

            context, _, _, _, _, _, outputs, _, _ = batch

            feed_dict = self.batch_2_feed(batch, None, use_prior=False, repeat=1)

            if if_multi_direction:
                rc_ppl, sample_mask = sess.run([self.rc_ppl, self.sample_mask], feed_dict)
                sample_masks.append(sample_mask)
            else:
                rc_ppl = sess.run(self.rc_ppl, feed_dict)

            ppls.append(rc_ppl)

#            mppls.append(my_ppl)


#        bleu_1 = get_bleu(write_to_file[1], write_to_file[2], 1)
#        bleu_2 = get_bleu(write_to_file[1], write_to_file[2], 2)
#        bleu_3 = get_bleu(write_to_file[1], write_to_file[2], 3)
#        bleu_4 = get_bleu(write_to_file[1], write_to_file[2], 4)
#
#        distinct_1 = get_distinct(write_to_file[2], 1)
#        distinct_2 = get_distinct(write_to_file[2], 2)
#        distinct_3 = get_distinct(write_to_file[2], 3)
#
#        metrics = [sum(mppls)/len(mppls), bleu_1, bleu_2, bleu_3, bleu_4, distinct_1, distinct_2, distinct_3]

        print('his ppl : {}'.format(sum(ppls)/len(ppls)))
#        print('my  ppl : {}'.format(sum(mppls) / len(mppls)))
        #return sum(mppls)/len(mppls), write_to_file
        if if_multi_direction:
            return sum(ppls)/len(ppls), sample_masks
        else:
            return sum(ppls)/len(ppls)


    def valid_for_sample(self, name, sess, if_multi_direction, sample_masks, valid_feed):
        write_to_file = [[],[],[],[]]

        batch_i_step = 0
        while True:
            batch = valid_feed.next_batch()

            if batch is None:
                break

            context, _, _, _, _, _, outputs, _, _ = batch

            feed_dict = self.batch_2_feed(batch, None, use_prior=True, repeat=1)

            word_outs = sess.run(self.dec_out_words, feed_dict)

            if if_multi_direction:
                sample_mask = sample_masks[batch_i_step]

            for b_id in range(valid_feed.batch_size):
                context_tokens = [self.vocab[e] for e in context[b_id][0] if e not in [self.go_id, self.eos_id, 0]]
                src_str = " ".join(context_tokens).replace("__", "")
                write_to_file[0].append(src_str)

                tgt_tokens = [self.vocab[e] for e in outputs[b_id] if e not in [self.go_id, self.eos_id, 0]]
                tgt_str = " ".join(tgt_tokens).replace("__", "")
                write_to_file[1].append(tgt_str)

                if if_multi_direction:
                    for dir_id in range(self.direction_num):
                        sample_id = dir_id + self.direction_num * b_id
                        if sample_mask[sample_id] > 0.0:
                            pred_tokens = [self.vocab[e] for e in word_outs[sample_id].tolist() if e not in [self.go_id, self.eos_id, 0]]
                            pred_str = " ".join(pred_tokens).replace("__", "")
                            write_to_file[2].append(pred_str)
                            write_to_file[3].append(dir_id)
                else:
                    pred_tokens = [self.vocab[e] for e in word_outs[b_id].tolist() if e not in [self.go_id, 0]]
                    pred_str = " ".join(pred_tokens).replace("__", "").split("</s>")[0].strip()
                    write_to_file[2].append(pred_str)

                if len(write_to_file[0]) != len(write_to_file[2]):
                    raise ValueError('there is an error')
            batch_i_step += 1

        assert len(write_to_file[0]) == len(write_to_file[2])

        # for idx in [0,5,10]:
        #     print('Source : {}'.format(write_to_file[0][idx]))
        #     print('Target : {}'.format(write_to_file[1][idx]))
        #     print('Gene   : {}'.format(write_to_file[2][idx]))
        #     print('\n')

        return write_to_file


    def test_for_sample(self, name, sess, if_multi_direction, test_feed):
        write_to_file = [[], [], [], []]

        batch_i_step = 0
        while True:
            batch = test_feed.next_batch()

            if batch is None:
                break

            context, _, _, _, _, _, outputs, _, _ = batch

            feed_dict = self.batch_2_feed(batch, None, use_prior=True, repeat=1)

            if if_multi_direction:
                word_outs, test_mask = sess.run([self.dec_out_words, self.test_samples_mask], feed_dict)
                #print(test_mask)
            else:
                word_outs = sess.run(self.dec_out_words, feed_dict)

            for b_id in range(test_feed.batch_size):
                context_tokens = [self.vocab[e] for e in context[b_id][0] if e not in [self.go_id, self.eos_id, 0]]
                src_str = " ".join(context_tokens).replace("__", "")

                tgt_tokens = [self.vocab[e] for e in outputs[b_id] if e not in [self.go_id, self.eos_id, 0]]
                tgt_str = " ".join(tgt_tokens).replace("__", "")

                if if_multi_direction:
                    for dir_id in range(self.direction_num):
                        sample_id = dir_id + self.direction_num * b_id
                        if test_mask[sample_id] >= 1:
                            pred_tokens = [self.vocab[e] for e in word_outs[sample_id].tolist() if
                                           e not in [self.go_id, self.eos_id, 0]]
                            pred_str = " ".join(pred_tokens).replace("__", "")
                            write_to_file[2].append(pred_str)
                            write_to_file[3].append(dir_id)
                            write_to_file[0].append(src_str)
                            write_to_file[1].append(tgt_str)
                else:
                    write_to_file[0].append(src_str)
                    write_to_file[1].append(tgt_str)

                    pred_tokens = [self.vocab[e] for e in word_outs[b_id].tolist() if e not in [self.go_id, 0]]
                    pred_str = " ".join(pred_tokens).replace("__", "").split("</s>")[0].strip()
                    write_to_file[2].append(pred_str)

                if len(write_to_file[0]) != len(write_to_file[2]):
                    raise ValueError('there is an error')
            batch_i_step += 1

        assert len(write_to_file[0]) == len(write_to_file[2])

        # for idx in [0, 5, 10]:
        #     print('Source : {}'.format(write_to_file[0][idx]))
        #     print('Target : {}'.format(write_to_file[1][idx]))
        #     print('Gene   : {}'.format(write_to_file[2][idx]))
        #     print('\n')

        return write_to_file


    def test(self, sess, test_feed, num_batch=None, repeat=1, dest=sys.stdout):
        local_t = 0
        recall_bleus = []
        prec_bleus = []
        results = []
        while True:
            batch = test_feed.next_batch()
            if batch is None or (num_batch is not None and local_t > num_batch):
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=True, repeat=repeat)
            word_outs, da_logits = sess.run([self.dec_out_words, self.da_logits], feed_dict)
            sample_words = np.split(word_outs, repeat, axis=0)
            true_outs = feed_dict[self.output_tokens]
            local_t += 1

            if dest != sys.stdout:
                if local_t % (test_feed.num_batch / 10) == 0:
                    print("%.2f >> " % (test_feed.ptr / float(test_feed.num_batch))),

            for b_id in range(test_feed.batch_size):
                # print the true outputs
                true_tokens = [self.vocab[e] for e in true_outs[b_id].tolist() if e not in [0, self.eos_id, self.go_id]]
                true_str = " ".join(true_tokens).replace(" ' ", "'")

                local_tokens = []
                pred_outs = sample_words[0]
                pred_tokens = [self.vocab[e] for e in pred_outs[b_id].tolist() if e != self.eos_id and e != 0]
                pred_str = " ".join(pred_tokens)
                local_tokens.append(pred_str)
                results.append({"pred":pred_str,"target":true_str})

        json.dump(results, dest, ensure_ascii=False,indent=4)
        print("Done testing")


