import os
import time

import numpy as np
import tensorflow as tf
from beeprint import pp
import argparse
from corpus import SWDADialogCorpus
from data_utils import SWDADataLoader
from model import CVAE

# constants
parser = argparse.ArgumentParser('parameters config for sepaCVAE')
string_settings = parser.add_argument_group('string settings')
string_settings.add_argument('--data_dir',type=str,default="dataset",help='dataset path')
string_settings.add_argument('--gpu',type=str,default="0",help='which gpu device to use')
string_settings.add_argument('--corpus',type=str,default="DailyDialog",help='select task to train')
string_settings.add_argument('--embedding',type=str,default="embedding",help='dataset path')
string_settings.add_argument('--work_dir',type=str,default="working",help='Experiment results directory.')
string_settings.add_argument('--test_path',type=str,default="run1500783422",help='the dir to load checkpoint for forward only')
string_settings.add_argument('--logsdir',type=str,default="logs",help='the dir to logs')
string_settings.add_argument('--samples_dir',type=str,default="samples",help='the dir to samples')
string_settings.add_argument('--test_samples_dir',type=str,default="test_samples",help='the dir to test samples')
string_settings.add_argument('--description',type=str,default=None,help='description')
string_settings.add_argument('--test_trick',type=str,default="encoder",help='encoder')
string_settings.add_argument('--cell_type',type=str,default="gru",help='gru or lstm')
string_settings.add_argument('--op',type=str,default="adam",help='encoder')
string_settings.add_argument("--sent_type",default="bi_rnn",help="encoder")

boolean_settings = parser.add_argument_group('boolean settings')
boolean_settings.add_argument('--equal_batch',type=bool,default=True,help='Make each batch has similar length.')
boolean_settings.add_argument('--resume',type=bool,default=False,help='Resume from previous')
boolean_settings.add_argument('--forward_only',type=bool,default=False,help='Only do decoding')
boolean_settings.add_argument('--save_model',type=bool,default=False,help='Create checkpoints')
boolean_settings.add_argument('--use_hcf',type=bool,default=False,help='use dialog act in training')
boolean_settings.add_argument('--if_multi_direction',type=bool,default=True,help='sepacvae')
boolean_settings.add_argument('--bidirectional',type=bool,default=True,help="whether using bidirectional rnn")
boolean_settings.add_argument('--early_stop',type=bool,default=True,help='whether early stop')


scaler_settings = parser.add_argument_group('scaler settings')
scaler_settings.add_argument('--update_limit',type=int,default=100,help='the number of mini-batch before evaluating the model')
scaler_settings.add_argument('--direction_num',type=int,default=8,help='direction number')
scaler_settings.add_argument('--dot_loss_weight',type=float,default=50.0,help='loss weight')
scaler_settings.add_argument('--latent_size',type=int,default=200,help='the dimension of latent variable')
scaler_settings.add_argument('--full_kl_step',type=int,default=10000,help='how many batch before KL cost weight reaches 1.0')
scaler_settings.add_argument('--dec_keep_prob',type=float,default=1.0,help='do we use word drop decoder')
scaler_settings.add_argument('--embed_size',type=int,default=512,help='word embedding size')
scaler_settings.add_argument('--topic_embed_size',type=int,default=1,help='topic embedding size')
scaler_settings.add_argument('--da_embed_size',type=int,default=1,help='dialog act embedding size')
scaler_settings.add_argument('--cxt_cell_size',type=int,default=300,help='context encoder hidden size')
scaler_settings.add_argument('--sent_cell_size',type=int,default=300,help='utterance encoder hidden size')
scaler_settings.add_argument('--dec_cell_size',type=int,default=300,help='response decoder hidden size')
scaler_settings.add_argument('--backward_size',type=int,default=2,help='how many utterance kept in the context window')
scaler_settings.add_argument('--step_size',type=int,default=1,help='internal usage')
scaler_settings.add_argument('--max_utt_len',type=int,default=25,help='max number of words in an utterance')
scaler_settings.add_argument('--maxlen1',type=int,default=500,help='max len2')
scaler_settings.add_argument('--maxlen2',type=int,default=50,help='max len1')
scaler_settings.add_argument('--num_layer',type=int,default=2,help='number of context RNN layers')
scaler_settings.add_argument('--grad_clip',type=float,default=5.0,help='gradient abs max cut')
scaler_settings.add_argument('--Graphseed',type=int,default=12345,help='ramdom seed')
scaler_settings.add_argument('--init_w',type=float,default=0.08,help='uniform random from [-init_w, init_w]')
scaler_settings.add_argument('--batch_size',type=int,default=2,help='mini-batch size')
scaler_settings.add_argument('--init_lr',type=float,default=0.001,help='only used by SGD')
scaler_settings.add_argument('--lr_hold',type=int,default=1,help='only used by SGD')
scaler_settings.add_argument('--lr_decay',type=float,default=0.95,help='only used by SGD')
scaler_settings.add_argument('--keep_prob',type=float,default=1.0,help='drop out rate')
scaler_settings.add_argument('--improve_threshold',type=float,default=0.996,help='for early stopping')
scaler_settings.add_argument('--patient_increase',type=int,default=2.0,help='for early stopping')
scaler_settings.add_argument('--max_epoch',type=int,default=50,help='max number of epoch of training')
scaler_settings.add_argument('--grad_noise',type=float,default=0.0,help='inject gradient noise?')

def main(config):
    
    # config for training
    logdir = os.path.join(config.logsdir,config.corpus)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    smaples_dir = os.path.join(config.samples_dir,config.corpus)
    if not os.path.exists(smaples_dir):
        os.makedirs(smaples_dir)
    test_sample_dir = os.path.join(config.test_samples_dir,config.corpus)
    if not os.path.exists(test_sample_dir):
        os.makedirs(test_sample_dir)
        
    exp_time = len(os.listdir(logdir))
    try:
        os.mkdir(os.path.join(config.samples_dir, 'exp_time_{}'.format(exp_time)))
        os.mkdir(os.path.join(config.test_samples_dir, 'exp_time_{}'.format(exp_time)))
    except FileExistsError:
        pass

    # get data set
    word2vec_path = os.path.join(config.embedding,config.corpus,"vectors.txt")
    api = SWDADialogCorpus(config.corpus, 
                            max_context_len = config.maxlen1,
                            max_response_len = config.maxlen2,
                           word2vec=word2vec_path, 
                           word2vec_dim=config.embed_size)
    dial_corpus = api.get_dialog_corpus()
    meta_corpus = api.get_meta_corpus()

    train_meta, valid_meta, test_meta = meta_corpus.get("train"), meta_corpus.get("valid"), meta_corpus.get("test")
    train_dial, valid_dial, test_dial = dial_corpus.get("train"), dial_corpus.get("valid"), dial_corpus.get("test")

    # convert to numeric input outputs that fits into TF models
    train_feed = SWDADataLoader("Train", train_dial, train_meta, config)
    valid_feed = SWDADataLoader("Valid", valid_dial, valid_meta, config)
    test_feed = SWDADataLoader("Test", test_dial, test_meta, config)
    
    if config.forward_only or config.resume:
        work_dir = os.path.join(config.work_dir, config.corpus,config.test_path)
    else:
        work_dir = os.path.join(config.work_dir, config.corpus,"run"+str(int(time.time())))
        
#    word2idx = api.rev_vocab

    # begin training

    Graph_cvae = tf.get_default_graph()
    Graph_cvae.seed = config.Graphseed

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(graph = Graph_cvae, config = sess_config) as sess:
        initializer = tf.random_uniform_initializer(-1.0 * config.init_w, config.init_w)
        scope = "model"
        with tf.variable_scope(scope, reuse=None, initializer=initializer):
            model = CVAE(sess, config, api, log_dir=None if config.forward_only else work_dir, forward=False, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            valid_model = CVAE(sess, config, api, log_dir=None, forward=False, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            test_model = CVAE(sess, config, api, log_dir=None, forward=True, scope=scope)
        
#        scope = "model"
#        model = KgRnnCVAE(sess, config, api, log_dir=None if FLAGS.forward_only else log_dir, forward=False, scope=scope)
#        valid_model = KgRnnCVAE(sess, valid_config, api, log_dir=None, forward=False, scope=scope)
#        test_model = KgRnnCVAE(sess, test_config, api, log_dir=None, forward=True, scope=scope)
        
        #sess.run(tf.global_variables_initializer())

        #print("Created computation graphs")
        #if api.word2vec is not None and not FLAGS.forward_only:
        #    print("Loaded word2vec")
        #    sess.run(model.embedding.assign(np.array(api.word2vec)))

        # write config to a file for logging
        if not config.forward_only:
            with open(os.path.join(work_dir, "run.log"), "w") as f:
                f.write(pp(config, output=False))

        # create a folder by force
        ckp_dir = os.path.join(work_dir, "checkpoints")
        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)

        ckpt = tf.train.get_checkpoint_state(ckp_dir)
        print("Created models with fresh parameters.")
        sess.run(tf.global_variables_initializer())

        if ckpt:
            print("Reading dm models parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)

        if not config.forward_only:
            dm_checkpoint_path = os.path.join(ckp_dir, model.__class__.__name__+ ".ckpt")
            global_t = 0
            patience = 10  # wait for at least 10 epoch before stop
            dev_loss_threshold = np.inf
            best_dev_loss = np.inf
            epoch = 0
#            for epoch in range(config.max_epoch):
            flag_valid_batches = config.update_limit
            while(epoch != config.max_epoch):
                print(">> Epoch %d with lr %f" % (epoch, model.learning_rate.eval()))

                # begin training
                if train_feed.num_batch is None or train_feed.ptr >= train_feed.num_batch:
                    train_feed.epoch_init(config.batch_size, config.backward_size,
                                          config.step_size, shuffle=True)
                    epoch += 1
                    done_epoch = epoch
                    
                global_t, train_loss = model.train(global_t, sess, train_feed, update_limit=flag_valid_batches)
                
                if global_t % 100 != 0:
                    flag_valid_batches = config.update_limit - (global_t % 100)
                else:
                    flag_valid_batches = config.update_limit
                    
                    # begin validation
                    print('\n')
                    valid_feed.epoch_init(config.batch_size, config.backward_size,
                                          config.step_size, shuffle=False, intra_shuffle=False)
                    
                    start_valid_time = time.time()                  
                    if config.if_multi_direction:
                        ppl, sample_masks = valid_model.valid("ELBO_VALID", sess, True, valid_feed)
                    else:
                        ppl = valid_model.valid("ELBO_VALID", sess, False, valid_feed)
                    end_valid_time = time.time()

                    with open(os.path.join(config.logsdir, 'ppl_loss_{}.txt'.format(exp_time)), 'a', encoding='utf-8') as f:
                        f.write('{}\t{}\t{}\t{:.4f}\n'.format(
                            epoch,
                            global_t,
                            ppl,
                            end_valid_time-start_valid_time))
#                    print(len(sample_masks))
#                    print(len(sample_masks[0]))

                    valid_feed.epoch_init(config.batch_size, config.backward_size,
                                          config.step_size, shuffle=False, intra_shuffle=False)

                    test_feed.epoch_init(config.batch_size, config.backward_size,
                                         config.step_size, shuffle=False, intra_shuffle=False)
    
    
                    # only save a models if the dev loss is smaller
                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if config.op == "sgd" and done_epoch > config.lr_hold:
                        sess.run(model.learning_rate_decay_op)
    
                    if ppl < best_dev_loss:
                        with open(os.path.join(config.samples_dir, 'exp_time_{}'.format(exp_time),
                                               'samples_epoch_{:0>4d}_batches_{:0>6d}_ppl_{}_result'.format(epoch,
                                                                                                            global_t,
                                                                                                            ppl)), 'w',
                                  encoding='utf-8') as f:
                            if config.if_multi_direction:
                                valid_samples = test_model.valid_for_sample("ELBO_VALID", sess, True, sample_masks,
                                                                            valid_feed)
                                for true_sent, t2, re_sent, dir_id in zip(valid_samples[0], valid_samples[1],
                                                                          valid_samples[2], valid_samples[3]):
                                    f.write('True A, True B, (True C): \n')
                                    f.write(true_sent + '\n')
                                    f.write(t2 + '\n')
                                    f.write('Beam Search B: \n')
                                    f.write(re_sent + '\n')
                                    f.write(str(dir_id) + '\n\n')
                            else:
                                valid_samples = test_model.valid_for_sample("ELBO_VALID", sess, False, [],
                                                                            valid_feed)
                                for true_sent, t2, re_sent in zip(valid_samples[0], valid_samples[1], valid_samples[2]):
                                    f.write('True A, True B, (True C): \n')
                                    f.write(true_sent + '\n')
                                    f.write(t2 + '\n')
                                    f.write('Beam Search B: \n')
                                    f.write(re_sent + '\n\n')

                        with open(os.path.join(config.test_samples_dir, 'exp_time_{}'.format(exp_time),
                                               'samples_epoch_{:0>4d}_batches_{:0>6d}_ppl_{}_result'.format(epoch,
                                                                                                            global_t,
                                                                                                            ppl)), 'w',
                                  encoding='utf-8') as f:
                            if config.if_multi_direction:
                                test_samples = test_model.test_for_sample("TEST", sess, True, test_feed)
                                for true_sent, t2, re_sent, dir_id in zip(test_samples[0], test_samples[1],
                                                                          test_samples[2], test_samples[3]):
                                    f.write('True A, True B, (True C): \n')
                                    f.write(true_sent + '\n')
                                    f.write(t2 + '\n')
                                    f.write('Beam Search B: \n')
                                    f.write(re_sent + '\n')
                                    f.write(str(dir_id) + '\n\n')
                            else:
                                test_samples = test_model.test_for_sample("TEST", sess, False, test_feed)
                                for true_sent, t2, re_sent in zip(test_samples[0], test_samples[1], test_samples[2]):
                                    f.write('True A, True B, (True C): \n')
                                    f.write(true_sent + '\n')
                                    f.write(t2 + '\n')
                                    f.write('Beam Search B: \n')
                                    f.write(re_sent + '\n\n')
                        # still save the best train model
                        model.saver.save(sess, dm_checkpoint_path, global_step=epoch)
                        best_dev_loss = ppl
    
    #                if config.early_stop and patience <= done_epoch:
    #                    print("!!Early stop due to run out of patience!!")
    #                    break
            print("Best validation loss %f" % best_dev_loss)
            print("Done training")
        else:
            # begin validation
            # begin validation
            # valid_feed.epoch_init(config.batch_size, config.backward_size,config.step_size, shuffle=False, intra_shuffle=False)
            # valid_model.valid("ELBO_VALID", sess, valid_feed)

            # test_feed.epoch_init(config.batch_size, config.backward_size,
            #                       config.step_size, shuffle=False, intra_shuffle=False)
            # valid_model.valid("ELBO_TEST", sess, True,test_feed)
            
            dest_f = open(os.path.join(work_dir, "test.json"), "w",encoding="utf-8")
            test_feed.epoch_init(config.batch_size, config.backward_size,config.step_size, shuffle=False, intra_shuffle=False)
            test_model.test(sess, test_feed, num_batch=None, repeat=1, dest=dest_f)
            dest_f.close()

if __name__ == "__main__":
    config = parser.parse_args()
    if config.forward_only:
        if config.test_path is None:
            print("Set test_path before forward only")
            exit(1)
    main(config)













