class KgCVAEConfig(object):

    gpuSet='3'

    data_name = 'daily'

    logsdir = 'logs'
    samples_dir = 'samples'
    test_samples_dir = 'test_samples'

    description= None
    use_hcf = False  # use dialog act in training (if turn off kgCVAE -> CVAE)
    update_limit = 100  # the number of mini-batch before evaluating the model

    if_multi_direction = True

    direction_num = 8
    dot_loss_weight = 50.0

    test_trick = "encoder" # [embedding or encoder]

    # how to encode utterance.
    # bow: add word embedding together
    # rnn: RNN utterance encoder
    # bi_rnn: bi_directional RNN utterance encoder
    sent_type = "bi_rnn"

    # latent variable (gaussian variable)
    latent_size = 64  # the dimension of latent variable
    full_kl_step = 10000  # how many batch before KL cost weight reaches 1.0
    dec_keep_prob = 1.0  # do we use word drop decoder [Bowman el al 2015]

    # Network general
    cell_type = "gru"  # gru or lstm
    embed_size = 300 # word embedding size
    topic_embed_size = 1  # topic embedding size
    da_embed_size = 1  # dialog act embedding size
    cxt_cell_size = 300  # context encoder hidden size
    sent_cell_size = 300  # utterance encoder hidden size
    dec_cell_size = 300  # response decoder hidden size
    backward_size = 2  # how many utterance kept in the context window
    step_size = 1  # internal usage
    max_utt_len = 25  # max number of words in an utterance
    maxlen1 = 25
    maxlen2 = 25
    num_layer = 2  # number of context RNN layers

    # Optimization parameters
    op = "adam"
    grad_clip = 5.0  # gradient abs max cut
    Graphseed = 123456
    init_w = 0.08  # uniform random from [-init_w, init_w]
    batch_size = 64 # mini-batch size
    init_lr = 0.001  # initial learning rate
    lr_hold = 1  # only used by SGD
    lr_decay = 0.95  # only used by SGD
    keep_prob = 1.0  # drop out rate
    improve_threshold = 0.996  # for early stopping
    patient_increase = 2.0  # for early stopping
    early_stop = True
    max_epoch = 50  # max number of epoch of training
    grad_noise = 0.0  # inject gradient noise?







