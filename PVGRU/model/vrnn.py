#coding=utf-8
import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils.file_utils import get_bow
from collections import Counter
from utils.beam_search import beam_search
loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
class BASICEncoder(keras.Model):
    def __init__(self, rnn_type,output_size,num_layers=1,bidirectional=False):
        super(BASICEncoder, self).__init__()
        assert rnn_type in ['GRU','gru','LSTM','lstm']
        if bidirectional:
            assert output_size % 2 == 0
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        units = int(output_size / self.num_directions)
        if rnn_type == 'GRU' or rnn_type == 'gru':
            rnnCell = [getattr(keras.layers, 'GRUCell')(units) for _ in range(num_layers)]
        else:
            rnnCell = [getattr(keras.layers, 'LSTMCell')(units) for _ in range(num_layers)]
        self.rnn = keras.layers.RNN(rnnCell, 
                                    return_sequences=True, 
                                    return_state=True)
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        if bidirectional:
            self.rnn = keras.layers.Bidirectional(self.rnn)
        self.bidirectional = bidirectional
    def call(self, x, mask,initial_state=None):  # [batch, timesteps, input_dim]
        outputs=  self.rnn(x,
                           mask=mask,
                           initial_state = initial_state)
        output = outputs[0] #batch * seq * d
        states = outputs[1:] #(num *bidirec) * batch * d
        return output,states
class BASICVRNN(keras.Model):
    def __init__(self, config,output_size,num_layers,bidirectional=False):
        super(BASICVRNN, self).__init__()
        self.config = config
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn = BASICEncoder(rnn_type=config.rnn_type,
                                output_size=output_size,
                                num_layers=num_layers,
                                bidirectional=bidirectional)
        
        self.phi_x = FFN()
        self.phi_z = keras.layers.Dense(config.d_model,activation="relu")
        self.enc = FFN()
        self.enc_mean = keras.layers.Dense(config.d_model)
        self.enc_std = keras.layers.Dense(config.d_model,activation="softplus")
        self.prior = FFN()
        self.prior_mean = keras.layers.Dense(config.d_model)
        self.prior_std = keras.layers.Dense(config.d_model,activation="softplus")
        self.noraml_initializer = keras.initializers.random_normal(mean=0., stddev=1.)
    def kl_div(self, mu1, var1, mu2, var2):
        kl_div = -0.5 * tf.reduce_sum(1 + (var1 - var2)
                               - tf.math.divide(tf.pow(mu2 - mu1, 2), tf.exp(var2))
                               - tf.math.divide(tf.exp(var1), tf.exp(var2)), axis=1)
        return kl_div
    def TopStates(self,states):
        """
            return: batch *hidden
        """
        concat_states =[]#num_layer * batch * d
        if self.bidirectional:
            for fw,bw in zip(states[:self.config.encoder_layers],states[self.config.encoder_layers:]):
                concat_states.append(tf.concat([fw,bw],axis=-1))
        else:
            concat_states = states
        return concat_states[-1]
    def call(self,x,mask=None,initial_state=None):
        """
        patameters:
            x: batch *seq * hidden
            mask: batch * seq
        """
        x_shape = x.shape
        kld_loss = 0
        if initial_state is None:
            state_h = self.noraml_initializer(shape=(self.num_layers,x_shape[0],x_shape[-1]))
        else:
            state_h = initial_state
        eps = self.noraml_initializer(shape=(x_shape[0], x_shape[-1]))
        for _index,item in enumerate(tf.unstack(x,axis=1)):
            if not isinstance(state_h,list):
                if self.bidirectional:
                    state_h = tf.reshape(state_h,shape=[self.num_layers*2,state_h.shape[1],state_h.shape[-1] // 2])
                    state_h = tf.unstack(state_h,axis=0)
                else:
                    state_h = tf.unstack(state_h,axis=0)
            # item_step = tf.expand_dims(item,axis=1)#batch * 1 * hidden
            top_hidden = self.TopStates(state_h)
            phi_x_t = self.phi_x(item,activation=tf.nn.relu) #batch * hidden
            #encoder
            enc_t = self.enc(tf.concat([phi_x_t,top_hidden], axis=-1),activation=tf.nn.relu)
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)
            #prior
            prior_t = self.prior(top_hidden,activation=tf.nn.relu)
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            #sampling and reparameterization
            z_t = enc_mean_t + tf.math.sqrt(enc_std_t) * eps
            phi_z_t = self.phi_z(z_t)
            #recurrence
            item_step = tf.expand_dims(tf.concat([phi_x_t,phi_z_t],axis=-1),axis=1)
            if mask is not None:
                item_mask = tf.expand_dims(mask[:,_index],axis=1)
            else:
                item_mask = None
            #print("initial_h",state_h)
            outputs, state_h = self.rnn(item_step,mask=item_mask,initial_state=state_h)
            kld_loss += self.kl_div(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
        return outputs,state_h,kld_loss
class Encoder(keras.Model):
    def __init__(self,embedFunction,output_size,config,bidirectional=False):
        super(Encoder, self).__init__()
        self.embedding = embedFunction
        self.encoder = BASICVRNN(config=config,
                                 output_size = output_size,
                                num_layers=config.encoder_layers,
                                bidirectional=bidirectional)
    def call(self, x, mask,hidden=None,useEmbedding=True):
        if useEmbedding:
            x = self.embedding(x)
        _,state_h,kld_loss = self.encoder(x,mask=mask,initial_state = hidden)
        return state_h,kld_loss
class Decoder(keras.Model):
    def __init__(self, config,output_size,embedFunction,bidirectional=False):
        super(Decoder, self).__init__()
        self.embedding = embedFunction
        self.decoder = BASICVRNN(config=config,
                                 output_size = output_size,
                                num_layers=config.decoder_layers,
                                bidirectional=bidirectional)
    def call(self, x, hidden):
        """
        parameter:
            enc_output: output of encoder (batch * seq * d)
            x:input of decoder (batch *1 * hidden)
            hidden:previous state (batch * d)
        """
        x = self.embedding(x)
        output, state_h,kld_loss = self.decoder(x,mask=None,initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))#(batch * 1) * hidden
        return output,state_h,kld_loss
class FFN(keras.layers.Layer):
    def __init__(self):
        super(FFN, self).__init__()
    def build(self, input_shape):
        self.dense1 = keras.layers.Dense(input_shape[-1])
        self.dense2 = keras.layers.Dense(input_shape[-1])
    def call(self,x,activation=None):
        x = self.dense1(x)
        if activation is not None:
            x = activation(x)
        x = self.dense2(x)
        if activation is not None:
            x = activation(x)
        return x
class VRNN(keras.Model):
    def __init__(self,vocab_size,embedding_dim,matrix,config,SOS=0,EOS=0,PAD=0):
        super(VRNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   embedding_dim,
                                                   embeddings_initializer=keras.initializers.constant(matrix),
                                                   trainable=True)
        self.encoder1 = Encoder(embedFunction=self.embedding,
                                output_size = config.d_model,
                                config=config,
                                bidirectional=config.bidirectional)
        self.encoder2 = Encoder(embedFunction=self.embedding,
                                output_size = config.d_model,
                                config=config,
                                bidirectional=config.bidirectional)
        self.decoder = Decoder(config=config,
                               embedFunction=self.embedding,
                               output_size = config.d_model,
                               bidirectional=False)
        self.teach_force = config.teach_force
        self.config = config
        self.output_size = vocab_size
        self.SOS = SOS
        self.EOS = EOS
        self.PAD = PAD
    def getEmbeddingTable(self):
        if self.config.fp16:
            return tf.cast(self.embedding.embeddings,dtype=tf.float16)
        return self.embedding.embeddings
    def outputLayer(self,logits):
        """
        parameters:
            logits:batch * d
        return: batch  * vocabSize
        """
        return tf.einsum("bd,vd->bv",logits,self.getEmbeddingTable())
        
    def masked_fill(self,t, mask, value=-float('inf')):
        return t * (1 - tf.cast(mask, tf.float32)) + value * tf.cast(mask, tf.float32)
    def loss_function(self,real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, self.PAD))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)
    def stackStates(self,states,con_methods="sum"):
        """
            return: batch *hidden
        """
        concat_states =[]#num_layer * batch * d
        if self.config.bidirectional:
            for fw,bw in zip(states[:self.config.encoder_layers],states[self.config.encoder_layers:]):
                concat_states.append(tf.concat([fw,bw],axis=-1))
        else:
            concat_states = states
        if con_methods =="sum":
            concat_states = tf.reduce_sum(tf.stack(concat_states,axis=0),axis=0)
        elif con_methods =="mean":
            concat_states = tf.reduce_mean(tf.stack(concat_states,axis=0),axis=0)
        return concat_states
    @tf.function
    def call(self,features,training=True):
        """
        parameters:
            src: batch * (max_turn * max_seq)
            tgt: batch * max_seq 
        """
        outputs = []
        src = features["src"]
        tgt = features["tgt"]
        src = tf.reshape(src,shape=[src.shape[0],self.config.max_turn,self.config.max_utterance_len])
        src = tf.unstack(src,axis=1)
        utterances,context_mask= [],[]
        kl_loss = 0.0
        for utt in src:
            mask = tf.not_equal(utt,self.PAD)
            mask = tf.cast(mask,dtype=tf.zeros(1).dtype)
            context_mask.append(tf.reduce_sum(mask,axis=1))
            state,kld_loss = self.encoder1(utt,mask=mask)#(num * bid) * batch * d
            kl_loss += kld_loss
            state = self.stackStates(state) #batch * hidden
            utterances.append(state)
        context_mask = tf.cast(tf.stack(context_mask,axis=1),dtype=tf.bool)
        context_mask = tf.cast(context_mask,dtype=tf.zeros(1).dtype)
        utterances = tf.stack(utterances, axis=1)#batch * max_turn * hidden
        enc_hidden,kld_loss = self.encoder2(utterances,mask=context_mask,useEmbedding=False)
        kl_loss += kld_loss
        dec_hidden = self.stackStates(enc_hidden) #batch * hidden
        dec_hidden = tf.tile(tf.expand_dims(dec_hidden,axis=0),multiples=[self.config.decoder_layers,1,1])
        dec_hidden = tf.unstack(dec_hidden,axis=0)
        dec_input = tf.expand_dims(tgt[:,0],1)# batch * 1
        loss = 0
        for t in range(1, tgt.shape[1]):
            decoderOut,dec_hidden,kld_loss = self.decoder(x = dec_input, hidden = None)
            kl_loss += kld_loss
            logits = self.outputLayer(decoderOut)#batch * vocab
            dec_hidden = dec_hidden
            outputs.append(tf.nn.top_k(logits,k=1).indices)
            loss += self.loss_function(tgt[:, t], logits)
            dec_input = tf.expand_dims(tgt[:,t], 1)
        outputs = tf.stack(outputs,axis=1) #batch * seq * 1
        outputs = tf.reshape(outputs,shape=[outputs.shape[0],outputs.shape[1]])
        loss = loss / tf.reduce_sum(tf.cast(tf.not_equal(tgt[:,1:],self.PAD),dtype=loss.dtype))
        return outputs,[loss,kl_loss]
    def BeamDecoder(self,features,training=False):
        src = features["src"]
        batchSize = src.shape[0]
        src = tf.reshape(src,shape=[src.shape[0],self.config.max_turn,self.config.max_utterance_len])
        src = tf.unstack(src,axis=1)
        utterances,context_mask= [],[]
        for utt in src:
            mask = tf.not_equal(utt,self.PAD)
            mask = tf.cast(mask,dtype=tf.zeros(1).dtype)
            context_mask.append(tf.reduce_sum(mask,axis=1))
            state,_ = self.encoder1(utt,mask=mask)
            state = self.stackStates(state) #batch * hidden
            utterances.append(state)
        context_mask = tf.cast(tf.stack(context_mask,axis=1),dtype=tf.bool)
        context_mask = tf.cast(context_mask,dtype=tf.zeros(1).dtype)
        utterances = tf.stack(utterances, axis=1)#batch * max_turn * hidden
        enc_hidden,_ = self.encoder2(utterances,mask=context_mask,useEmbedding=False)
        dec_hidden = self.stackStates(enc_hidden)#batch * hidden
        dec_hidden = tf.tile(tf.expand_dims(dec_hidden,axis=0),multiples=[self.config.decoder_layers,1,1])
        dec_hidden = tf.unstack(dec_hidden,axis=0)
        startIdx = [self.SOS] * batchSize
        states = {"dec_hidden":dec_hidden}
        def symbols_to_logits_fn(tgtids, i, states):
            """
                tgtids:batch * seq
            """
            dec_input = tf.expand_dims(tgtids[:,i],axis=1)# batch * 1
            decoderOut, dec_hidden,_ = self.decoder(x = dec_input, 
                        hidden = states["dec_hidden"])
            logits = self.outputLayer(decoderOut)#batch * vocab
            states["dec_hidden"] = dec_hidden
            return logits, states
        ids, scores=beam_search(symbols_to_logits_fn=symbols_to_logits_fn,
                initial_ids=startIdx,
                beam_size=self.config.beam_size,
                decode_length=self.config.decode_length,
                vocab_size=self.output_size,
                alpha=self.config.alpha,
                states=states,
                eos_id=self.EOS,
                stop_early=True)
        
        return ids[:,0,:]
                
            
            
        
