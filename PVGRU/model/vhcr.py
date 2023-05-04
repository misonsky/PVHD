#coding=utf-8
import tensorflow as tf
from tensorflow import keras
import numpy as np
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
class Encoder(keras.Model):
    def __init__(self,embedFunction,config,bidirectional=False):
        super(Encoder, self).__init__()
        self.embedding = embedFunction
        self.encoder = BASICEncoder(rnn_type=config.rnn_type,
                                    output_size=config.d_model,
                                    num_layers=config.encoder_layers,
                                    bidirectional=bidirectional)
    def call(self, x, mask,hidden=None,useEmbedding=True):
        if useEmbedding:
            x = self.embedding(x)
        output, state = self.encoder(x,mask=mask,initial_state = hidden)
        return output, state
class Decoder(keras.Model):
    def __init__(self, config,embedFunction,bidirectional=False):
        super(Decoder, self).__init__()
        self.embedding = embedFunction
        self.decoder = BASICEncoder(rnn_type=config.rnn_type,
                                    output_size=config.d_model,
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
        output, state = self.decoder(x,mask=None,initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))#(batch * 1) * hidden
    
        return output, state
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
class VariableLayer(keras.Model):
    def __init__(self, context_hidden, encoder_hidden, z_hidden,dytype):
        super(VariableLayer, self).__init__()
        self.context_hidden = context_hidden
        self.encoder_hidden = encoder_hidden
        self.z_hidden = z_hidden
        self.dytype = dytype
        self.conv_posterior_h = FFN()
        self.conv_posterior_mu = keras.layers.Dense(z_hidden)
        self.conv_posterior_var = keras.layers.Dense(z_hidden)
        
        self.prior_h = FFN()
        self.prior_mu = keras.layers.Dense(z_hidden)
        self.prior_var = keras.layers.Dense(z_hidden)
        self.posterior_h = FFN()
        self.posterior_mu = keras.layers.Dense(z_hidden)
        self.posterior_var = keras.layers.Dense(z_hidden)
        self.noraml_initializer = keras.initializers.random_normal(mean=0., stddev=1.)
    def conv_prior(self):
        return tf.zeros([1],dtype=self.dytype),tf.ones([1],dtype=self.dytype)
    def conv_epsion(self,batch_size,hidden_size):
        eps = self.noraml_initializer(shape=(batch_size, hidden_size))
        return  eps
    def conv_posterior(self,context_inference_hidden):
        h_posterior = self.conv_posterior_h(context_inference_hidden)
        mu_posterior = self.conv_posterior_mu(h_posterior)
        var_posterior = tf.nn.softplus(self.conv_posterior_var(h_posterior))
        return mu_posterior, var_posterior
    def prior(self, context_outputs,z_conv):
        # context_outputs: [batch, context_hidden]
        h_prior = self.prior_h(tf.concat([context_outputs,z_conv],axis=1),activation = tf.nn.tanh)
        mu_prior = self.prior_mu(h_prior)
        var_prior = tf.nn.softplus(self.prior_var(h_prior))
        return mu_prior, var_prior
    def posterior(self, context_outputs, encoder_hidden,z_conv):
        # context_outputs: [batch, context_hidden]
        # encoder_hidden: [batch, encoder_hidden]
        h_posterior = tf.concat([context_outputs, encoder_hidden,z_conv],axis=1)
        h_posterior = self.posterior_h(h_posterior,activation = tf.nn.tanh)
        mu_posterior = self.posterior_mu(h_posterior)
        var_posterior = tf.nn.softplus(self.posterior_var(h_posterior))
        return mu_posterior, var_posterior
    def normal_logpdf(self,x, mean, var):
        """
        Args:
            x: (Variable, FloatTensor) [batch_size, dim]
            mean: (Variable, FloatTensor) [batch_size, dim] or [batch_size] or [1]
            var: (Variable, FloatTensor) [batch_size, dim]: positive value
        Return:
            log_p: (Variable, FloatTensor) [batch_size]
        """
        log2pi = tf.math.log(2. * np.pi)
        return 0.5 * tf.reduce_sum(-log2pi -tf.math.log(var) - (tf.pow(x-mean,2.0) /var),axis=1)
    def kl_div(self, mu1, var1, mu2, var2):
        kl_div = -0.5 * tf.reduce_sum(1 + (var1 - var2)
                               - tf.math.divide(tf.pow(mu2 - mu1, 2), tf.exp(var2))
                               - tf.math.divide(tf.exp(var1), tf.exp(var2)), axis=1)
        return kl_div
    def sent_call(self,context_outputs, tag_context, z_conv,training=True):
        mu_prior, var_prior = self.prior(context_outputs,z_conv)
        eps = self.noraml_initializer(shape=(context_outputs.shape[0], self.z_hidden))
        if training:
            mu_posterior, var_posterior = self.posterior(context_outputs,tag_context,z_conv)
            z_sent = mu_posterior + tf.math.sqrt(var_posterior) * eps
            log_q_zx = tf.reduce_sum(self.normal_logpdf(z_sent, mu_posterior, var_posterior))
            log_p_z = tf.reduce_sum(self.normal_logpdf(z_sent, mu_prior, var_prior))
            kl_div = self.kl_div(mu_posterior, var_posterior, mu_prior, var_prior)
            # kl_div = tf.reduce_sum(kl_div)
        else:
            z_sent = mu_prior + tf.math.sqrt(var_prior) * eps
            kl_div = None
            log_p_z = tf.reduce_sum(self.normal_logpdf(z_sent, mu_prior, var_prior))
            log_q_zx = None
        return z_sent, kl_div,log_p_z, log_q_zx
    def conv_call(self,context_inference_hidden,training=True):
        conv_eps = self.conv_epsion(context_inference_hidden.shape[0],self.z_hidden)
        conv_mu_prior, conv_var_prior = self.conv_prior()
        if training:
            conv_mu_posterior, conv_var_posterior = self.conv_posterior(context_inference_hidden)
            z_conv = conv_mu_posterior + tf.math.sqrt(conv_var_posterior) * conv_eps
            log_q_zx_conv = tf.reduce_sum(self.normal_logpdf(z_conv, conv_mu_posterior, conv_var_posterior))
            log_p_z_conv = tf.reduce_sum(self.normal_logpdf(z_conv, conv_mu_prior, conv_var_prior))
            kl_div_conv = self.kl_div(conv_mu_posterior, conv_var_posterior,conv_mu_prior, conv_var_prior)
        else:
            z_conv = conv_mu_prior + tf.math.sqrt(conv_var_prior) * conv_eps
            log_q_zx_conv=None
            log_p_z_conv = tf.reduce_sum(self.normal_logpdf(z_conv, conv_mu_prior, conv_var_prior))
            kl_div_conv = None
        return z_conv,kl_div_conv,log_p_z_conv,log_q_zx_conv
    def call(self,context_outputs, tag_context=None,z_conv=None,training=True,vtype="conv"):
        # context_outputs: [batch, context_hidden]
        # Return: z_sent [batch, z_hidden]
        # Return: kl_div, scalar for calculating the loss
        if vtype=="conv":
            return self.conv_call(context_outputs,training=training)
        else:
            return  self.sent_call(context_outputs,tag_context,z_conv,training)
class VHCR(keras.Model):
    def __init__(self,vocab_size,embedding_dim,matrix,config,SOS=0,EOS=0,PAD=0):
        super(VHCR, self).__init__()
        self.dytype = tf.float16 if config.fp16 else tf.float32
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   embedding_dim,
                                                   embeddings_initializer=keras.initializers.constant(matrix),
                                                   trainable=True)
        self.encoder1 = Encoder(embedFunction=self.embedding,config=config,bidirectional=config.bidirectional)
        self.encoder2 = Encoder(embedFunction=self.embedding,config=config,bidirectional=config.bidirectional)
        self.context_inference = Encoder(embedFunction=self.embedding,config=config,bidirectional=config.bidirectional)
        self.decoder = Decoder(config=config,embedFunction=self.embedding,bidirectional=False)
        self.variablelayer = VariableLayer(config.d_model, config.d_model, config.d_model,self.dytype)
        self.z_conv2context = keras.layers.Dense(config.encoder_layers*config.d_model)
        self.context2decoder = keras.layers.Dense(config.decoder_layers*config.d_model,activation=tf.nn.tanh)
        self.outporj = keras.layers.Dense(embedding_dim)
        self.teach_force = config.teach_force
        self.config = config
        self.output_size = vocab_size
        self.SOS = SOS
        self.EOS = EOS
        self.PAD = PAD
        self.kl_mult = 0.0
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
        
        return tf.einsum("bd,vd->bv",self.outporj(logits),self.getEmbeddingTable())
        
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
        for utt in src:
            mask = tf.not_equal(utt,self.PAD)
            mask = tf.cast(mask,dtype=tf.zeros(1).dtype)
            context_mask.append(tf.reduce_sum(mask,axis=1))
            _,state = self.encoder1(utt,mask=mask)#(num * bid) * batch * d
            state = self.stackStates(state)# batch * d
            utterances.append(state)
        context_mask = tf.cast(tf.stack(context_mask,axis=1),dtype=tf.bool)
        context_mask = tf.cast(context_mask,dtype=tf.zeros(1).dtype)
        utterances = tf.stack(utterances, axis=1)#batch * max_turn * hidden
        _,context_inference_hidden = self.context_inference(utterances,mask=context_mask,useEmbedding=False)
        context_inference_hidden = self.stackStates(context_inference_hidden)# batch * d
        z_conv,kl_div_conv,log_p_z_conv,log_q_zx_conv = self.variablelayer(context_outputs=context_inference_hidden,
                                                                           tag_context=None,
                                                                           z_conv=None,
                                                                           training=True,
                                                                           vtype="conv")
        context_init = tf.reshape(self.z_conv2context(z_conv),shape=[self.config.encoder_layers*2,z_conv.shape[0],-1])
        context_init = tf.unstack(context_init,axis=0)
        z_conv_expand = tf.tile(tf.expand_dims(z_conv,axis=1),multiples=[1,utterances.shape[1],1])
        z_conv_utterance = tf.concat([utterances,z_conv_expand],axis=-1)
        
        context_outputs, enc_context = self.encoder2(z_conv_utterance,mask=context_mask,hidden=context_init,useEmbedding=False)
        _,tgt_context= self.encoder1(tgt,tf.cast(tf.not_equal(tgt,self.PAD),dtype=self.dytype))
        tgt_context = self.stackStates(tgt_context)# batch * d
        tgt_context = tf.stop_gradient(tgt_context, "tgt_encoder")
        context_expand = tf.concat([context_outputs,tf.expand_dims(tgt_context,axis=1)],axis=1)
        pre_context = tf.reshape(context_expand[:,:-1,:],shape=[context_outputs.shape[0]*context_outputs.shape[1],context_outputs.shape[-1]])
        suf_context = tf.reshape(context_expand[:,1:,:],shape=[context_outputs.shape[0]*context_outputs.shape[1],context_outputs.shape[-1]])
        flat_conv_z = tf.reshape(z_conv_expand,shape=[z_conv_expand.shape[0]*z_conv_expand.shape[1],z_conv_expand.shape[-1]])
        z_sent, kl_loss,log_p_z,log_q_z = self.variablelayer(context_outputs = pre_context, 
                                            tag_context=suf_context,
                                            z_conv=flat_conv_z,
                                            training=True,
                                            vtype="sent")
        z_sent = tf.reshape(z_sent,shape=context_outputs.shape)
        enc_context = self.context2decoder(tf.concat([self.stackStates(enc_context),z_sent[:,-1,:],z_conv],axis=-1))#batch * (hidden * number_lyaers)
        enc_context = tf.reshape(enc_context,shape=[self.config.decoder_layers,z_conv.shape[0],z_conv.shape[-1]])
        dec_hidden = tf.unstack(enc_context, axis=0)
        dec_input = tf.expand_dims(tgt[:,0],1)# batch * 1
        loss = 0
        for t in range(1, tgt.shape[1]):
            decoderOut, dec_hidden = self.decoder(x = dec_input, hidden = dec_hidden)
            logits = self.outputLayer(decoderOut)#batch * vocab
            dec_hidden = dec_hidden
            outputs.append(tf.nn.top_k(logits,k=1).indices)
            loss += self.loss_function(tgt[:, t], logits)
            dec_input = tf.expand_dims(tgt[:,t], 1)
        outputs = tf.stack(outputs,axis=1) #batch * seq * 1
        outputs = tf.reshape(outputs,shape=[outputs.shape[0],outputs.shape[1]])
        loss = loss / tf.reduce_sum(tf.cast(tf.not_equal(tgt[:,1:],self.PAD),dtype=loss.dtype))
        kl_loss = self.kl_mult * ((tf.reduce_mean(kl_loss) + tf.reduce_mean(kl_div_conv))/2.0)
        self.kl_mult = min(self.kl_mult + 1.0 / self.config.kl_annealing_iter, 1.0)
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
            _,state = self.encoder1(utt,mask=mask)#(num * bid) * batch * d
            state = self.stackStates(state)# batch * d
            utterances.append(state)
        context_mask = tf.cast(tf.stack(context_mask,axis=1),dtype=tf.bool)
        context_mask = tf.cast(context_mask,dtype=tf.zeros(1).dtype)
        utterances = tf.stack(utterances, axis=1)#batch * max_turn * hidden
        _,context_inference_hidden = self.context_inference(utterances,mask=context_mask,useEmbedding=False)
        context_inference_hidden = self.stackStates(context_inference_hidden)# batch * d
        z_conv,kl_div_conv,log_p_z_conv,log_q_zx_conv = self.variablelayer(context_outputs=context_inference_hidden,
                                                                           tag_context=None,
                                                                           z_conv=None,
                                                                           training=False,
                                                                           vtype="conv")
        context_init = tf.reshape(self.z_conv2context(z_conv),shape=[self.config.encoder_layers*2,z_conv.shape[0],-1])
        context_init = tf.unstack(context_init,axis=0)
        z_conv_expand = tf.tile(tf.expand_dims(z_conv,axis=1),multiples=[1,utterances.shape[1],1])
        z_conv_utterance = tf.concat([utterances,z_conv_expand],axis=-1)    
        context_outputs, enc_context = self.encoder2(z_conv_utterance,mask=context_mask,hidden=context_init,useEmbedding=False)
        pre_context = tf.reshape(context_outputs,shape=[context_outputs.shape[0]*context_outputs.shape[1],context_outputs.shape[-1]])
        flat_conv_z = tf.reshape(z_conv_expand,shape=[z_conv_expand.shape[0]*z_conv_expand.shape[1],z_conv_expand.shape[-1]])
        z_sent, kl_loss,log_p_z,log_q_z = self.variablelayer(context_outputs = pre_context, 
                                            tag_context=None,
                                            z_conv=flat_conv_z,
                                            training=False,
                                            vtype="sent")
        z_sent = tf.reshape(z_sent,shape=context_outputs.shape)
        enc_context = self.context2decoder(tf.concat([self.stackStates(enc_context),z_sent[:,-1,:],z_conv],axis=-1))#batch * (hidden * number_lyaers)
        enc_context = tf.reshape(enc_context,shape=[self.config.decoder_layers,z_conv.shape[0],z_conv.shape[-1]])
        dec_hidden = tf.unstack(enc_context, axis=0)
        startIdx = [self.SOS] * batchSize
        states = {"dec_hidden":dec_hidden}
        def symbols_to_logits_fn(tgtids, i, states):
            """
                tgtids:batch * seq
            """
            dec_input = tf.expand_dims(tgtids[:,i],axis=1)# batch * 1
            decoderOut, dec_hidden = self.decoder(x = dec_input, hidden = states["dec_hidden"])
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
                
            
            
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

