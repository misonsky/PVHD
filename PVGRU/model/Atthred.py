#coding=utf-8
import tensorflow as tf
from tensorflow import keras
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
    def __init__(self,embedFunction,output_size,config,bidirectional=False):
        super(Encoder, self).__init__()
        self.embedding = embedFunction
        self.encoder = BASICEncoder(rnn_type=config.rnn_type,
                                    output_size=output_size,
                                    num_layers=config.encoder_layers,
                                    bidirectional=bidirectional)
    def call(self, x, mask,hidden=None,useEmbedding=True):
        if useEmbedding:
            x = self.embedding(x)
        output, state = self.encoder(x,mask=mask,initial_state = hidden)
        return output, state
class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, query, values):
        """
        parameter:
            query:(batch * d)
            values:(batch * seq * d)
        return:
            context_vector:(batch *d)
        """
        #(batch * d)----->(batch * 1 * d)
        if isinstance(query,list):
            query = tf.reduce_mean(tf.stack(query,axis=0),axis=0)
        hidden_with_time_axis = tf.expand_dims(query, 1)#(batch * 1 * d)
        #score:(batch * seq * 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector:(batch * seq * d)
        context_vector = attention_weights * values
        # context_vector:(batch * d )
        context_vector = tf.reduce_sum(context_vector, axis=1)
    
        return context_vector
class Decoder(keras.Model):
    def __init__(self,embedFunction,output_size,config,bidirectional=False):
        super(Decoder, self).__init__()
        self.embedding = embedFunction
        self.decoder = BASICEncoder(rnn_type=config.rnn_type,
                                    output_size=output_size,
                                    num_layers=config.decoder_layers,
                                    bidirectional=bidirectional)
        # 用于注意力
        self.attention = BahdanauAttention(config.d_model)
    def call(self, x, hidden, enc_output):
        """
        parameter:
            x:input of decoder (batch *1 * hidden)
            hidden:previous state (batch * d)
            enc_output: output of encoder (batch * seq * d)
        """
        x = self.embedding(x)
        context_vector = self.attention(hidden, enc_output)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.decoder(x,mask=None,initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))#(batch * 1) * hidden
    
        return output, state
class AttHRED(keras.Model):
    def __init__(self,vocab_size,embedding_dim,matrix,config,SOS=0,EOS=0,PAD=0):
        super(AttHRED, self).__init__()
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
        enc_output, enc_hidden = self.encoder2(utterances,mask=context_mask,useEmbedding=False)
        dec_hidden = self.stackStates(enc_hidden)# batch * d
        dec_hidden = tf.tile(tf.expand_dims(dec_hidden,axis=0),multiples=[self.config.decoder_layers,1,1])
        dec_hidden = tf.unstack(dec_hidden,axis=0)
        dec_input = tf.expand_dims(tgt[:,0],1)# batch * 1
        loss = 0
        for t in range(1, tgt.shape[1]):
            decoderOut, dec_hidden = self.decoder(x = dec_input, 
                         hidden = dec_hidden, 
                         enc_output = enc_output)
            logits = self.outputLayer(decoderOut)#batch * vocab
            dec_hidden = dec_hidden
            outputs.append(tf.nn.top_k(logits,k=1).indices)
            loss += self.loss_function(tgt[:, t], logits)
            dec_input = tf.expand_dims(tgt[:,t], 1)
        outputs = tf.stack(outputs,axis=1) #batch * seq * 1
        outputs = tf.reshape(outputs,shape=[outputs.shape[0],outputs.shape[1]])
        loss = loss / tf.reduce_sum(tf.cast(tf.not_equal(tgt[:,1:],self.PAD),dtype=loss.dtype))
        return outputs,loss
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
            _,state = self.encoder1(utt,mask=mask)
            state = self.stackStates(state)
            utterances.append(state)
        context_mask = tf.cast(tf.stack(context_mask,axis=1),dtype=tf.bool)
        context_mask = tf.cast(context_mask,dtype=tf.zeros(1).dtype)
        utterances = tf.stack(utterances, axis=1)#batch * max_turn * hidden
        enc_output, enc_hidden = self.encoder2(utterances,mask=context_mask,useEmbedding=False)
        dec_hidden = self.stackStates(enc_hidden)
        dec_hidden = tf.tile(tf.expand_dims(dec_hidden,axis=0),multiples=[self.config.decoder_layers,1,1])
        dec_hidden = tf.unstack(dec_hidden,axis=0)
#         enc_output = tf.tile(enc_output,multiples=[self.config.beam_size,1,1])
        startIdx = [self.SOS] * batchSize
        states = {"dec_hidden":dec_hidden,"enc_output":enc_output}
        def symbols_to_logits_fn(tgtids, i, states):
            """
                tgtids:batch * seq
            """
            dec_input = tf.expand_dims(tgtids[:,i],axis=1)# batch * 1
            decoderOut, dec_hidden = self.decoder(x = dec_input, 
                        hidden = states["dec_hidden"], 
                        enc_output = states["enc_output"])
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

class AttSeq2Seq(keras.Model):
    def __init__(self,vocab_size,embedding_dim,matrix,config,SOS=0,EOS=0,PAD=0):
        super(AttSeq2Seq, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   embedding_dim,
                                                   embeddings_initializer=keras.initializers.constant(matrix),
                                                   trainable=True)
        self.encoder = Encoder(embedFunction=self.embedding,
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
        mask = tf.not_equal(src,self.PAD)
        mask = tf.cast(mask,dtype=tf.zeros(1).dtype)
        enc_output, enc_hidden = self.encoder(src,mask=mask)
        dec_hidden = self.stackStates(enc_hidden)#batch * hidden
        dec_hidden = tf.tile(tf.expand_dims(dec_hidden,axis=0),multiples=[self.config.decoder_layers,1,1])
        dec_hidden = tf.unstack(dec_hidden,axis=0)
        dec_input = tf.expand_dims(tgt[:,0],1)# batch * 1
        loss = 0
        for t in range(1, tgt.shape[1]):
            decoderOut, dec_hidden = self.decoder(x = dec_input, 
                         hidden = dec_hidden, 
                         enc_output = enc_output)
            logits = self.outputLayer(decoderOut)#batch * vocab
            dec_hidden = dec_hidden
            outputs.append(tf.nn.top_k(logits,k=1).indices)
            loss += self.loss_function(tgt[:, t], logits)
            dec_input = tf.expand_dims(tgt[:,t], 1)
        outputs = tf.stack(outputs,axis=1) #batch * seq * 1
        outputs = tf.reshape(outputs,shape=[outputs.shape[0],outputs.shape[1]])
        loss = loss / tf.reduce_sum(tf.cast(tf.not_equal(tgt[:,1:],self.PAD),dtype=loss.dtype))
        return outputs,loss
    def BeamDecoder(self,features,training=False):
        src = features["src"]
        batchSize = src.shape[0]
        mask = tf.not_equal(src,self.PAD)
        mask = tf.cast(mask,dtype=tf.zeros(1).dtype)
        enc_output, enc_hidden = self.encoder(src,mask=mask)
        dec_hidden = self.stackStates(enc_hidden)
        dec_hidden = tf.tile(tf.expand_dims(dec_hidden,axis=0),multiples=[self.config.decoder_layers,1,1])
        dec_hidden = tf.unstack(dec_hidden,axis=0)
        startIdx = [self.SOS] * batchSize
        states = {"dec_hidden":dec_hidden,"enc_output":enc_output}
        def symbols_to_logits_fn(tgtids, i, states):
            """
                tgtids:batch * seq
            """
            dec_input = tf.expand_dims(tgtids[:,i],axis=1)# batch * 1
            decoderOut, dec_hidden = self.decoder(x = dec_input, 
                        hidden = states["dec_hidden"], 
                        enc_output = states["enc_output"])
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
                
            
            
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

