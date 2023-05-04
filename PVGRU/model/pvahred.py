#coding=utf-8
import tensorflow as tf
from tensorflow import keras
from utils.beam_search import beam_search
from utils.VarLSTMCelll import VALSTMCell,VAGRUCell
from utils.VarLSTMCelll import MyRNN
from utils.VarLSTMCelll import MyBidirectional
loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
kl_object = keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
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

class VariationEncoder(keras.Model):
    def __init__(self, rnn_type,output_size,num_layers=1,bidirectional=False):
        super(VariationEncoder, self).__init__()
        # assert rnn_type in ['GRU','gru','LSTM','lstm',"vlstm","vgru"]
        if bidirectional:
            assert output_size % 2 == 0
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        units = int(output_size / self.num_directions)
        rnnCell = [VAGRUCell(units) for _ in range(num_layers)]
        self.rnn = MyRNN(rnnCell, 
                        return_sequences=True, 
                        return_state=True)
        self.cells = self.rnn.cell.cells
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        if bidirectional:
            self.rnn = MyBidirectional(self.rnn)
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
        self.encoder = VariationEncoder(rnn_type=config.rnn_type,
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
        self.decoder = VariationEncoder(rnn_type=config.rnn_type,
                                    output_size=output_size,
                                    num_layers=config.decoder_layers,
                                    bidirectional=bidirectional)
        # 用于注意力
        self.attention = BahdanauAttention(config.d_model)
    def call(self, x, hidden, enc_output):
        """
        parameter:
            x:input of decoder (batch *1 * hidden)
            hidden:previous state (layers * batch * d)
            enc_output: output of encoder (batch * seq * d)
        """
        x = self.embedding(x)
        context_vector = self.attention(hidden[-1], enc_output)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        dec_outputs, dec_kl = self.decoder(x,mask=None,initial_state=hidden)
        # output = tf.reshape(output, (-1, output.shape[2]))#(batch * 1) * hidden
        return dec_outputs, dec_kl
class PVAttHRED(keras.Model):
    def __init__(self,vocab_size,embedding_dim,matrix,config,SOS=0,EOS=0,PAD=0):
        super(PVAttHRED, self).__init__()
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
    def separate_latent_state(self,inputs,bidirectional=True):
        def latent_state(sequence):
            states,latent=[],[]
            for pair_element in sequence:
                states.append(pair_element[0])
                latent.append(pair_element[1])
            return states,latent
        num = len(inputs)
        if bidirectional:
            forward_elements = inputs[:num//2]
            backward_elements = inputs[num//2:]
            f_states,f_latents = latent_state(forward_elements)
            b_states,b_latents = latent_state(backward_elements)
            states = tf.concat([tf.stack(f_states,axis=0),tf.stack(b_states,axis=0)],axis=-1)
            latents = tf.concat([tf.stack(f_latents,axis=0),tf.stack(b_latents,axis=0)],axis=-1)
            return states,latents
        else:
            states,latents  = latent_state(inputs)
            return tf.stack(states,axis=0),tf.stack(latents,axis=0)
    def stackStates(self,states,con_methods="sum"):
        """
            return: batch *hidden
        """
        if con_methods =="sum":
            concat_states = tf.reduce_sum(states,axis=0)
        elif con_methods =="mean":
            concat_states = tf.reduce_mean(states,axis=0)
        return concat_states
    @tf.function
    def call(self,features,training=True):
        """
        parameters:
            src: batch * (max_turn * max_seq)
            tgt: batch * max_seq 
        """
        outputs = []
        kl_scores = []
        src = features["src"]
        tgt = features["tgt"]
        src = tf.reshape(src,shape=[src.shape[0],self.config.max_turn,self.config.max_utterance_len])
        src = tf.unstack(src,axis=1)
        utterances,context_mask= [],[]
        for utt in src:
            mask = tf.not_equal(utt,self.PAD)
            mask = tf.cast(mask,dtype=tf.zeros(1).dtype)
            context_mask.append(tf.reduce_sum(mask,axis=1))
            voutputs,kl_loss = self.encoder1(utt,mask=mask)#(num * bid) * batch * d
            kl_scores.append(tf.reduce_mean(kl_loss))
            states,_ = self.separate_latent_state(voutputs[1:],bidirectional=self.config.bidirectional)
            states = self.stackStates(states)# batch * d
            utterances.append(states)
        context_mask = tf.cast(tf.stack(context_mask,axis=1),dtype=tf.bool)
        context_mask = tf.cast(context_mask,dtype=tf.zeros(1).dtype)
        utterances = tf.stack(utterances, axis=1)#batch * max_turn * hidden
        voutputs, kl_loss = self.encoder2(utterances,mask=context_mask,useEmbedding=False)
        kl_scores.append(tf.reduce_mean(kl_loss))
        enc_output = voutputs[0]
        states,latents = self.separate_latent_state(voutputs[1:], bidirectional=self.config.bidirectional)
        states = self.stackStates(states)
        latents = self.stackStates(latents)
        dec_hidden=[[states,latents] for _ in range(self.config.decoder_layers)]
        dec_input = tf.expand_dims(tgt[:,0],1)# batch * 1
        loss = 0
        decoder_kl = None
        for t in range(1, tgt.shape[1]):
            dec_outputs, dec_kl = self.decoder(x = dec_input, 
                         hidden = dec_hidden, 
                         enc_output = enc_output)
            step_outputs = dec_outputs[0]
            step_outputs = tf.reshape(step_outputs, (step_outputs.shape[0], step_outputs.shape[2]))
            logits = self.outputLayer(step_outputs)#batch * vocab
            if decoder_kl is None:
                decoder_kl = dec_kl
            else:
                decoder_kl = tf.concat([decoder_kl,dec_kl],axis=-1)
            states,latents = self.separate_latent_state(dec_outputs[1:], bidirectional=False)
            dec_hidden=[[state,latent] for state,latent in zip(tf.unstack(states,axis=0),tf.unstack(latents,axis=0))]
            outputs.append(tf.nn.top_k(logits,k=1).indices)
            loss += self.loss_function(tgt[:, t], logits)
            dec_input = tf.expand_dims(tgt[:,t], 1)
        outputs = tf.stack(outputs,axis=1) #batch * seq * 1
        outputs = tf.reshape(outputs,shape=[outputs.shape[0],outputs.shape[1]])
        loss = loss / tf.reduce_sum(tf.cast(tf.not_equal(tgt[:,1:],self.PAD),dtype=loss.dtype))
        kl_scores.append(tf.reduce_mean(decoder_kl))
        return outputs,[loss,tf.reduce_mean(kl_scores)]
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
            voutputs,_ = self.encoder1(utt,mask=mask)
            states,_ = self.separate_latent_state(voutputs[1:],bidirectional=self.config.bidirectional)
            states = self.stackStates(states) #batch * d
            utterances.append(states)
        context_mask = tf.cast(tf.stack(context_mask,axis=1),dtype=tf.bool)
        context_mask = tf.cast(context_mask,dtype=tf.zeros(1).dtype)
        utterances = tf.stack(utterances, axis=1)#batch * max_turn * hidden
        voutputs, _ = self.encoder2(utterances,mask=context_mask,useEmbedding=False)
        enc_output = voutputs[0]
        states,latents = self.separate_latent_state(voutputs[1:], bidirectional=self.config.bidirectional)
        states = self.stackStates(states)
        latents = self.stackStates(latents)
        dec_hidden=[[states,latents] for _ in range(self.config.decoder_layers)]
        startIdx = [self.SOS] * batchSize
        states = {"dec_hidden":dec_hidden,"enc_output":enc_output}
        def symbols_to_logits_fn(tgtids, i, states):
            """
                tgtids:batch * seq
            """
            dec_input = tf.expand_dims(tgtids[:,i],axis=1)# batch * 1
            dec_outputs, _ = self.decoder(x = dec_input, 
                        hidden = states["dec_hidden"], 
                        enc_output = states["enc_output"])
            step_outputs = dec_outputs[0]
            step_outputs = tf.reshape(step_outputs, (step_outputs.shape[0], step_outputs.shape[2]))
            logits = self.outputLayer(step_outputs)#batch * vocab
            states_step,latents_step = self.separate_latent_state(dec_outputs[1:], bidirectional=False)
            dec_hidden=[[state,latent] for state,latent in zip(tf.unstack(states_step,axis=0),tf.unstack(latents_step,axis=0))]
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
