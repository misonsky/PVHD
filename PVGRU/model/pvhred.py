#coding=utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.util import nest
from utils.beam_search import beam_search
from utils.VarLSTMCelll import VALSTMCell,VAGRUCell
from utils.VarLSTMCelll import MyRNN
from utils.VarLSTMCelll import MyBidirectional
from utils.beam_search import beam_search
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
        keras.layers.LSTMCell
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
class VEncoder(keras.Model):
    def __init__(self,embedFunction,output_size,config,bidirectional=False):
        super(VEncoder, self).__init__()
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
class VDecoder(keras.Model):
    def __init__(self,embedFunction,output_size,config,bidirectional=False):
        super(VDecoder, self).__init__()
        self.embedding = embedFunction
        self.decoder = VariationEncoder(rnn_type=config.rnn_type,
                                    output_size=output_size,
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
        return output, state
class PVHRED(keras.Model):
    def __init__(self,vocab_size,embedding_dim,matrix,config,SOS=0,EOS=0,PAD=0):
        super(PVHRED, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   embedding_dim,
                                                   embeddings_initializer=keras.initializers.constant(matrix),
                                                   trainable=True)
        self.vencoder1 = VEncoder(embedFunction=self.embedding,
                                output_size = config.d_model,
                                config=config,
                                bidirectional=config.bidirectional)
        self.vencoder2 = VEncoder(embedFunction=self.embedding,
                                output_size = config.d_model,
                                config=config,
                                bidirectional=config.bidirectional)
        self.vdecoder = VDecoder(config=config,
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
    # @tf.function
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
            voutputs,kl_loss = self.vencoder1(utt,mask=mask)#(num * bid) * batch * d
            kl_scores.append(tf.reduce_mean(kl_loss))
            states,_ = self.separate_latent_state(voutputs[1:],bidirectional=self.config.bidirectional)
            state = self.stackStates(states) #batch * hidden
            utterances.append(state)
        context_mask = tf.cast(tf.stack(context_mask,axis=1),dtype=tf.bool)
        context_mask = tf.cast(context_mask,dtype=tf.zeros(1).dtype)
        utterances = tf.stack(utterances, axis=1)#batch * max_turn * hidden
        voutputs, kl_loss = self.vencoder2(utterances,mask=context_mask,useEmbedding=False)
        kl_scores.append(tf.reduce_mean(kl_loss))
        states,latents = self.separate_latent_state(voutputs[1:], bidirectional=self.config.bidirectional)
        states = self.stackStates(states)
        latents = self.stackStates(latents)
        dec_hidden=[[states,latents] for _ in range(self.config.decoder_layers)]
        dec_input = tf.expand_dims(tgt[:,0],1)# batch * 1
        loss = 0
        decoder_kl = None
        for t in range(1, tgt.shape[1]):
            dec_outputs, dec_kl = self.vdecoder(x = dec_input, hidden = dec_hidden)
            if decoder_kl is None:
                decoder_kl = dec_kl
            else:
                decoder_kl = tf.concat([decoder_kl,dec_kl],axis=-1)
            states,latents = self.separate_latent_state(dec_outputs[1:], bidirectional=False)
            dec_hidden=[[state,latent] for state,latent in zip(tf.unstack(states,axis=0),tf.unstack(latents,axis=0))]
            dec_outputs = dec_outputs[0]
            dec_outputs = tf.reshape(dec_outputs,shape=[dec_outputs.shape[0],dec_outputs.shape[-1]])
            logits = self.outputLayer(dec_outputs)#batch * vocab
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
            voutputs,_ = self.vencoder1(utt,mask=mask)#(num * bid) * batch * d
            states,_ = self.separate_latent_state(voutputs[1:],bidirectional=self.config.bidirectional)
            state = self.stackStates(states) #batch * hidden
            utterances.append(state)
        context_mask = tf.cast(tf.stack(context_mask,axis=1),dtype=tf.bool)
        context_mask = tf.cast(context_mask,dtype=tf.zeros(1).dtype)
        utterances = tf.stack(utterances, axis=1)#batch * max_turn * hidden
        voutputs, _ = self.vencoder2(utterances,mask=context_mask,useEmbedding=False)
        states,latents = self.separate_latent_state(voutputs[1:], bidirectional=self.config.bidirectional)
        states = self.stackStates(states)
        latents = self.stackStates(latents)
        dec_hidden=[[states,latents] for _ in range(self.config.decoder_layers)]
        startIdx = [self.SOS] * batchSize
        states = {"dec_hidden":dec_hidden}
        def symbols_to_logits_fn(tgtids, i, states):
            """
                tgtids:batch * seq
            """
            dec_input = tf.expand_dims(tgtids[:,i],axis=1)# batch * 1
            dec_outputs, _ = self.vdecoder(x = dec_input, hidden = states["dec_hidden"])
            vstates,vlatents = self.separate_latent_state(dec_outputs[1:], bidirectional=False)
            dec_hidden=[[state,latent] for state,latent in zip(tf.unstack(vstates,axis=0),tf.unstack(vlatents,axis=0))]
            states["dec_hidden"] = dec_hidden
            dec_outputs = dec_outputs[0]
            dec_outputs = tf.reshape(dec_outputs,shape=[dec_outputs.shape[0],dec_outputs.shape[-1]])
            logits = self.outputLayer(dec_outputs)#batch * vocab
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

class PVSeq2Seq(keras.Model):
    def __init__(self,vocab_size,embedding_dim,matrix,config,SOS=0,EOS=0,PAD=0):
        super(PVSeq2Seq, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   embedding_dim,
                                                   embeddings_initializer=keras.initializers.constant(matrix),
                                                   trainable=True)
        self.vencoder = VEncoder(embedFunction=self.embedding,
                                output_size = config.d_model,
                                config=config,
                                bidirectional=config.bidirectional)
        self.vdecoder = VDecoder(config=config,
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
        mask = tf.not_equal(src,self.PAD)
        mask = tf.cast(mask,dtype=tf.zeros(1).dtype)
        voutputs,kl_loss = self.vencoder(src,mask=mask)
        kl_scores.append(tf.reduce_mean(kl_loss))
        states,latents = self.separate_latent_state(voutputs[1:],bidirectional=self.config.bidirectional)
        states = self.stackStates(states)
        latents = self.stackStates(latents)
        dec_hidden=[[states,latents] for _ in range(self.config.decoder_layers)]
        dec_input = tf.expand_dims(tgt[:,0],1)# batch * 1
        loss = 0
        decoder_kl = None
        for t in range(1, tgt.shape[1]):
            dec_outputs, dec_kl = self.vdecoder(x = dec_input, hidden = dec_hidden)
            if decoder_kl is None:
                decoder_kl = dec_kl
            else:
                decoder_kl = tf.concat([decoder_kl,dec_kl],axis=-1)
            states,latents = self.separate_latent_state(dec_outputs[1:], bidirectional=False)
            dec_hidden=[[state,latent] for state,latent in zip(tf.unstack(states,axis=0),tf.unstack(latents,axis=0))]
            dec_outputs = dec_outputs[0]
            dec_outputs = tf.reshape(dec_outputs,shape=[dec_outputs.shape[0],dec_outputs.shape[-1]])
            logits = self.outputLayer(dec_outputs)#batch * vocab
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
        mask = tf.not_equal(src,self.PAD)
        mask = tf.cast(mask,dtype=tf.zeros(1).dtype)
        voutputs,_ = self.vencoder(src,mask=mask)
        states,latents = self.separate_latent_state(voutputs[1:],bidirectional=self.config.bidirectional)
        states = self.stackStates(states)
        latents = self.stackStates(latents)
        dec_hidden=[[states,latents] for _ in range(self.config.decoder_layers)]
        startIdx = [self.SOS] * batchSize
        states = {"dec_hidden":dec_hidden}
        def symbols_to_logits_fn(tgtids, i, states):
            """
                tgtids:batch * seq
            """
            dec_input = tf.expand_dims(tgtids[:,i],axis=1)# batch * 1
            dec_outputs, _ = self.vdecoder(x = dec_input, hidden = states["dec_hidden"])
            vstates,vlatents = self.separate_latent_state(dec_outputs[1:], bidirectional=False)
            dec_hidden=[[state,latent] for state,latent in zip(tf.unstack(vstates,axis=0),tf.unstack(vlatents,axis=0))]
            states["dec_hidden"] = dec_hidden
            dec_outputs = dec_outputs[0]
            dec_outputs = tf.reshape(dec_outputs,shape=[dec_outputs.shape[0],dec_outputs.shape[-1]])
            logits = self.outputLayer(dec_outputs)#batch * vocab
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
                
            
            
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

