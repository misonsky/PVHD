#coding=utf-8
import tensorflow as tf

import unicodedata
import re
from tqdm import tqdm
import collections
from collections import Counter
from glob import glob
from nltk import word_tokenize
import gensim
import numpy as np
import pickle
import io
import os
from tensorflow_datasets.core.deprecated.text.subword_text_encoder import SubwordTextEncoder

class DataSetBeam(object):
    def __init__(self,strategy):
        self.beam_dict={}
        self.strategy = strategy
        self.EOS = '<eos>'
        self.SOS = '<sos>'
        self.UNK = '<unk>'
        self.PAD = '<pad>'
        self.beam_dict["tokenize"] = self.construct_tokenize(strategy.config)
        self.generateEmbedding()
    def construct_tokenize(self,config):
        filePath = os.path.join(self.strategy.config.data_dir,self.strategy.config.corpus,"*")
        filesName = glob(filePath)
        words = []
        for special_token in [self.PAD,self.SOS,self.EOS,self.UNK]:
            words.append(special_token)
        for file in filesName:
            with open(file,encoding="utf-8") as f:
                for line in tqdm(f.readlines()):
                    line = line.rstrip()
                    sents = line.split("__eou__")
                    sents =[word_tokenize(sent) for sent in sents if len(sent.rstrip())>0]
                    for sent in sents:
                        for token in sent:
                            if token not in words:
                                words.append(token)
        
        return SubwordTextEncoder(words)
    def generateEmbedding(self):
        embeddingPath = os.path.join(self.strategy.config.embedding,self.strategy.config.corpus,"vectors.txt")
        dic ={}
        with open(embeddingPath,"r",encoding="utf-8") as f:
            for line in f:
                line  = line.rstrip()
                line = line.split()
                assert len(line) == self.strategy.config.emb_size +1
                dic[line[0]] = line[-self.strategy.config.emb_size:]
        embebPath = os.path.join(self.strategy.config.processDir,self.strategy.config.corpus,"embed.pkl")
        embedHandle = None
        if not os.path.exists(embebPath):
            embedHandle = open(embebPath,"wb")
        if embedHandle is not None:
            vectors = []
            for w_index in tqdm(range(self.vocabSize())):
                token = self.beam_dict["tokenize"].decode([w_index])
                if token in dic:
                    vector = dic[token]
                elif token.lower() in dic:
                    vector = dic[token.lower()]
                else:
                    vector = [0.0 for _ in range(self.strategy.config.emb_size)]
                vector = [float(item) for item in vector]
                vectors.append(vector)
            pickle.dump(vectors,embedHandle)
            tf.print("save embedding into {}".format(embebPath))
    def update_strategy(self,strategy):
        self.strategy = strategy
    def getSOSID(self):
        return self.beam_dict["tokenize"].encode(self.SOS)[0]
    def getEOSID(self):
        return self.beam_dict["tokenize"].encode(self.EOS)[0]
    def getUNKID(self):
        return self.beam_dict["tokenize"].encode(self.UNK)[0]
    def getPADID(self):
        return self.beam_dict["tokenize"].encode(self.PAD)[0]
    def vocabSize(self):
        return self.beam_dict["tokenize"].vocab_size
    def generate_graph(self,dialogs, path, fully=False, threshold=0.75, bidir=False, lang='en',self_loop=False):
        pass
    
    def unicode_to_ascii(self,s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    def create_dataset(self,fileName):
        refName = os.path.join(self.strategy.config.processDir,self.strategy.config.corpus,fileName.split(".")[0]+".ref")
        fileName = os.path.join(self.strategy.config.data_dir,self.strategy.config.corpus,fileName)
        examples=[]
        reference=[]
        with open(fileName,'r',encoding="utf-8") as f:
            for line in f:
                line=line.rstrip()
                utterances = line.split("__eou__")
                utterances = [utt for utt in utterances if len(utt)> 0]
                src = utterances[:-1]
                context,srcu,tgtu=[],[],None
                for _index,utt in enumerate(src,1):
                    if _index %2 == 0:
                        srcu.append(1)
                    else:
                        srcu.append(0)
                    tokens = word_tokenize(utt)
                    context.append(tokens)
                tgt = utterances[-1]
                tgtu = 1- srcu[-1]
                tokens = word_tokenize(tgt)
                if "train" not in fileName:
                    reference.append(" ".join(tokens))
                examples.append({"src":context,
                                 "srcu":srcu,
                                 "tgt":tokens,
                                 "tgtu":tgtu})
        if len(reference) >0:
            with open(refName,"w",encoding="utf-8") as f:
                for answer in reference:
                    f.write(answer.rstrip()+"\n")
        return examples
    def convert_id2text(self,idx):
        text = [self.beam_dict["tokenize"].decode([_id]) for _id in idx]
        return text
    def convet_text2id(self,text):
        idxs = [self.beam_dict["tokenize"].encode(token)[0] for token in text]
        return idxs
    def create_hier_example(self,examples):
        # <sos> context <eos>
        for example in tqdm(examples):
            for i in range(len(example["src"])):
                example["src"][i] = [self.SOS] + example["src"][i] +[self.EOS]
            example["tgt"] = [self.SOS] + example["tgt"] +[self.EOS]
        for example in tqdm(examples):
            if len(example["src"]) > self.strategy.config.max_turn:
                example["src"] = example["src"][-self.strategy.config.max_turn:]
                example["srcu"] = example["srcu"][-self.strategy.config.max_turn:]
            elif len(example["src"]) < self.strategy.config.max_turn:
                example["src"] = example["src"] + [[self.PAD]] * (self.strategy.config.max_turn-len(example["src"]))
                while len(example["srcu"]) < self.strategy.config.max_turn:
                    example["srcu"] = example["srcu"] +[1-example["srcu"][-1]]
        for example in tqdm(examples):
            for i in range(len(example["src"])):
                if self.strategy.config.graphModel:
                    example["src"][i] = self.withRolePad(example["src"][i], self.strategy.config.max_utterance_len)
                else:
                    example["src"][i] = self.withoutRolePad(example["src"][i], self.strategy.config.max_utterance_len)
            if self.strategy.config.graphModel:
                example["tgt"] = self.withRolePad(example["tgt"], self.strategy.config.max_utterance_len)
            else:
                example["tgt"] = self.withoutRolePad(example["tgt"], self.strategy.config.max_utterance_len)
        #convert2toknes
        for example in examples:
            if self.strategy.config.hier:
                example["src"] = [self.convet_text2id(src) for src in example["src"]]
            else:
                example["src"] = self.convet_text2id(example["src"])
            example["tgt"] = self.convet_text2id(example["tgt"])
        return examples
    def create_concate_example(self,examples):
        for example in tqdm(examples):
            example["src"] = sum(example["src"],[])
        for example in tqdm(examples):
            example["src"] = [self.SOS] + example["src"] +[self.EOS]
            example["tgt"] = [self.SOS] + example["tgt"] +[self.EOS]
        for example in tqdm(examples):
            if len(example["srcu"]) > self.strategy.config.max_turn:
                example["srcu"] = example["srcu"][-self.strategy.config.max_turn:]
            while len(example["srcu"]) < self.strategy.config.max_turn:
                example["srcu"] = example["srcu"] +[1-example["srcu"][-1]]
        for example in tqdm(examples):
            context_length = self.strategy.config.max_utterance_len * self.strategy.config.max_turn
            if self.strategy.config.graphModel:
                example["src"] = self.withRolePad(example["src"], context_length)
            else:
                example["src"] = self.withoutRolePad(example["src"], context_length)
            if self.strategy.config.graphModel:
                example["tgt"] = self.withRolePad(example["tgt"], self.strategy.config.max_utterance_len)
            else:
                example["tgt"] = self.withoutRolePad(example["tgt"], self.strategy.config.max_utterance_len)
        #convert2toknes
        for example in examples:
            example["src"] = self.convet_text2id(example["src"])
            example["tgt"] = self.convet_text2id(example["tgt"])
        return examples
    def compute_bow(self,examples):
        #only using for target
        for example in examples:
            tgt = example["tgt"]
            countDict = Counter(tgt)
            vec = [countDict.get(token,0) for token in tgt]
            example["tgt_bow"] = vec
        return examples
    def padd_bow(self,examples,max_len):
        for example in examples:
            if len(example["tgt_bow"]) < max_len:
                example["tgt_bow"] = example["tgt_bow"] + [0] *(max_len - len(example["tgt_bow"]))
            else:
                example["tgt_bow"] = example["tgt_bow"][:max_len]
        return examples
    def addRole(self,examples):
        for example in examples:
            assert len(example["srcu"]) == len(example["src"])
            for i in range(len(example["srcu"])):
                example["src"][i] =  ["user"+str(example["srcu"][i])] + example["src"][i]
            example["tgt"] = ["user"+str(example["tgtu"])] + example["tgt"]
        return examples
    def addRoleFloor(self,examples):
        for example in examples:
            floor = [int(example["tgtu"]==role) for role in example["srcu"]]
            example["floor"] = floor
        return examples
    def withRolePad(self,tokenList,padLength):
        if len(tokenList) < padLength:
            tokenList = tokenList + [self.PAD] * (padLength-len(tokenList))
        elif len(tokenList) > padLength:
            tokenList = [self.SOS,tokenList[1]] + tokenList[-padLength+2:]
        return tokenList
    def withoutRolePad(self,tokenList,padLength):
        if len(tokenList) < padLength:
            tokenList = tokenList + [self.PAD] * (padLength-len(tokenList))
        elif len(tokenList) > padLength:
            tokenList = [self.SOS] + tokenList[-padLength+1:]
        return tokenList
    def write_example_to_tfrecord(self,fileName,outputFile):
        def create_int_feature(values):
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return feature
        def create_float_feature(values):
            feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return feature
        outputFile = os.path.join(self.strategy.config.processDir,self.strategy.config.corpus,outputFile)
        writer = tf.io.TFRecordWriter(outputFile)
        examples = self.create_dataset(fileName)
        #add pow
        examples = self.compute_bow(examples)
        examples = self.padd_bow(examples, max_len = self.strategy.config.max_utterance_len)
        #add user role
        if self.strategy.config.graphModel:
            examples = self.addRole(examples)
        if self.strategy.config.hier:
            examples = self.create_hier_example(examples)
        else:
            examples = self.create_concate_example(examples)
        #add floor features
        examples = self.addRoleFloor(examples)
        total_example =0
        for example in tqdm(examples):
            if self.strategy.config.hier:
                src = sum(example["src"],[])
            else:
                src = example["src"]
            srcu = example["srcu"]
            tgt = example["tgt"]
            tgtu = example["tgtu"]
            tbow = example["tgt_bow"]
            floor = example["floor"]
            assert len(src) == self.strategy.config.max_utterance_len * self.strategy.config.max_turn
            assert len(srcu) == self.strategy.config.max_turn
            assert len(tgt) == self.strategy.config.max_utterance_len
            assert len(tbow) == self.strategy.config.max_utterance_len
            assert len(floor) == self.strategy.config.max_turn
            assert tgtu in [0,1]
            features = collections.OrderedDict()
            features["src"] = create_int_feature(src)
            features["tgt"] = create_int_feature(tgt)
            features["srcu"] = create_int_feature(srcu)
            features["tgtu"] = create_int_feature([tgtu])
            features["tbow"] = create_int_feature(tbow)
            features["floor"] = create_int_feature(floor)
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            tf_example = tf_example.SerializeToString()
            writer.write(tf_example)
            total_example +=1
        writer.close()
        tf.print("save the tfrecord file into {}".format(outputFile))
        return total_example
    def get_batchDataset(self,inputFile,batch_size,is_training=False):
        name_to_features = {
            "src":tf.io.FixedLenFeature([self.strategy.config.max_utterance_len * self.strategy.config.max_turn], tf.int64),
            "tgt":tf.io.FixedLenFeature([self.strategy.config.max_utterance_len], tf.int64),
            "tbow":tf.io.FixedLenFeature([self.strategy.config.max_utterance_len], tf.int64),
            "floor":tf.io.FixedLenFeature([self.strategy.config.max_turn], tf.int64),
            "srcu":tf.io.FixedLenFeature([self.strategy.config.max_turn], tf.int64),
            "tgtu":tf.io.FixedLenFeature([],tf.int64)}
        def _parse_function(example):
            example= tf.io.parse_single_example(example,name_to_features)
            for name in list(example.keys()):
                t=example[name]
                if t.dtype==tf.int64:
                    t=tf.cast(t,tf.int32)
                example[name]=t
            return example
        d = tf.data.TFRecordDataset(inputFile)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=1024)
        parse_data=d.map(_parse_function,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        parse_data = parse_data.prefetch(tf.data.experimental.AUTOTUNE).batch(batch_size,drop_remainder=self.strategy.config.drop_last)
        return self.strategy.strategy.experimental_distribute_dataset(parse_data)
        
        
        
        
        
        
        
        
        
        