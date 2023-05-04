#coding=utf-8
import pickle as pkl
from collections import Counter
import numpy as np
import nltk
from nltk import word_tokenize
import os
        

class SWDADialogCorpus(object):
    dialog_act_id = 0
    sentiment_id = 1
    liwc_id = 2

    def __init__(self, dataname, max_context_len=500,max_response_len=50, word2vec=None, word2vec_dim=None):
        """
        :param corpus_path: the folder that contains the SWDA dialog corpus
        """
        self._dataname = dataname
        self.word_vec_path = word2vec
        self.word2vec_dim = word2vec_dim
        self.word2vec = None
        self.dialog_id = 0
        self.meta_id = 1
        self.utt_id = 2
        self.sil_utt = ["<s>", "<pad>", "</s>"]
        self.max_sent_len1 = max_context_len
        self.max_sent_len2 = max_response_len
        #data = pkl.load(open(self._path, "rb"))
        self.train_corpus = self.process('train')
        self.valid_corpus = self.process("valid")
        self.test_corpus = self.process("test")
        #self.build_vocab(max_vocab_cnt)
        self.load_vocab()
        self.load_word2vec()
        #self.max_sent_len = max_sent_len
        print("Done loading corpus")


    def process(self, mode):
        """new_dialog: [(a, 1/0), (a,1/0)], new_meta: (a, b, topic), new_utt: [[a,b,c)"""
        """ 1 is own utt and 0 is other's utt"""
        data = []
        if self._dataname == 'DailyDialog':
            if mode == 'train':
                fpath1 = 'dataset/DailyDialog/train.source.tok'
                fpath2 = 'dataset/DailyDialog/train.target.tok'
            elif mode == 'valid':
                fpath1 = 'dataset/DailyDialog/valid.source.tok'
                fpath2 = 'dataset/DailyDialog/valid.target.tok'
            elif mode == 'test':
                fpath1 = 'dataset/DailyDialog/test.source.tok'
                fpath2 = 'dataset/DailyDialog/test.target.tok'
        elif self._dataname == 'DSTC7_AVSD':
            if mode == 'train':
                fpath1 = 'dataset/DSTC7_AVSD/train.source.tok'
                fpath2 = 'dataset/DSTC7_AVSD/train.target.tok'
            elif mode == 'valid':
                fpath1 = 'dataset/DSTC7_AVSD/valid.source.tok'
                fpath2 = 'dataset/DSTC7_AVSD/valid.target.tok'
            elif mode == 'test':
                fpath1 = 'dataset/DSTC7_AVSD/test.source.tok'
                fpath2 = 'dataset/DSTC7_AVSD/test.target.tok'

        with open(fpath1, 'r', encoding='utf-8') as f1, open(fpath2, 'r', encoding='utf-8') as f2:
            for sent1, sent2 in zip(f1,f2):
                sent1 = sent1.rstrip()
                sent2 = sent2.rstrip()
                history = " ".join(sent1.split("__eou__"))
                if len(history.split())+2 > self.max_sent_len1 or len(sent2.split())+1>self.max_sent_len2:
                    continue
                data.append([history, sent2])
        new_dialog = []
        new_meta = []
        new_utts = []
        bod_utt = ["<s>", "<pad>", "</s>"]
        all_lenes = []

        for l in data:
            #lower_utts = [(caller, ["<s>"] + nltk.WordPunctTokenizer().tokenize(utt.lower()) + ["</s>"], feat)
            #              for caller, utt, feat in l["utts"]]
            lower_utts = [(0, ['<s>'] + l[0].split() + ['</s>'], 0)]
            #all_lenes.extend([len(u) for c, u, f in lower_utts])
            all_lenes.extend([len(u) for c, u, f in lower_utts])

            #a_age = float(l["A"]["age"])/100.0
            #b_age = float(l["B"]["age"])/100.0
            #a_edu = float(l["A"]["education"])/3.0
            #b_edu = float(l["B"]["education"])/3.0
            #vec_a_meta = [a_age, a_edu] + ([0, 1] if l["A"]["sex"] == "FEMALE" else [1, 0])
            #vec_b_meta = [b_age, b_edu] + ([0, 1] if l["B"]["sex"] == "FEMALE" else [1, 0])

            # for joint model we mode two side of speakers together. if A then its 0 other wise 1
            #meta = (vec_a_meta, vec_b_meta, l["topic"])
            meta = (0, 0, 0)
            #dialog = [(bod_utt, 0, None)] + [(utt, int(caller=="B"), feat) for caller, utt, feat in lower_utts]
            dialog = [(l[0].split(), 0, None), 
                      (['<s>'] + l[1].split() + ['</s>'], 1, None)]

            new_utts.extend([bod_utt] + [utt for caller, utt, feat in lower_utts])
            new_dialog.append(dialog)
            new_meta.append(meta)

        print("Max utt len %d, mean utt len %.2f" % (np.max(all_lenes), float(np.mean(all_lenes))))
        return new_dialog, new_meta, new_utts


    def load_vocab(self):
        if self._dataname == "DailyDialog":
            vocab_path = 'dataset/DailyDialog/vocab.txt'
        elif self._dataname == "DSTC7_AVSD":
            vocab_path = 'dataset/DSTC7_AVSD/vocab.txt'
        vocab = []
        for token in self.sil_utt:
            vocab.append(token)
        with open(vocab_path,"r",encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                vocab.append(line)
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        self.id2token = {idx: token for idx, token in enumerate(vocab)}
        self.vocab = vocab
        self.unk_id = self.token2idx['<unk>']
        self.topic_vocab = self.rev_topic_vocab = self.dialog_act_vocab = self.rev_dialog_act_vocab = None


    def build_vocab(self, max_vocab_cnt):
        all_words = []
        for tokens in self.train_corpus[self.utt_id]:
            all_words.extend(tokens)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus), len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1], float(discard_wc) / len(all_words)))

        self.vocab = ["<pad>", "<unk>"] + [t for t, cnt in vocab_count]
        self.token2idx = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.token2idx["<unk>"]
        print("<d> index %d" % self.token2idx["<d>"])
        print("<sil> index %d" % self.token2idx.get("<sil>", -1))

        # create topic vocab
        all_topics = []
        for a, b, topic in self.train_corpus[self.meta_id]:
            all_topics.append(topic)
        self.topic_vocab = [t for t, cnt in Counter(all_topics).most_common()]
        self.rev_topic_vocab = {t: idx for idx, t in enumerate(self.topic_vocab)}
        print("%d topics in train data" % len(self.topic_vocab))

        # get dialog act labels
        all_dialog_acts = []
        for dialog in self.train_corpus[self.dialog_id]:
            all_dialog_acts.extend([feat[self.dialog_act_id] for caller, utt, feat in dialog if feat is not None])
        self.dialog_act_vocab = [t for t, cnt in Counter(all_dialog_acts).most_common()]
        self.rev_dialog_act_vocab = {t: idx for idx, t in enumerate(self.dialog_act_vocab)}
        print(self.dialog_act_vocab)
        print("%d dialog acts in train data" % len(self.dialog_act_vocab))


    def load_word2vec(self):
        self.word2vec = []
        if self.word_vec_path is None:
            return
        emb = dict()
        with open(self.word_vec_path, "r", encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                items = line.split()
                token = items[0]
                vec = items[-self.word2vec_dim:]
                vec = [float(x) for x in vec]
                emb[token] = vec
        for token in self.vocab:
            if token in emb:
                self.word2vec.append(emb[token])
            else:
                self.word2vec.append([0.0 for _ in range(self.word2vec_dim)])

    def get_utt_corpus(self):
        def _to_id_corpus(data):
            results = []
            for line in data:
                results.append([self.token2idx.get(t, self.unk_id) for t in line])
            return results
        # convert the corpus into ID
        id_train = _to_id_corpus(self.train_corpus[self.utt_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.utt_id])
        id_test = _to_id_corpus(self.test_corpus[self.utt_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}


    def get_dialog_corpus(self):
        def _to_id_corpus(data):
            results = []
            for dialog in data:
                temp = []
                # convert utterance and feature into numeric numbers
                for utt, floor, feat in dialog:
                    if feat is not None:
                        id_feat = list(feat)
                        id_feat[self.dialog_act_id] = self.rev_dialog_act_vocab[feat[self.dialog_act_id]]
                    else:
                        id_feat = None
                    temp.append(([self.token2idx.get(t, self.unk_id) for t in utt], floor, id_feat))
                results.append(temp)
            return results
        id_train = _to_id_corpus(self.train_corpus[self.dialog_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.dialog_id])
        id_test = _to_id_corpus(self.test_corpus[self.dialog_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}


    def get_meta_corpus(self):
        def _to_id_corpus(data):
            results = []
            for m_meta, o_meta, topic in data:
                #results.append((m_meta, o_meta, self.rev_topic_vocab[topic]))
                results.append((0, 0, 0))
            return results

        id_train = _to_id_corpus(self.train_corpus[self.meta_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.meta_id])
        id_test = _to_id_corpus(self.test_corpus[self.meta_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

