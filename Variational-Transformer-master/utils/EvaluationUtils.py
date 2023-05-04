#coding=utf-8
from typing import List
from collections import OrderedDict
import numpy as np
from copy import deepcopy
import math
import os
import json
from torchtext.vocab import GloVe
from tqdm import tqdm
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder,TrigramCollocationFinder,QuadgramCollocationFinder
from utils.bleu_metric.bleu import Bleu
from utils.rouge_metric.rouge import Rouge
def map_tokens2vectors(tokens,dic,dim):
    """
    parameters:
        tokens:list of token
        dic:embedding dict
        dim: integer
    """
    vectors = []
    if len(tokens) == 0:
        tokens=["[unk]"]
    for w in tokens:
        try:
            vector = dic[w]
            vector = [float(item) for item in vector]
            vector = np.array(vector)
        except KeyError:
            vector = np.random.randn(dim)
        vectors.append(vector)
    return np.stack(vectors)
def cal_greedy_matching_matrix(vec_x:List[float], vec_y:List[float],dim=300):
    # vec_x = np.array(vec_x)
    # vec_y = np.array(vec_y)
    matrix = np.dot(vec_x, vec_y.T)
    matrix = matrix / np.linalg.norm(vec_x, axis=1, keepdims=True)  # [x, 1]
    matrix = matrix / np.linalg.norm(vec_y, axis=1).reshape(1, -1)  # [1, y]
    x_matrix_max = np.mean(np.max(matrix, axis=1)) # [x]
    y_matrix_max = np.mean(np.max(matrix, axis=0)) # [y]
    return (x_matrix_max + y_matrix_max) / 2

def cal_embedding_average(x:List[float], y:List[float],dim=512):
    """
    parameters:
        x:list of tokens
        y: list of tokens
        dic: embedding dict
        dim: embedding dimension
    """
    # assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
    # x = np.array(x)
    # y = np.array(y)
    vec_x = np.array([0 for _ in range(dim)])
    for x_v in x:
        x_v = x_v
        vec_x = np.add(x_v, vec_x)
    vec_x = vec_x / math.sqrt(sum(np.square(vec_x)))
    vec_y = np.array([0 for _ in range(len(y[0]))])
    for y_v in y:
        y_v = np.array(y_v)
        vec_y = np.add(y_v, vec_y)
    vec_y = vec_y / math.sqrt(sum(np.square(vec_y)))
    assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
    zero_list = np.array([0 for _ in range(len(vec_x))])
    if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
        return float(1) if vec_x.all() == vec_y.all() else float(0)
    vec_x = np.mat(vec_x)
    vec_y = np.mat(vec_y)
    num = float(vec_x * vec_y.T)
    denom = np.linalg.norm(vec_x) * np.linalg.norm(vec_y)
    cos = num / denom
    return cos
def cal_vector_extrema(x:List[float], y:List[float]):
    # x = np.array(x)
    # y = np.array(y)
    vec_x = np.max(x, axis=0)
    vec_y = np.max(y, axis=0)
    assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
    zero_list = np.zeros(len(vec_x))
    if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
        return float(1) if vec_x.all() == vec_y.all() else float(0)
    res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return cos
def calculationEmbedding(config,predictions:List[List[str]],references:List[List[str]]):
    metricsResults = {}
    assert len(predictions) == len(references)
    gloveDic ={}
    if os.path.exists(os.path.join(config.glove,"glove.json")):
        with open(os.path.join(config.glove,"glove.json"),"r") as f:
            gloveDic = json.load(f)
    else:
        # embed_path = os.path.join(config.glove,"glove.42B.300d.txt")
        with open(config.emb_file,"r",encoding="utf-8") as f:
            for line in f:
                line  = line.rstrip()
                line = line.split()
                gloveDic[line[0]] = line[-config.emb_dim:]
        with open(os.path.join(config.glove,"glove.json"),"w") as f:
            json.dump(gloveDic,f)
    ea_sum, vx_sum, gm_sum, counterp = 0, 0, 0, 0
    for gold, con in tqdm(list(zip(references, predictions))):
        vec_gold = map_tokens2vectors(gold,gloveDic,config.emb_dim)
        vec_con =  map_tokens2vectors(con,gloveDic,config.emb_dim)
        ea_sum += cal_embedding_average(deepcopy(vec_gold), deepcopy(vec_con))
        vx_sum += cal_vector_extrema(deepcopy(vec_gold), deepcopy(vec_con))
        gm_sum += cal_greedy_matching_matrix(deepcopy(vec_gold), deepcopy(vec_con))
        counterp += 1
    metricsResults["EAVE"] = ea_sum / counterp
    metricsResults["EEXT"] = gm_sum / counterp
    metricsResults["EGRE"] = vx_sum / counterp
    return metricsResults
def cal_Distinct(corpus:List[List[str]]):
    """
    Calculates unigram and bigram diversity
    Args:
        corpus: tokenized list of sentences sampled
    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score
    """
    tokens =[]
    for words in corpus:
        tokens.extend(words)
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    if bigram_finder.N > 0:
        bi_diversity = len(bigram_finder.ngram_fd) / bigram_finder.N
    else:
        bi_diversity = 0
    trigram_finder = TrigramCollocationFinder.from_words(tokens)
    if trigram_finder.N > 0:
        tri_diversity = len(trigram_finder.ngram_fd) / trigram_finder.N
    else:
        tri_diversity = 0
    quagram_finder = QuadgramCollocationFinder.from_words(tokens)
    if quagram_finder.N > 0:
        qua_diversity = len(quagram_finder.ngram_fd) / quagram_finder.N
    else:
        qua_diversity = 0
    dist = FreqDist(tokens)
    if len(tokens) > 0:
        uni_diversity = len(dist) / len(tokens)
    else:
        uni_diversity = 0
    distincts = {"dist-1":uni_diversity,
                 "dist-2":bi_diversity,
                 "dist-3":tri_diversity,
                 "dist-4":qua_diversity,}
    return distincts
def compute_bleu_rouge_single_prediction(pred_dict, ref_dict, bleu_order=4):
    """
    parameters:
        predict_dict:{"key":[str]}
        ref_dict:{"key":[str1_ref,str2_ref]}
    for example:
        en:
            predict={"1":["this is a demo"]}
            ref_dict={"1":["this is a demo","this is a second demo"]}
        zh:
            predict={"1":["这 里 中国"]}
            ref_dict={"1":["你 是 日本 人","这 是 中国 人"]}
        result=compute_bleu_rouge(predict,ref_dict)
    """
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
            "missing keys: {}".format(set(ref_dict.keys()) - set(pred_dict.keys()))
    scores = {}
    bleu_scores, _ = Bleu(bleu_order).compute_score(ref_dict, pred_dict)
    for i, bleu_score in enumerate(bleu_scores):
        scores['Bleu-%d' % (i + 1)] = bleu_score
    rouge_score, _ = Rouge().compute_score(ref_dict, pred_dict)
    scores['Rouge-L'] = rouge_score
    return scores
def compute_bleu_rouge_multi_prediction(pred_dict, ref_dict, bleu_order=4,best_metrics="Rouge-L"):
    """
    parameters:
        predict_dict:{"key":[str]}
        ref_dict:{"key":[str1_ref,str2_ref]}
    for example:
        en:
            predict={"1":["this is a demo"]}
            ref_dict={"1":["this is a demo","this is a second demo"]}
        zh:
            predict={"1":["这 里 中国"]}
            ref_dict={"1":["你 是 日本 人","这 是 中国 人"]}
        result=compute_bleu_rouge(predict,ref_dict)
    """
    scores_prediction=OrderedDict()
    Best_prediction=OrderedDict()
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
            "missing keys: {}".format(set(ref_dict.keys()) - set(pred_dict.keys()))
    for qid,refs in ref_dict.items():
        predictions = pred_dict[qid]
        metrics_dict ={}
        best_score = -1
        for prediction in predictions:
            p_dict = {qid:[prediction]}
            r_dict = {qid:refs}
            metrics = compute_bleu_rouge_single_prediction(p_dict,r_dict,bleu_order=bleu_order)
            if metrics[best_metrics] > best_score:
                best_score = metrics[best_metrics]
                Best_prediction[qid] = [prediction]
            metrics_dict[prediction] = metrics
        scores_prediction[qid]=metrics_dict
    best_metrics = compute_bleu_rouge_single_prediction(Best_prediction,ref_dict)
    return scores_prediction,Best_prediction,best_metrics