#coding=utf-8
import json
from nltk import word_tokenize
import argparse
from collections import OrderedDict
from tqdm import tqdm
import os
from metrics.evaluationtool import cal_greedy_matching_matrix,cal_embedding_average,cal_vector_extrema
from metrics.evaluationtool import cal_Distinct,compute_bleu_rouge_single_prediction,cal_corpus_bleu

parser = argparse.ArgumentParser('parameters config for evaluation')
parameters_settings = parser.add_argument_group('parameters settings')
parameters_settings.add_argument('--embedding',type=str,default="embeddings",help="golve word2 vector")
parameters_settings.add_argument('--corpus',type=str,default="DSTC7_AVSD",help='select task to train')
parameters_settings.add_argument('--embedding_size',type=int,default=512,help="GoogleNews embeddings for evaluation")
parameters_settings.add_argument('--result_file',type=str,default="utils/dstc_sepa_o.json",help="the result filenames")
config=parser.parse_args()

def get_predictions():
    total_num = 0
    predictions,targets = [],[]
    with open(config.result_file,"r",encoding="utf-8") as f:
        results= json.load(f)
    for item in results:
        token_pred = word_tokenize(item["pred"])
        total_num += len(token_pred)
        predictions.append(" ".join(token_pred))
        targets.append(" ".join(word_tokenize(item["target"])))
    print(total_num*1.0/len(predictions))
    return predictions,targets

def calculationEmbedding(preddctions,references,dic):
        """
        parameters:
            preddctions:["this is a demo","this is a demo"]
            references:["this is a demo","this is a demo"]
        """
        assert len(preddctions) == len(references)
        preddctions = [prediction.split() for prediction in preddctions]
        references = [ref.split() for ref in references]
        ea_sum, vx_sum, gm_sum, counterp = 0, 0, 0, 0
        for rr, cc in list(zip(references, preddctions)):
            ea_sum += cal_embedding_average(rr, cc, dic,config.embedding_size)
            vx_sum += cal_vector_extrema(rr, cc, dic,config.embedding_size)
            gm_sum += cal_greedy_matching_matrix(rr, cc, dic,config.embedding_size)
            counterp += 1
        
        return ea_sum / counterp, gm_sum / counterp, vx_sum / counterp

def evaluate_metrics():
    results=[]
    emb_dic = OrderedDict()
    embedding = os.path.join(config.embedding,config.corpus,"vectors.txt")
    with open(embedding,"r",encoding="utf-8") as f:
        for line in f:
            line  = line.rstrip()
            line = line.split()
            token = str(line[0])
            vector = [float(item) for item in line[-config.embedding_size:]]
            assert len(vector) == config.embedding_size
            emb_dic[token] = vector
    predictions,ground_response = get_predictions()
    ea_sum,gm_sum,vx_sum = calculationEmbedding(predictions,ground_response,emb_dic)
    distinct_result = cal_Distinct(predictions)
    predDict = {i:[pred] for i,pred in enumerate(predictions)}
    labelDict = {i:[ref] for i,ref in enumerate(ground_response)}
    nltkBleu = cal_corpus_bleu(ground_response,predictions)
    metricsResult = compute_bleu_rouge_single_prediction(predDict,labelDict)
    metricsResult["EA"] = ea_sum
    metricsResult["GA"] = gm_sum
    metricsResult["VX"] = vx_sum
    for _key,_value in distinct_result.items():
        metricsResult[_key] = _value
    for _key,_value in nltkBleu.items():
        metricsResult[_key] = _value
    for keyId,tokens in predDict.items():
        results.append({"pred":tokens[0],"tgt":labelDict[keyId][0]})
    formatString="the prediction metrics is "
    for _key in metricsResult:
        formatString +=_key
        formatString +=" {} ".format(metricsResult[_key])
    print(formatString)
evaluate_metrics()
    



