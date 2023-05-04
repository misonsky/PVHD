#coding=utf-8
import json

with open("dstc_sepa.json","r",encoding="utf-8") as f:
    results = json.load(f)

_ids = []
with open("test_id_DSTC7_AVSD","r",encoding="utf-8") as f:
    for line in f:
        line = line.rstrip()
        _ids.append(int(line))

resultDict = {}

for _ids,item in zip(_ids,results):
    resultDict[_ids] = item

OrderResult = []
for i in sorted (resultDict):
    OrderResult.append(resultDict[i])

with open("dstc_sepa_o.json","w",encoding="utf-8") as f:
    json.dump(OrderResult, f,ensure_ascii=False,indent=4)















