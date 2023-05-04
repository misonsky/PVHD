#coding=utf-8
import json
results = []
with open("dstc.txt","r",encoding="utf-8") as f:
    for line in f:
        line = line.rstrip()
        items = line.split("\t")
        real = items[0]
        prediction = items[1]
        results.append({"pred":prediction,"tgt":real})

with open("dstc.json","w",encoding="utf-8") as f:
    json.dump(results, f,ensure_ascii=False,indent=4)
        
        