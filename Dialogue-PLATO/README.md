# PLATO
**PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable**
[paper link](https://www.aclweb.org/anthology/2020.acl-main.9.pdf)

**\*\*\*\*\* Update \*\*\*\*\***

## Requirements
```
- python >= 3.6
- paddlepaddle == 1.6.1
- numpy
- nltk
- tqdm
- visualdl >= 1.3.0 (optional)
- regex
```
Recommend you install to python packages by command: `pip install -r requirement.txt`


## Fine-tuning
We also provide instructions to fine-tune PLATO on different conversation datasets (chit-chat, knowledge grounded dialogues and conversational question answering).

### Data preparation
Download data from the [link](https://baidu-nlp.bj.bcebos.com/PLATO/data.tar.gz).
The tar file contains three processed datasets: `DailyDialog`, `PersonaChat` and `DSTC7_AVSD`.
```bash
mv /path/to/data.tar.gz .
tar xzf data.tar.gz
```

### Data format
Our model supports two kinds of data formats for dialogue context: `multi` and `multi_knowledge`.
* `multi`: multi-turn dialogue context.
```txt
u_1 __eou__ u_2 __eou__ ... u_n \t r
```
If you want to use this model on other datasets, you can process your data accordingly.

### Train
Fine-tuning the pre-trained model on different `${DATASET}`.
```bash
# DailyDialog / PersonaChat / DSTC7_AVSD
DATASET=DailyDialog
sh scripts/${DATASET}/train.sh
```
After training, you can find the output folder `outputs/${DATASET}` (by default). It contatins `best.model` (best results on validation dataset), `hparams.json` (hyper-parameters of training script) and `trainer.log` (training log).


Fine-tuning the pre-trained model on multiple GPUs.

Note: You need to install NCCL library and set up the environment variable `LD_LIBRARY` properly.
```bash
sh scripts/DailyDialog/multi_gpu_train.sh
```

You can fine-tune PLATO w/o latent on different `${DATASET}`. We provide an example script on DailyDialog dataset.
```bash
sh scripts/DailyDialog/baseline_train.sh
```

#### Recommended settings

For the fine-tuning of our pre-trained model, it usually requires about 10 epochs to reach convergence with learning rate = 1e-5 and about 2-3 epochs to reach convergence with learning rate = 5e-5.

| GPU Memory | batch size | max len |
| ---------- | ---------- | ------- |
| 16G        | 6          | 256     |
| 32G        | 12         | 256     |

### Infer
Running inference on test dataset.
```bash
# DailyDialog / PersonaChat / DSTC7_AVSD
DATASET=DailyDialog
sh scripts/${DATASET}/infer.sh

# Running inference of PLATO w/o latent
sh scripts/DailyDialog/baseline_infer.sh
```
After inference, you can find the output foler `outputs/${DATASET}.infer` (by default). It contains `infer_0.result.json` (the inference result), `hparams.json` (hyper-parameters of inference scipt) and `trainer.log` (inference log).

If you want to use top-k sampling (beam search by default), you can follow the example script:
```bash
sh scripts/DailyDialog/topk_infer.sh
```

### Evaluation
```
python pvgru/PVGRU/eval_tool.py --corpus DailyDialog --result_file predictons.json
```
