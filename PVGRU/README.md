### PVGRU

The warehouse contains VHCR，CVAE，VHRED， HRAN， VRNN/HVRNN, seq2seq, hred, CSG(atthred: static attention + hred) and PVGRU/PVHD models.

### RUN

```
python run.py \
       --gpu 0 \
       --corpus DailyDialog \
       --model vhcr \
       --hier True \
       --beam_size 5 \
       --encoder_layers 2 \
       --decoder_layers 1 \
       --eval_steps 1000 \
       --log_steps 50 \
       --do_train
```

### Using PVGRU Module
Please specify the "model" parameter:
> pvhred (PVHD)  
> pvatthred (csg + pvgru)  
> pvhran (hran + pvgru)  
> varseq2seq (seq2seq + pvgru)

