import os
import sys
sys.path.append('DialogVED-main')

from utils.processor import convert_daily_dialog, check

FINETUNE_PREFIX_PATH = 'data/finetune'


ORIGINAL_PATH = os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/original_data')
PROCESSED_PATH = os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/processed')


convert_daily_dialog(
    fin=os.path.join(ORIGINAL_PATH, 'train.txt'),
    src_fout=os.path.join(PROCESSED_PATH, 'train.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'train.tgt'),
)
convert_daily_dialog(
    fin=os.path.join(ORIGINAL_PATH, 'valid.txt'),
    src_fout=os.path.join(PROCESSED_PATH, 'valid.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'valid.tgt'),
)
convert_daily_dialog(
    fin=os.path.join(ORIGINAL_PATH, 'test.txt'),
    src_fout=os.path.join(PROCESSED_PATH, 'test.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'test.tgt')
)

check(PROCESSED_PATH, mode='train')
check(PROCESSED_PATH, mode='valid')
check(PROCESSED_PATH, mode='test')
