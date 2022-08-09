import h5py
import os
from collections import Counter
import numpy as np
import json

meta_dir = os.environ['DATASET_CLEVR_KIWI_META']

vocab = json.loads(open("%s/vocab.json" % meta_dir).read())
train_h5 = h5py.File("%s/train_questions.h5" % meta_dir, "r")
answers = train_h5['answers'][:]

token2idx = vocab['answer_token_to_idx']

ans2count = dict()
cc = 0
for key, val in token2idx.items():
    ans_count = (answers == val).sum()
    ans_count_percent = ans_count / len(answers) * 100.
    print(key, ans_count_percent, "%")
    cc += ans_count_percent

print(cc)
