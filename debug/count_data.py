import json
import numpy as np
import glob
from collections import Counter

"""
q_files = glob.glob("/clevr_kiwi_test/*/questions.json")
s_files = glob.glob("/clevr_kiwi_test/*/scenes.json")

print("counting how many questions...")
n_questions = 0
for file_ in q_files:
    dd = json.loads(open(file_).read())
    questions = dd['questions']
    n_questions += len(questions)
print("number of question files:", len(q_files))
print("number of questions:", n_questions)


n_scenes = 0
for file_ in s_files:
    dd = json.loads(open(file_).read())
    scenes = dd['scenes']
    n_scenes += len(scenes)
print("number of scenes files:", len(s_files))
print("number of scenes:", n_scenes)
"""

import h5py

f_train = h5py.File('/clevr_kiwi_preprocessed/train_questions.h5', 'r')
f_valid = h5py.File('/clevr_kiwi_preprocessed/valid_questions.h5', 'r')

n_questions_train = f_train['questions'].shape[0]*1.0
n_questions_valid = f_valid['questions'].shape[0]*1.0

percentage_valid = n_questions_valid / (n_questions_train+n_questions_valid)*100.

import pdb
pdb.set_trace()
