import sys
import h5py

#f = h5py.File('clevr_kiwi_v1_preprocessed/chris-v4/train_questions.h5', 'r')

DATA_DIR = "/mnt/public/datasets/clevr-mrt/v2/"

f = h5py.File("{}/metadata/train_questions.h5".format(DATA_DIR), "r")

filename = sys.argv[1]

print(filename)

filenames = f['image_filenames'][:].astype('str')
questions = f['question_strs'][:]
questions = questions.astype('str')

print( questions[filenames == filename] )

import pdb
pdb.set_trace()