from .baselines import (LstmEncoder,
                       LstmModel,
                       CnnLstmModel)
import json
import torch
import h5py

def invert_dict(d):
  return {v: k for k, v in d.items()}

def load_vocab(path):
    # https://github.com/ethanjperez/film/blob/master/vr/utils.py
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>'] == 2
    return vocab

if __name__ == '__main__':

    #vocab = json.loads(
    #    open("/clevr_preprocessed/vocab.json", "r").read())

    vocab = load_vocab("/clevr_preprocessed/vocab.json")
    for key in vocab.keys():
        print(key)
        print(vocab[key])
        print()

    lstm = LstmModel(vocab)

    # Feed some sentence into the LSTM
    h5f = h5py.File("/clevr_preprocessed/val_questions.h5", "r")
    questions = h5f['questions'][0:16]
    x_batch = torch.from_numpy(questions).long()

    from .probe import ClevrProbe

    net = ClevrProbe(vocab)

    print(net)

    xfake = torch.randn((4,2048,7,7))
    print(net.classifier(xfake).shape)

    #out = lstm(x_batch)

    ############
