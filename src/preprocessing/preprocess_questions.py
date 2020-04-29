#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import h5py
import numpy as np
import os
import argparse

from preprocessing.text_functions import tokenize, encode, build_vocab

"""
Preprocessing script for CLEVR question files.
"""
def extract_short_json(json_data_path, out_path, num_questions):

  with open(json_data_path, 'r') as f:
    questions = json.load(f)['questions']
  select_questions = questions[:num_questions]
  out_json = {'questions': select_questions}

  with open(out_path, 'w') as f:
    json.dump(out_json, f)

def preprocess_questions(min_token_count, punct_to_keep, punct_to_remove, add_start_token, add_end_token, json_data_path, vocab_out_path, h5_out_path):

  print('Loading Data...')
  with open(json_data_path, 'r') as f:
    questions = json.load(f)['questions']

  print('number of questions in json file: {}'.format(len(questions)))

  if os.path.isfile(vocab_out_path):
    print('Loading vocab...')
    with open(vocab_out_path, 'r') as f:
      vocab = json.load(f)
  else:
    print('Building vocab...')
    list_questions = [q['question'] for q in questions]
    question_token_to_idx = build_vocab(sequences=list_questions,
                                          min_token_count=min_token_count,
                                          punct_to_keep=punct_to_keep,
                                          punct_to_remove=punct_to_remove)
    print('number of words in vocab: {}'.format(len(question_token_to_idx)))
    vocab = {'question_token_to_idx': question_token_to_idx}

    with open(vocab_out_path, 'w') as f:
        json.dump(vocab, f)

  print('Encoding questions...')
  input_questions_encoded = []
  target_questions_encoded = []
  for orig_idx, q in enumerate(questions):
    question = q['question']

    question_tokens = tokenize(s=question,
                              punct_to_keep=punct_to_keep,
                              punct_to_remove=punct_to_remove,
                              add_start_token=add_start_token,
                              add_end_token=add_end_token)
    question_encoded = encode(seq_tokens=question_tokens,
                              token_to_idx=vocab['question_token_to_idx'],
                              allow_unk=True)
    input_question, target_question = question_encoded[:-1], question_encoded[1:]
    assert len(input_question) == len(target_question)
    input_questions_encoded.append(input_question)
    target_questions_encoded.append(target_question)

  # Pad encoded questions
  max_question_length = max(len(x) for x in input_questions_encoded)
  for iqe, tqe in zip(input_questions_encoded, target_questions_encoded):
    while len(iqe) < max_question_length:
      iqe.append(vocab['question_token_to_idx']['<PAD>'])
      tqe.append(vocab['question_token_to_idx']['<PAD>'])
    assert len(iqe) == len(tqe)

  # Create h5 file
  print('Writing output...')
  input_questions_encoded = np.asarray(input_questions_encoded, dtype=np.int32)
  target_questions_encoded = np.asarray(target_questions_encoded, dtype=np.int32)
  print("input questions shape", input_questions_encoded.shape)
  print('target questions shape', target_questions_encoded.shape)
  with h5py.File(h5_out_path, 'w') as f:
    f.create_dataset('input_questions', data=input_questions_encoded)
    f.create_dataset('target_questions', data=target_questions_encoded)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-data_path", type=str, required=True, help="path for CLEVR questions json files")
  parser.add_argument("-out_vocab_path", type=str, default="../../data/vocab.json", required=True, help="output path for vocab")
  parser.add_argument("-out_h5_path", type=str, required=True, help="output path for questions encoded dataset")
  parser.add_argument("-min_token_count", type=int, default=1, required=True, help="min count for adding token in vocab")
  parser.add_argument('-num_samples', type=int, default=100, help="used to select a subset of the whole CLEVR dataset")

  args = parser.parse_args()

  punct_to_keep = [';', ',', '?']
  punct_to_remove = ['.']

  #if len(args.num_samples) > 0:
  #print("selecting a subset of {} questions....".format(args.num_samples))
  # num_samples = 50000
  # val_samples = 20000
  # # train_data_path = "../../data/CLEVR_v1.0/questions/CLEVR_train_questions.json"
  # # train_out_path = "../../data/CLEVR_v1.0/temp/train_questions_{}_samples.json".format(num_samples)
  # # extract_short_json(out_path=train_out_path, json_data_path=train_data_path, num_questions=num_samples)
  # # val_data_path = "../../data/CLEVR_v1.0/questions/CLEVR_val_questions.json"
  # # val_out_path = "../../data/CLEVR_v1.0/temp/train_val_{}_samples.json".format(num_samples)
  # # extract_short_json(out_path=val_out_path, json_data_path=val_data_path, num_questions=val_samples)
  # test_data_path = "../../data/CLEVR_v1.0/questions/CLEVR_test_questions.json"
  # test_out_path = "../../data/CLEVR_v1.0/temp/test_questions_{}_samples.json".format(val_samples)
  # extract_short_json(out_path=test_out_path, json_data_path=test_data_path, num_questions=val_samples)


  preprocess_questions(min_token_count=1,
                       punct_to_keep=punct_to_keep,
                       punct_to_remove=punct_to_remove,
                       add_start_token=True,
                       add_end_token=True,
                       json_data_path=args.data_path,
                       vocab_out_path=args.out_vocab_path,
                       h5_out_path=args.out_h5_path)
