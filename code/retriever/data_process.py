# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script
"""

from datetime import datetime
from utils import *
import os
import sys
import transformers
import pickle

from utils.general_utils import *

examples_dir = "APOLLO/dataset/retriever/"
features_dir = "APOLLO/dataset/retriever/"

root_path = "APOLLO/"
op_list_file = root_path + "dataset/operation_list.txt"
const_list_file = root_path + "dataset/constant_list.txt"
dataset_type = "finqa"  # or convfinqa
pretrained_model = "deberta"  # or roberta, bert
max_seq_length = 512
number_aware = True
model_size = "microsoft/deberta-v3-large"

if dataset_type == "finqa":
    if number_aware:
        train_file = root_path + "dataset/FinQA/train_number_aware.json"
    else:
        train_file = root_path + "dataset/FinQA/train.json"
    dev_file = root_path + "dataset/FinQA/dev.json"
    test_file = root_path + "dataset/FinQA/test.json"
else:
    if number_aware:
        train_file = root_path + "dataset/ConvFinQA/train_number_aware.json"
    else:
        train_file = root_path + "dataset/ConvFinQA/train.json"
    dev_file = root_path + "dataset/ConvFinQA/dev.json"
    test_file = ''


def create_features():
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    op_list = read_txt(op_list_file)
    op_list = [op + '(' for op in op_list]
    op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
    const_list = read_txt(const_list_file)
    const_list = [const.lower().replace('.', '_') for const in const_list]
    reserved_token_size = len(op_list) + len(const_list)

    if pretrained_model == "bert":
        print("Using bert")
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(model_size)

    elif pretrained_model == "roberta":
        print("Using roberta")
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(model_size)

    elif pretrained_model == "deberta":
        print("Using Deberta")
        from transformers import DebertaV2Tokenizer
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_size)

    else:
        print("Wrong pretrained Model, import Nothing!!")
        tokenizer = None

    train_data, train_examples, op_list, const_list = \
        read_examples(input_path=train_file, tokenizer=tokenizer,
                      op_list=op_list, const_list=const_list, dataset_type=dataset_type)
    dev_data, dev_examples, op_list, const_list = \
        read_examples(input_path=dev_file, tokenizer=tokenizer,
                      op_list=op_list, const_list=const_list, dataset_type=dataset_type)

    f = open(os.path.join(examples_dir, 'train_examples.pickle'), 'wb')
    pickle.dump([train_examples, op_list, const_list], f, 0)
    f.close()
    f = open(os.path.join(examples_dir, 'dev_examples.pickle'), 'wb')
    pickle.dump([dev_examples, op_list, const_list], f, 0)
    f.close()

    kwargs = {
        "examples": train_examples,
        "tokenizer": tokenizer,
        "is_training": True,
        "max_seq_length": max_seq_length,
    }

    train_features = convert_examples_to_features(**kwargs)
    kwargs["examples"] = dev_examples
    kwargs["is_training"] = False

    dev_features = convert_examples_to_features(**kwargs)

    f = open(os.path.join(features_dir, 'train_features.pickle'), 'wb')
    pickle.dump([train_features, op_list, const_list], f, 0)
    f.close()
    f = open(os.path.join(features_dir, 'dev_features.pickle'), 'wb')
    pickle.dump([dev_features, op_list, const_list], f, 0)
    f.close()

    if test_file:
        test_data, test_examples, op_list, const_list = \
            read_examples(input_path=test_file, tokenizer=tokenizer,
                          op_list=op_list, const_list=const_list, dataset_type=dataset_type)
        f = open(os.path.join(examples_dir, 'test_examples.pickle'), 'wb')
        pickle.dump([test_examples, op_list, const_list], f, 0)
        f.close()

        kwargs = {
            "examples": test_examples,
            "tokenizer": tokenizer,
            "is_training": False,
            "max_seq_length": max_seq_length,
            "pretrained_model": pretrained_model,
        }
        test_features = convert_examples_to_features(**kwargs)

        f = open(os.path.join(features_dir, 'test_features.pickle'), 'wb')
        pickle.dump([test_features, op_list, const_list], f, 0)
        f.close()


if __name__ == '__main__':
    create_features()
