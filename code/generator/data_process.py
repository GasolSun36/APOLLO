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

examples_dir = "APOLLO/dataset/generator/"
features_dir = "APOLLO/dataset/generator/"

root_path = "APOLLO/"
op_list_file = root_path + "dataset/operation_list.txt"
const_list_file = root_path + "dataset/constant_list.txt"
dataset_type = "finqa"  # or convfinqa
pretrained_model = "roberta"  # or roberta, bert
max_seq_length = 512
max_program_length = 30
model_size = "roberta-large"
retrieve_mode = "single"
program_mode = "seq"
if dataset_type == "finqa":
    train_file = root_path + "dataset/FinQA/train_retrieve_output.json"
    dev_file = root_path + "dataset/FinQA/dev_retrieve_output.json"
    test_file = root_path + "dataset/FinQA/test_retrieve_output.json"
else:
    train_file = root_path + "dataset/ConvFinQA/train_retrieve_output.json"
    dev_file = root_path + "dataset/ConvFinQA/dev_retrieve_output.json"
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
                      op_list=op_list, const_list=const_list, retrieve_mode=retrieve_mode, dataset_type=dataset_type, max_seq_length=max_seq_length, program_mode=program_mode)
    dev_data, dev_examples, op_list, const_list = \
        read_examples(input_path=dev_file, tokenizer=tokenizer,
                      op_list=op_list, const_list=const_list, retrieve_mode=retrieve_mode, dataset_type=dataset_type, max_seq_length=max_seq_length, program_mode=program_mode)

    f = open(os.path.join(examples_dir, 'train_examples.pickle'), 'wb')
    pickle.dump([train_examples, op_list, const_list], f, 0)
    f.close()
    f = open(os.path.join(examples_dir, 'dev_examples.pickle'), 'wb')
    pickle.dump([dev_examples, op_list, const_list], f, 0)
    f.close()

    kwargs = {
        "examples": train_examples,
        "tokenizer": tokenizer,
        "max_seq_length": max_seq_length,
        "max_program_length": max_program_length,
        "is_training": True,
        "op_list": op_list,
        "op_list_size": len(op_list),
        "const_list": const_list,
        "const_list_size": len(const_list),
        "verbose": True
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
                          op_list=op_list, const_list=const_list, retrieve_mode=retrieve_mode, dataset_type=dataset_type, max_seq_length=max_seq_length, program_mode=program_mode)
        f = open(os.path.join(examples_dir, 'test_examples.pickle'), 'wb')
        pickle.dump([test_examples, op_list, const_list], f, 0)
        f.close()

        kwargs = {
            "examples": test_examples,
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
            "max_program_length": max_program_length,
            "is_training": False,
            "op_list": op_list,
            "op_list_size": len(op_list),
            "const_list": const_list,
            "const_list_size": len(const_list),
            "verbose": True
        }

        test_features = convert_examples_to_features(**kwargs)

        f = open(os.path.join(features_dir, 'test_features.pickle'), 'wb')
        pickle.dump([test_features, op_list, const_list], f, 0)
        f.close()


if __name__ == '__main__':
    create_features()
