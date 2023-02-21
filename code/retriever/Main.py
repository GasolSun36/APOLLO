#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script
"""
from utils.training_config import *
from utils.general_utils import *
from Model import Bert_model
from transformers import (
    get_linear_schedule_with_warmup, )
import pickle
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler
import torch
from torch import nn
from utils import *
from tqdm import tqdm
import json
import logging
import os
import random

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def collate_fn(features):
    batch_data = {
        "input_ids": [],
        "input_mask": [],
        "segment_ids": [],
        "filename_id": [],
        "label": [],
        "ind": []
    }
    for each_data in features:
        batch_data["input_ids"].append(each_data["input_ids"])
        batch_data["input_mask"].append(each_data["input_mask"])
        batch_data["segment_ids"].append(each_data["segment_ids"])
        batch_data["filename_id"].append(each_data["filename_id"])
        batch_data["label"].append(each_data["label"])
        batch_data["ind"].append(each_data["ind"])
    return batch_data


def load_model(args, op_list, const_list):
    if args.pretrained_model == "bert":
        logger.info("Using bert")
        from transformers import BertTokenizer
        from transformers import BertConfig

        tokenizer = BertTokenizer.from_pretrained(args.model_size)
        model_config = BertConfig.from_pretrained(args.model_size)

    elif args.pretrained_model == "roberta":
        logger.info("Using roberta")
        from transformers import RobertaTokenizer
        from transformers import RobertaConfig

        tokenizer = RobertaTokenizer.from_pretrained(args.model_size)
        model_config = RobertaConfig.from_pretrained(args.model_size)

    elif args.pretrained_model == "longformer":
        logger.info("Using longformer")
        from transformers import LongformerTokenizer, LongformerConfig

        tokenizer = LongformerTokenizer.from_pretrained(args.model_size)
        model_config = LongformerConfig.from_pretrained(args.model_size)
    elif args.pretrained_model == "deberta":
        logger.info("Using Deberta")
        from transformers import DebertaV2Tokenizer, DebertaV2Config

        tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_size)
        model_config = DebertaV2Config.from_pretrained(args.model_size)

    else:
        logger.info("Wrong pretrained Model, import Nothing!!")
        tokenizer = None
        model_config = None

    model = Bert_model(hidden_size=model_config.hidden_size,
                       dropout_rate=args.dropout_rate,
                       args=args)

    return tokenizer, model


def get_arguments():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--prog_name",
        default="retriever",
        type=str,
        help="Model type: retriever or generator",
    )
    parser.add_argument("--root_path",
                        default="APOLLO/",
                        type=str,
                        help="root path for APOLLO")
    parser.add_argument("--output_dir", type=str,
                        help="output path for APOLLO")
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="cache dir for APOLLO to save pre-train models. Metioned that on server training you need to download them in advance."
    )

    parser.add_argument("--model_save_name",
                        default="retriever-deberta",
                        type=str,
                        help="Names that saves output and inference.")

    # Other parameters
    parser.add_argument("--train_file",
                        type=str,
                        help="file using in training.")
    parser.add_argument("--valid_file",
                        type=str,
                        help="file using in validing.")
    parser.add_argument("--test_file", type=str, help="file using in testing.")
    parser.add_argument("--op_list_file",
                        default="operation_list.txt",
                        type=str,
                        help="file that saves all the possible operation.")
    parser.add_argument("--const_list_file",
                        default="constant_list.txt",
                        type=str,
                        help="file that saves all the possible const.")
    parser.add_argument("--pretrained_model",
                        default="deberta",
                        type=str,
                        help="choose your pretrained_model.")

    parser.add_argument("--model_size",
                        default="microsoft/deberta-v3-large",
                        type=str,
                        help="choose your pretrained_model size.")

    parser.add_argument("--world_size",
                        type=int,
                        help="world size,only for mutil-mechians")
    parser.add_argument("--device",
                        default="cuda",
                        type=str,
                        help="device,cuda or cpu")
    parser.add_argument("--mode",
                        default="train",
                        type=str,
                        help="mode, train for training, test for inference.")
    parser.add_argument("--build_summary",
                        default=False,
                        type=bool,
                        help="Do not know how to use, just default.")
    parser.add_argument("--saved_model_path",
                        type=str,
                        help="saved model path for testing")
    parser.add_argument("--option", default="rand", type=str, help="option")
    parser.add_argument(
        "--neg_rate",
        default=3,
        type=int,
        help="neg_rage for create features, not used in longformer retriever")
    parser.add_argument("--topn", default=5, type=int, help="topn to evaluate")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="2k or 3k for longformer, 512 for others.")
    parser.add_argument("--max_program_length",
                        default=30,
                        type=int,
                        help="max program length.")
    parser.add_argument("--n_best_size", default=20, type=int, help="none")
    parser.add_argument("--dropout_rate",
                        default=0.1,
                        type=float,
                        help="dropout rate")
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="batch size for training")
    parser.add_argument("--batch_size_test",
                        type=int,
                        default=16,
                        help="batch size for test")
    parser.add_argument("--epoch",
                        default=100,
                        type=int,
                        help="epoch for training")
    parser.add_argument(
        "--report",
        default=500,
        type=int,
        help="iterations for reporting validation on valid set.")
    parser.add_argument("--report_loss",
                        default=100,
                        type=int,
                        help="iterations for reporting loss.")
    parser.add_argument("--seed",
                        type=int,
                        default=2022,
                        help="random seed for initialization")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--rank", type=int, default=-1, help="rank")
    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--fp16",
                        action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="gradient_accumulation_steps")
    parser.add_argument("--max_grad_norm",
                        type=float,
                        default=0.2,
                        help="max_grad_norm")
    parser.add_argument("--learning_rate",
                        default=4e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--features_dir",
        type=str,
        default="FinQA/dataset/retriever/",
        help="Max gradient norm.")
    parser.add_argument(
        "--examples_dir",
        type=str,
        default="FinQA/dataset/retriever/",
        help="Max gradient norm.")
    parser.add_argument("--tags",
                        type=str,
                        required=True,
                        help="which experiment your are doing right now ")
    parser.add_argument("--result_dir",
                        type=str,
                        help="which saves the result of test")
    parser.add_argument("--dataset_type",
                        default='finqa',
                        type=str,
                        help="which dataset to train")
    args = parser.parse_args()
    return args


def process_data_features(args, features, is_training):
    data_pos = features[0]
    data_neg = features[1]
    if is_training:
        random.shuffle(data_neg)
        num_neg = len(data_pos) * args.neg_rate
        data = data_pos + data_neg[:num_neg]
    else:
        data = data_pos + data_neg
    return data


def train(args):
    args.output_dir = args.root_path + "output/"
    args.cache_dir = args.root_path + "cache/"
    if args.dataset_type == "finqa":
        args.train_file = args.root_path + "dataset/FinQA/train.json"
        args.dev_file = args.root_path + "dataset/FinQA/dev.json"
        args.test_file = args.root_path + "dataset/FinQA/test.json"
    else:
        args.train_file = args.root_path + "dataset/ConvFinQA/train.json"
        args.dev_file = args.root_path + "dataset/ConvFinQA/dev.json"
        args.test_file = ''
    args.model_save_name = args.prog_name + \
        args.pretrained_model + args.model_size + args.tags
    args.output_dir = args.output_dir + "{}_setting".format(
        args.model_save_name)

    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    torch.distributed.barrier()
    log_path = os.path.join(args.output_dir, 'log.txt')
    handler = logging.FileHandler(log_path, 'a', 'utf-8')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if args.local_rank in
                    [-1, 0] else logging.WARN)
    print(logger)

    # tensorboard writer
    logger.info("Training/evaluation parameters %s", args)
    tb_writer = None
    if is_first_worker():
        args.log_dir = os.path.join(args.output_dir, "tb_dir")
        tb_writer = SummaryWriter(log_dir=args.log_dir)
    torch.distributed.barrier()

    with open(os.path.join(args.features_dir, 'train_features.pickle'),
              'rb') as f:
        train_features, op_list, const_list = pickle.load(f)
    with open(os.path.join(args.features_dir, 'dev_features.pickle'),
              'rb') as f:
        dev_features, op_list, const_list = pickle.load(f)
    with open(os.path.join(args.examples_dir, 'dev_examples.pickle'),
              'rb') as f:
        dev_examples, op_list, const_list = pickle.load(f)
    if args.test_file:
        with open(os.path.join(args.features_dir, 'test_features.pickle'),
                  'rb') as f:
            test_features, op_list, const_list = pickle.load(f)
        with open(os.path.join(args.features_dir, 'test_examples.pickle'),
                  'rb') as f:
            test_examples, op_list, const_list = pickle.load(f)
        test_features = process_data_features(args, test_features, False)

    train_features = process_data_features(args, train_features, True)
    dev_features = process_data_features(args, dev_features, False)

    logger.info("train set length:{}".format(len(train_features)))
    logger.info("dev set length:{}".format(len(dev_features)))
    logger.info("test set length:{}".format(len(test_features)))

    train_sampler = RandomSampler(
        train_features) if args.local_rank == -1 else DistributedSampler(
            train_features)
    train_dataloader = DataLoader(train_features,
                                  sampler=train_sampler,
                                  collate_fn=collate_fn,
                                  batch_size=args.batch_size,
                                  num_workers=10)

    tokenizer, model = load_model(args, op_list, const_list)
    optimizer = get_optimizer(args, model, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    model.to(args.device)
    logger.info("total train data size:{}".format(len(train_features)))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=args.fp16_opt_level)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    model.train()
    # Train!
    logger.info("***** Running setting *****")
    args.world_size = torch.distributed.get_world_size()
    batch_size_all = args.batch_size * \
        args.world_size * args.gradient_accumulation_steps

    args.max_steps = args.epoch * len(train_features) // batch_size_all
    logger.info("  Max steps = %d", args.max_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.batch_size * args.gradient_accumulation_steps *
        (args.world_size if args.local_rank != -1 else 1),
    )
    # torch.autograd.set_detect_anomaly(True)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.max_steps // 20,
        num_training_steps=args.max_steps)

    record_loss = 0.0
    global_step = 0
    logger.info("***** Running training *****")
    for _ in range(args.epoch):
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, x in enumerate(epoch_iterator):
            input_ids = torch.tensor(x['input_ids']).to(args.device)
            input_mask = torch.tensor(x['input_mask']).to(args.device)
            segment_ids = torch.tensor(x['segment_ids']).to(args.device)
            label = torch.tensor(x['label']).to(args.device)

            this_logits = model(True,
                                input_ids,
                                input_mask,
                                segment_ids,
                                device=args.device)

            this_loss = criterion(this_logits.view(-1, this_logits.shape[-1]),
                                  label.view(-1))

            this_loss = this_loss.sum()
            if args.fp16:
                with amp.scale_loss(this_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                this_loss.backward()

            record_loss += this_loss.item() * 100
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if args.report_loss > 0 and global_step % args.report_loss == 0:
                logs = {}
                loss_scalar = record_loss / args.report_loss
                learning_rate_scalar = scheduler.get_last_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                record_loss = 0
                if is_first_worker():
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

                if args.report > 0 and global_step % args.report == 0:
                    results_path = os.path.join(args.output_dir, "results")
                    results_path_cnt = os.path.join(results_path, 'loads',
                                                    str(global_step))
                    if is_first_worker():
                        evaluate(args, dev_features, model, results_path_cnt,
                                 'dev')
                        saved_model_path_cnt = os.path.join(
                            args.output_dir, 'loads', str(global_step))
                        os.makedirs(saved_model_path_cnt, exist_ok=True)
                        torch.save(model.state_dict(),
                                   saved_model_path_cnt + "/model.pt")
                model.train()


def evaluate(args, data_features, model, ksave_dir, mode='dev'):
    if mode == 'dev':
        data = args.dev_file
    else:
        data = args.test_file
    ksave_dir_mode = os.path.join(ksave_dir, mode)
    temp_dir_mode = os.path.join(ksave_dir_mode, 'temp')
    os.makedirs(ksave_dir_mode, exist_ok=True)
    os.makedirs(temp_dir_mode, exist_ok=True)
    test_dataloader = DataLoader(data_features,
                                 collate_fn=collate_fn,
                                 batch_size=args.batch_size_test,
                                 num_workers=10)
    all_logits = []
    all_filename_id = []
    all_ind = []
    with torch.no_grad():
        model.eval()
        for x in tqdm(test_dataloader):
            input_ids = torch.tensor(x['input_ids']).to(args.device)
            input_mask = torch.tensor(x['input_mask']).to(args.device)
            segment_ids = torch.tensor(x['segment_ids']).to(args.device)
            filename_id = x["filename_id"]
            ind = x["ind"]

            logits = model(False,
                           input_ids,
                           input_mask,
                           segment_ids,
                           device=args.device)

            all_logits.extend(logits.tolist())
            all_filename_id.extend(filename_id)
            all_ind.extend(ind)

    output_prediction_file = os.path.join(ksave_dir_mode,
                                          "predictions_{}.json".format(mode))

    print_res = retrieve_evaluate(all_logits,
                                  all_filename_id,
                                  all_ind,
                                  output_prediction_file,
                                  data,
                                  topn=args.topn)

    logger.info("Result :{} \n".format(print_res))
    return


def inference(args):
    args.output_dir = args.root_path + "output/"
    args.cache_dir = args.root_path + "cache/"
    if args.dataset_type == "finqa":
        args.train_file = args.root_path + "dataset/FinQA/train.json"
        args.dev_file = args.root_path + "dataset/FinQA/dev.json"
        args.test_file = args.root_path + "dataset/FinQA/test.json"
    else:
        args.train_file = args.root_path + "dataset/ConvFinQA/train.json"
        args.dev_file = args.root_path + "dataset/ConvFinQA/dev.json"
        args.test_file = ''
    args.model_save_name = args.prog_name + \
        args.pretrained_model + args.model_size + args.tags
    args.output_dir = args.output_dir + "{}_setting".format(
        args.model_save_name)
    str = args.saved_model_path.split("/")
    args.result_dir = args.root_path + "model/" + "{}".format(str[-4] + '_' +
                                                              str[-2])

    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    # Create output directory if needed
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    log_path = os.path.join(args.result_dir, 'log.txt')
    handler = logging.FileHandler(log_path, 'a', 'utf-8')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if args.local_rank in
                    [-1, 0] else logging.WARN)
    print(logger)

    # tensorboard writer
    logger.info("Training/evaluation parameters %s", args)
    tb_writer = None
    args.log_dir = os.path.join(args.result_dir, "tb_dir")
    tb_writer = SummaryWriter(log_dir=args.log_dir)

    with open(os.path.join(args.features_dir, 'train_features.pickle'),
              'rb') as f:
        train_features, op_list, const_list = pickle.load(f)
    with open(os.path.join(args.features_dir, 'dev_features.pickle'),
              'rb') as f:
        dev_features, op_list, const_list = pickle.load(f)
    with open(os.path.join(args.examples_dir, 'dev_examples.pickle'),
              'rb') as f:
        dev_examples, op_list, const_list = pickle.load(f)
    if args.test_file:
        with open(os.path.join(args.features_dir, 'test_features.pickle'),
                  'rb') as f:
            test_features, op_list, const_list = pickle.load(f)
        with open(os.path.join(args.features_dir, 'test_examples.pickle'),
                  'rb') as f:
            test_examples, op_list, const_list = pickle.load(f)
        test_features = process_data_features(args, test_features, False)

    train_features = process_data_features(args, train_features, True)
    dev_features = process_data_features(args, dev_features, False)

    reserved_token_size = len(op_list) + len(const_list)
    print("loading model from {}".format(args.saved_model_path))

    tokenizer, model = load_model(args, op_list, const_list)
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    model.load_state_dict(torch.load(args.saved_model_path))
    model.eval()
    logger.info("total train data size:{}".format(len(train_features)))
    logger.info("total test data size:{}".format(len(test_features)))
    # Test!
    logger.info("***** Running inference *****")
    if is_first_worker():
        generate_retrieve(train_features, model, args.result_dir, mode='train')
        generate_retrieve(dev_features, model, args.result_dir, mode='dev')
        if args.test_file:
            generate_retrieve(test_features,
                              model,
                              args.result_dir,
                              mode='test')
        # generate_retrieve(private_features, model, args.result_dir, mode='private')


def generate_retrieve(features, model, ksave_dir, mode='dev'):
    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)
    if mode == 'test':
        data = args.test_file
    elif mode == 'dev':
        data = args.dev_file
    else:
        data = args.train_file

    test_dataloader = DataLoader(features,
                                 collate_fn=collate_fn,
                                 batch_size=args.batch_size_test,
                                 num_workers=10)
    all_logits = []
    all_filename_id = []
    all_ind = []
    with torch.no_grad():
        for x in tqdm(test_dataloader):
            input_ids = torch.tensor(x['input_ids']).to(args.device)
            input_mask = torch.tensor(x['input_mask']).to(args.device)
            segment_ids = torch.tensor(x['segment_ids']).to(args.device)
            filename_id = x["filename_id"]
            ind = x["ind"]

            logits = model(False,
                           input_ids,
                           input_mask,
                           segment_ids,
                           device=args.device)

            all_logits.extend(logits.tolist())
            all_filename_id.extend(filename_id)
            all_ind.extend(ind)

    output_prediction_file = os.path.join(ksave_dir_mode,
                                          "{}_retrieve.json".format(mode))
    print_res = retrieve_evaluate(all_logits,
                                  all_filename_id,
                                  all_ind,
                                  output_prediction_file,
                                  data,
                                  topn=args.topn)

    logger.info("res :{}".format(print_res))
    return


def convert(args):
    args.output_dir = args.root_path + "output/"
    args.cache_dir = args.root_path + "cache/"
    if args.dataset_type == "finqa":
        args.train_file = args.root_path + "dataset/FinQA/train.json"
        args.dev_file = args.root_path + "dataset/FinQA/dev.json"
        args.test_file = args.root_path + "dataset/FinQA/test.json"
    else:
        args.train_file = args.root_path + "dataset/ConvFinQA/train.json"
        args.dev_file = args.root_path + "dataset/ConvFinQA/dev.json"
        args.test_file = ''
    args.model_save_name = args.prog_name + \
        args.pretrained_model + args.model_size + args.tags
    args.output_dir = args.output_dir + "{}_setting".format(
        args.model_save_name)
    str = args.saved_model_path.split("/")
    args.result_dir = args.root_path + "model/" + "{}".format(str[-4] + '_' +
                                                              str[-2])
    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    # Create output directory if needed
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    log_path = os.path.join(args.result_dir, 'log.txt')
    handler = logging.FileHandler(log_path, 'a', 'utf-8')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if args.local_rank in
                    [-1, 0] else logging.WARN)
    print(logger)
    logger.info("***** Running converting *****")
    train_retrieve_file = os.path.join(args.result_dir, "train",
                                       "train_retrieve.json")
    train_output_file = os.path.join(args.result_dir, "train",
                                     "train_retrieve_output.json")
    dev_retrieve_file = os.path.join(args.result_dir, "dev",
                                     "dev_retrieve.json")
    dev_output_file = os.path.join(args.result_dir, "dev",
                                   "dev_retrieve_output.json")
    if args.test_file:
        test_retrieve_file = os.path.join(args.result_dir, "test",
                                          "test_retrieve.json")
        test_output_file = os.path.join(args.result_dir, "test",
                                        "test_retrieve_output.json")

    if is_first_worker():
        convert_train(train_retrieve_file,
                      train_output_file,
                      topn=3,
                      max_len=290,
                      dataset_type=args.dataset_type)
        convert_test(dev_retrieve_file,
                     dev_output_file,
                     topn=3,
                     max_len=290,
                     dataset_type=args.dataset_type)
        if args.test_file:
            convert_test(test_retrieve_file,
                         test_output_file,
                         topn=3,
                         max_len=290,
                         dataset_type=args.dataset_type)


if __name__ == '__main__':
    args = get_arguments()
    set_env(args)
    if args.mode == "train":
        train(args)
    elif args.mode == "inference":
        inference(args)
    elif args.mode == "convert":
        convert(args)
