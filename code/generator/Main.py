#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transformers import (
    get_linear_schedule_with_warmup, )
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler
from utils import *
import pickle
from transformers import AdamW
import torch.distributed as dist
from Model import Bert_model
from torch import nn
import torch
import os
import tqdm

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def dataloader_collate_fn(args):

    def collate_fn(features):
        batch_data = {
            "unique_id": [],
            "example_index": [],
            "tokens": [],
            "question": [],
            "input_ids": [],
            "input_mask": [],
            "option_mask": [],
            "segment_ids": [],
            "options": [],
            "answer": [],
            "program": [],
            "program_ids": [],
            "program_weight": [],
            "program_mask": [],
            "example": []
        }
        for each_data in features:
            batch_data["option_mask"].append(each_data.option_mask)
            batch_data["input_mask"].append(each_data.input_mask)
            batch_data["unique_id"].append(each_data.unique_id)
            batch_data["example_index"].append(each_data.example_index)
            batch_data["tokens"].append(each_data.tokens)
            batch_data["question"].append(each_data.question)
            batch_data["input_ids"].append(each_data.input_ids)
            batch_data["segment_ids"].append(each_data.segment_ids)
            batch_data["options"].append(each_data.options)
            batch_data["answer"].append(each_data.answer)
            batch_data["program"].append(each_data.program)
            batch_data["program_ids"].append(each_data.program_ids)
            batch_data["program_weight"].append(each_data.program_weight)
            batch_data["program_mask"].append(each_data.program_mask)
            batch_data["example"].append(each_data.example)
        return batch_data

    return collate_fn


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

    elif args.pretrained_model == "finbert":
        logger.info("Using finbert")
        from transformers import BertTokenizer
        from transformers import BertConfig

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model_config = BertConfig.from_pretrained(args.model_size)

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
    model = Bert_model(num_decoder_layers=args.num_decoder_layers,
                       hidden_size=model_config.hidden_size,
                       dropout_rate=args.dropout_rate,
                       input_length=args.max_seq_length,
                       program_length=args.max_program_length,
                       op_list=op_list,
                       const_list=const_list,
                       args=args)
    return tokenizer, model


def get_arguments():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--prog_name",
        default="generator",
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
                        default="generator-roberta-large",
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
                        default="roberta",
                        type=str,
                        help="choose your pretrained_model.")

    parser.add_argument("--model_size",
                        default="roberta-large",
                        type=str,
                        help="choose your pretrained_model size.")

    parser.add_argument(
        "--retrieve_mode",
        default="single",
        type=str,
        help="single, slide, gold, none. none for longformer,gold for best,slide for sliding window,single for single sentence."
    )

    parser.add_argument(
        "--program_mode",
        default="seq",
        type=str,
        help="use seq program or nested program, seq or nested")
    parser.add_argument("--world_size",
                        type=int,
                        help="world size,only for mutil-mechians")
    parser.add_argument("--device", type=str, help="device,cuda or cpu")
    parser.add_argument("--mode",
                        default="train",
                        type=str,
                        help="mode, train for training, test for inference.")
    parser.add_argument("--build_summary",
                        default=False,
                        type=bool,
                        help="Do not know how to use, just default.")
    parser.add_argument("--sep_attention",
                        default=True,
                        type=bool,
                        help="seq_attention for LSTM.")
    parser.add_argument("--layer_norm",
                        default=True,
                        type=bool,
                        help="layer_norm for LSTM.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="2k or 3k for longformer, 512 for others.")
    parser.add_argument("--max_program_length",
                        default=30,
                        type=int,
                        help="max program length.")
    parser.add_argument("--num_decoder_layers",
                        default=1,
                        type=int,
                        help="LSTM decoder layers")
    parser.add_argument("--n_best_size",
                        default=20,
                        type=int,
                        help="for inference")
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
                        default=300,
                        type=int,
                        help="epoch for training")
    parser.add_argument(
        "--report",
        default=300,
        type=int,
        help="iterations for reporting validation on valid set.")
    parser.add_argument("--report_loss",
                        default=100,
                        type=int,
                        help="iterations for reporting loss.")
    parser.add_argument("--max_step_ind",
                        default=11,
                        type=int,
                        help="max step size")
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
                        default=2e-5,
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
        default="FinQA/dataset/generator/",
        help="Max gradient norm.")
    parser.add_argument(
        "--examples_dir",
        type=str,
        default="FinQA/dataset/generator/",
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
    parser.add_argument("--rl",
                        action="store_true",
                        help="start consistency-based reinforcement learning.")
    parser.add_argument("--tpa",
                        action="store_true",
                        help="start TPA training")
    parser.add_argument("--tpa_methods",
                        typs=str,
                        default='switch',
                        help="choose TPA methods")
    args = parser.parse_args()
    return args


def train(args):
    args.output_dir = args.root_path + "output/"
    args.cache_dir = args.root_path + "cache/"
    args.model_save_name = args.prog_name + \
        args.pretrained_model + args.model_size + args.tags
    if args.dataset_type == "finqa":
        args.train_file = args.root_path + "dataset/FinQA/train_retrieve_output.json"
        args.dev_file = args.root_path + "dataset/FinQA/dev_retrieve_output.json"
        args.test_file = args.root_path + "dataset/FinQA/test_retrieve_output.json"
    else:
        args.train_file = args.root_path + "dataset/ConvFinQA/train_retrieve_output.json"
        args.dev_file = args.root_path + "dataset/ConvFinQA/dev_retrieve_output.json"
        args.test_file = ''
    args.output_dir = args.output_dir + "{}_setting".format(
        args.model_save_name)

    if not (args.rl and args.saved_model_path):
        logger.info("rl requires well trained generator.")

    # logger 设置
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

    reserved_token_size = len(op_list) + len(const_list)
    op_list_size = len(op_list)
    const_list_size = len(const_list)
    train_sampler = RandomSampler(
        train_features) if args.local_rank == -1 else DistributedSampler(
            train_features)
    train_dataloader = DataLoader(train_features,
                                  sampler=train_sampler,
                                  collate_fn=dataloader_collate_fn(args),
                                  batch_size=args.batch_size,
                                  num_workers=10)

    tokenizer, model = load_model(args, op_list, const_list)
    optimizer = get_optimizer(args, model, weight_decay=args.weight_decay)
    model.to(args.device)
    model.train()
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
    if args.rl:
        model.load_state_dict(torch.load(args.saved_model_path))

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
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    logger.info("***** Running training *****")
    for _ in range(args.epoch):
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, x in enumerate(epoch_iterator):
            input_ids = torch.tensor(x['input_ids']).to(args.device)
            input_mask = torch.tensor(x['input_mask']).to(args.device)
            segment_ids = torch.tensor(x['segment_ids']).to(args.device)
            program_ids = torch.tensor(x['program_ids']).to(args.device)
            program_mask = torch.tensor(x['program_mask']).to(args.device)
            option_mask = torch.tensor(x['option_mask']).to(args.device)
            examples = x['example']
            this_logits, m_list = model(True,
                                        input_ids,
                                        input_mask,
                                        segment_ids,
                                        option_mask,
                                        program_ids,
                                        program_mask,
                                        device=args.device)
            if args.rl:
                rewards = []
                cum_reward = 0.0

                choices = m_list[1]
                m_list = m_list[0]
                for idx, (logits, example, program_id) in enumerate(
                        zip(this_logits, examples, program_ids)):

                    pred_prog_ids, loss = compute_prog_from_logits(
                        logits.tolist(), args.max_program_length, example)

                    pred_prog = indices_to_prog(pred_prog_ids, example.numbers,
                                                example.number_indices,
                                                args.max_seq_length, op_list,
                                                op_list_size, const_list,
                                                const_list_size)

                    true_prog = indices_to_prog(program_id, example.numbers,
                                                example.number_indices,
                                                args.max_seq_length, op_list,
                                                op_list_size, const_list,
                                                const_list_size)

                    true_flag, true_result = eval_program(
                        true_prog, example.table)
                    invalid_flag, pred_result = eval_program(
                        pred_prog, example.table)

                    if pred_result is None or invalid_flag == 1:
                        print(pred_result)
                        print("------------------")
                        rewards.append(-2)
                    elif invalid_flag == 0 and pred_result != true_result:
                        rewards.append(-1)
                    else:
                        rewards.append(2)

                cum_reward += (sum(rewards))

                loss = model.module.reinforce_backward(choices, rewards,
                                                       m_list)
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward(torch.ones_like(loss))
                record_loss += float(loss.sum())
            else:
                this_loss = criterion(
                    this_logits.contiguous().view(-1, this_logits.shape[-1]),
                    program_ids.view(-1))
                this_loss = this_loss * program_mask.view(-1)
                this_loss = this_loss.sum() / program_mask.sum()

                if args.fp16:
                    with amp.scale_loss(this_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    this_loss.backward()

                record_loss += this_loss.item()
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
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

                if args.report > 0 and global_step % args.report == 0:
                    results_path = os.path.join(args.output_dir, "results")
                    results_path_cnt = os.path.join(results_path, 'loads',
                                                    str(global_step))
                    if is_first_worker():
                        evaluate(dev_examples,
                                 dev_features,
                                 model,
                                 results_path_cnt,
                                 'dev',
                                 tokenizer=tokenizer,
                                 reserved_token_size=reserved_token_size,
                                 op_list=op_list,
                                 const_list=const_list,
                                 args=args)
                        saved_model_path_cnt = os.path.join(
                            args.output_dir, 'loads', str(global_step))
                        os.makedirs(saved_model_path_cnt, exist_ok=True)
                        torch.save(model.state_dict(),
                                   saved_model_path_cnt + "/model.pt")
                model.train()


def evaluate(data_ori, data, model, ksave_dir, mode='dev',
             tokenizer=None, reserved_token_size=None, op_list=None, const_list=None, args=None):
    pred_list = []
    pred_unk = []

    ksave_dir_mode = os.path.join(ksave_dir, mode)
    temp_dir_mode = os.path.join(ksave_dir_mode, 'temp')
    os.makedirs(ksave_dir, exist_ok=True)
    os.makedirs(ksave_dir_mode, exist_ok=True)
    os.makedirs(temp_dir_mode, exist_ok=True)

    dev_dataloader = DataLoader(data,
                                collate_fn=dataloader_collate_fn(args),
                                batch_size=args.batch_size,
                                num_workers=10)
    k = 0
    all_results = []
    with torch.no_grad():
        model.eval()
        for x in tqdm(dev_dataloader):

            input_ids = x['input_ids']
            input_mask = x['input_mask']
            segment_ids = x['segment_ids']
            program_ids = x['program_ids']
            program_mask = x['program_mask']
            option_mask = x['option_mask']

            ori_len = len(input_ids)
            for each_item in [
                    input_ids, input_mask, segment_ids, program_ids,
                    program_mask, option_mask
            ]:
                if ori_len < args.batch_size_test:
                    each_len = len(each_item[0])
                    pad_x = [0] * each_len
                    each_item += [pad_x] * (args.batch_size_test - ori_len)

            input_ids = torch.tensor(input_ids).to(args.device)
            input_mask = torch.tensor(input_mask).to(args.device)
            segment_ids = torch.tensor(segment_ids).to(args.device)
            program_ids = torch.tensor(program_ids).to(args.device)
            program_mask = torch.tensor(program_mask).to(args.device)
            option_mask = torch.tensor(option_mask).to(args.device)

            logits, m_list = model(True,
                                   input_ids,
                                   input_mask,
                                   segment_ids,
                                   option_mask,
                                   program_ids,
                                   program_mask,
                                   device=args.device)

            for this_logit, this_id in zip(logits.tolist(), x["unique_id"]):
                all_results.append(
                    RawResult(unique_id=int(this_id),
                              logits=this_logit,
                              loss=None))

        output_prediction_file = os.path.join(ksave_dir_mode,
                                              "predictions.json")
        output_nbest_file = os.path.join(ksave_dir_mode,
                                         "nbest_predictions.json")
        output_eval_file = os.path.join(ksave_dir_mode, "full_results.json")
        output_error_file = os.path.join(ksave_dir_mode,
                                         "full_results_error.json")

        all_predictions, all_nbest = compute_predictions(
            data_ori,
            data,
            all_results,
            n_best_size=args.n_best_size,
            max_program_length=args.max_program_length,
            tokenizer=tokenizer,
            op_list=op_list,
            op_list_size=len(op_list),
            const_list=const_list,
            const_list_size=len(const_list))
        write_predictions(all_predictions, output_prediction_file)
        write_predictions(all_nbest, output_nbest_file)

        if mode == 'dev':
            exe_acc, prog_acc = evaluate_result(output_nbest_file,
                                                args.dev_file,
                                                output_eval_file,
                                                output_error_file,
                                                program_mode=args.program_mode)
        else:
            exe_acc, prog_acc = evaluate_result(output_nbest_file,
                                                args.test_file,
                                                output_eval_file,
                                                output_error_file,
                                                program_mode=args.program_mode)

        prog_res = "exe acc: " + str(exe_acc) + " prog acc: " + str(prog_acc)
        logger.info(prog_res)
        return


def inference(args):
    args.output_dir = args.root_path + "output/"
    args.cache_dir = args.root_path + "cache/"
    if args.dataset_type == "finqa":
        args.train_file = args.root_path + "dataset/FinQA/train_retrieve_output.json"
        args.dev_file = args.root_path + "dataset/FinQA/dev_retrieve_output.json"
        args.test_file = args.root_path + "dataset/FinQA/test_retrieve_output.json"
    else:
        args.train_file = args.root_path + "dataset/ConvFinQA/train_retrieve_output.json"
        args.dev_file = args.root_path + "dataset/ConvFinQA/dev_retrieve_output.json"
        args.test_file = ''
    args.model_save_name = args.prog_name + \
        args.pretrained_model + args.model_size + args.tags
    args.output_dir = args.output_dir + "{}_setting".format(
        args.model_save_name)
    args.result_dir = args.root_path + "model/generator/"
    # logger 设置
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

    with open(os.path.join(args.features_dir, 'test_features.pickle'),
              'rb') as f:
        test_features, op_list, const_list = pickle.load(f)
    with open(os.path.join(args.features_dir, 'test_examples.pickle'),
              'rb') as f:
        test_examples, op_list, const_list = pickle.load(f)

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
    # Test!
    logger.info("***** Running inference *****")
    if is_first_worker():
        generate(test_examples,
                 test_features,
                 model,
                 args.result_dir,
                 tokenizer,
                 op_list,
                 const_list,
                 mode='test')


def generate(data_ori,
             data,
             model,
             ksave_dir,
             tokenizer,
             op_list,
             const_list,
             mode='test'):
    ksave_dir_mode = os.path.join(ksave_dir, mode)
    temp_dir_mode = os.path.join(ksave_dir_mode, 'temp')
    os.makedirs(ksave_dir, exist_ok=True)
    os.makedirs(ksave_dir_mode, exist_ok=True)
    os.makedirs(temp_dir_mode, exist_ok=True)

    dev_dataloader = DataLoader(data,
                                collate_fn=dataloader_collate_fn(args),
                                batch_size=args.batch_size,
                                num_workers=10)
    k = 0
    all_results = []
    with torch.no_grad():
        model.eval()
        for x in tqdm(dev_dataloader):

            input_ids = x['input_ids']
            input_mask = x['input_mask']
            segment_ids = x['segment_ids']
            program_ids = x['program_ids']
            program_mask = x['program_mask']
            option_mask = x['option_mask']

            ori_len = len(input_ids)
            for each_item in [
                    input_ids, input_mask, segment_ids, program_ids,
                    program_mask, option_mask
            ]:
                if ori_len < args.batch_size_test:
                    each_len = len(each_item[0])
                    pad_x = [0] * each_len
                    each_item += [pad_x] * (args.batch_size_test - ori_len)

            input_ids = torch.tensor(input_ids).to(args.device)
            input_mask = torch.tensor(input_mask).to(args.device)
            segment_ids = torch.tensor(segment_ids).to(args.device)
            program_ids = torch.tensor(program_ids).to(args.device)
            program_mask = torch.tensor(program_mask).to(args.device)
            option_mask = torch.tensor(option_mask).to(args.device)

            logits, m_list = model(True,
                                   input_ids,
                                   input_mask,
                                   segment_ids,
                                   option_mask,
                                   program_ids,
                                   program_mask,
                                   device=args.device)

            for this_logit, this_id in zip(logits.tolist(), x["unique_id"]):
                all_results.append(
                    RawResult(unique_id=int(this_id),
                              logits=this_logit,
                              loss=None))

        output_prediction_file = os.path.join(ksave_dir_mode,
                                              "predictions.json")
        output_nbest_file = os.path.join(ksave_dir_mode,
                                         "nbest_predictions.json")
        output_eval_file = os.path.join(ksave_dir_mode, "full_results.json")
        output_error_file = os.path.join(ksave_dir_mode,
                                         "full_results_error.json")

        all_predictions, all_nbest = compute_predictions(
            data_ori,
            data,
            all_results,
            n_best_size=args.n_best_size,
            max_program_length=args.max_program_length,
            tokenizer=tokenizer,
            op_list=op_list,
            op_list_size=len(op_list),
            const_list=const_list,
            const_list_size=len(const_list))
        write_predictions(all_predictions, output_prediction_file)
        write_predictions(all_nbest, output_nbest_file)

        exe_acc, prog_acc = evaluate_result(output_nbest_file,
                                            args.test_file,
                                            output_eval_file,
                                            output_error_file,
                                            program_mode=args.program_mode)

        prog_res = "exe acc: " + str(exe_acc) + " prog acc: " + str(prog_acc)
        logger.info(prog_res)
        return


def tpa(args):
    args.output_dir = args.root_path + "output/"
    args.cache_dir = args.root_path + "cache/"
    args.model_save_name = args.prog_name + \
        args.pretrained_model + args.model_size + args.tags
    if args.dataset_type == "finqa":
        if args.tpa_methods == 'switch':
            args.train_file = args.root_path + "dataset/FinQA/train_TPA_Switch.json"
        elif args.tpa_methods == 'add':
            args.train_file = args.root_path + "dataset/FinQA/train_TPA_Add_Subtract.json"
        elif args.tpa_methods == 'mul':
            args.train_file = args.root_path + \
                "dataset/FinQA/train_TPA_Multiply_Divide.json.json"
        elif args.tpa_methods == 'mul-div':
            args.train_file = args.root_path + "dataset/FinQA/train_TPA_Mul-Div.json"
        args.dev_file = args.root_path + "dataset/FinQA/dev_retrieve_output.json"
        args.test_file = args.root_path + "dataset/FinQA/test_retrieve_output.json"
    else:
        if args.tpa_methods == 'switch':
            args.train_file = args.root_path + "dataset/ConvFinQA/train_TPA_Switch.json"
        elif args.tpa_methods == 'add':
            args.train_file = args.root_path + "dataset/ConvFinQA/train_TPA_Add_Subtract.json"
        elif args.tpa_methods == 'mul':
            args.train_file = args.root_path + \
                "dataset/ConvFinQA/train_TPA_Multiply_Divide.json.json"
        elif args.tpa_methods == 'mul-div':
            args.train_file = args.root_path + "dataset/ConvFinQA/train_TPA_Mul-Div.json"
        args.dev_file = args.root_path + "dataset/ConvFinQA/dev_retrieve_output.json"
        args.test_file = ''
    args.output_dir = args.output_dir + "{}_setting".format(
        args.model_save_name)

    if not (args.rl and args.saved_model_path):
        logger.info("rl requires well trained generator.")

    # logger 设置
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
    if args.tpa_methods == 'switch':
        with open(os.path.join(args.features_dir, 'train_features_switch.pickle'),
                  'rb') as f:
            train_features, op_list, const_list = pickle.load(f)
        with open(os.path.join(args.features_dir, 'dev_features_switch.pickle'),
                  'rb') as f:
            dev_features, op_list, const_list = pickle.load(f)
        with open(os.path.join(args.examples_dir, 'dev_examples_switch.pickle'),
                  'rb') as f:
            dev_examples, op_list, const_list = pickle.load(f)
    elif args.tpa_methods == 'add':
        with open(os.path.join(args.features_dir, 'train_features_add.pickle'),
                  'rb') as f:
            train_features, op_list, const_list = pickle.load(f)
        with open(os.path.join(args.features_dir, 'dev_features_add.pickle'),
                  'rb') as f:
            dev_features, op_list, const_list = pickle.load(f)
        with open(os.path.join(args.examples_dir, 'dev_examples_add.pickle'),
                  'rb') as f:
            dev_examples, op_list, const_list = pickle.load(f)
    elif args.tpa_methods == 'mul':
        with open(os.path.join(args.features_dir, 'train_features_mul.pickle'),
                  'rb') as f:
            train_features, op_list, const_list = pickle.load(f)
        with open(os.path.join(args.features_dir, 'dev_features_mul.pickle'),
                  'rb') as f:
            dev_features, op_list, const_list = pickle.load(f)
        with open(os.path.join(args.examples_dir, 'dev_examples_mul.pickle'),
                  'rb') as f:
            dev_examples, op_list, const_list = pickle.load(f)
    elif args.tpa_methods == 'mul-div':
        with open(os.path.join(args.features_dir, 'train_features_mul_div.pickle'),
                  'rb') as f:
            train_features, op_list, const_list = pickle.load(f)
        with open(os.path.join(args.features_dir, 'dev_features_mul_div.pickle'),
                  'rb') as f:
            dev_features, op_list, const_list = pickle.load(f)
        with open(os.path.join(args.examples_dir, 'dev_examples_mul_div.pickle'),
                  'rb') as f:
            dev_examples, op_list, const_list = pickle.load(f)

    reserved_token_size = len(op_list) + len(const_list)
    op_list_size = len(op_list)
    const_list_size = len(const_list)
    train_sampler = RandomSampler(
        train_features) if args.local_rank == -1 else DistributedSampler(
            train_features)
    train_dataloader = DataLoader(train_features,
                                  sampler=train_sampler,
                                  collate_fn=dataloader_collate_fn(args),
                                  batch_size=args.batch_size,
                                  num_workers=10)

    tokenizer, model = load_model(args, op_list, const_list)
    optimizer = get_optimizer(args, model, weight_decay=args.weight_decay)
    model.to(args.device)
    model.train()
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
    model.load_state_dict(torch.load(args.saved_model_path))

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
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    logger.info("***** Running training *****")
    for _ in range(args.epoch):
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, x in enumerate(epoch_iterator):
            input_ids = torch.tensor(x['input_ids']).to(args.device)
            input_mask = torch.tensor(x['input_mask']).to(args.device)
            segment_ids = torch.tensor(x['segment_ids']).to(args.device)
            program_ids = torch.tensor(x['program_ids']).to(args.device)
            program_mask = torch.tensor(x['program_mask']).to(args.device)
            option_mask = torch.tensor(x['option_mask']).to(args.device)
            examples = x['example']
            this_logits, m_list = model(True,
                                        input_ids,
                                        input_mask,
                                        segment_ids,
                                        option_mask,
                                        program_ids,
                                        program_mask,
                                        device=args.device)
            this_loss = criterion(
                this_logits.contiguous().view(-1, this_logits.shape[-1]),
                program_ids.view(-1))
            this_loss = this_loss * program_mask.view(-1)
            this_loss = this_loss.sum() / program_mask.sum()

            if args.fp16:
                with amp.scale_loss(this_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                this_loss.backward()

            record_loss += this_loss.item()
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
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

                if args.report > 0 and global_step % args.report == 0:
                    results_path = os.path.join(args.output_dir, "results")
                    results_path_cnt = os.path.join(results_path, 'loads',
                                                    str(global_step))
                    if is_first_worker():
                        evaluate(dev_examples,
                                 dev_features,
                                 model,
                                 results_path_cnt,
                                 'dev',
                                 tokenizer=tokenizer,
                                 reserved_token_size=reserved_token_size,
                                 op_list=op_list,
                                 const_list=const_list,
                                 args=args)
                        saved_model_path_cnt = os.path.join(
                            args.output_dir, 'loads', str(global_step))
                        os.makedirs(saved_model_path_cnt, exist_ok=True)
                        torch.save(model.state_dict(),
                                   saved_model_path_cnt + "/model.pt")
                model.train()


if __name__ == '__main__':
    args = get_arguments()
    set_env(args)
    if args.tpa:
        tpa(args)
    else:
        if args.mode == "train":
            train(args)
        elif args.mode == "inference":
            inference(args)
