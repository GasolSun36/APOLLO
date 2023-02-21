import collections
import tqdm
import math
import numpy as np
import json
import sys
from sympy import simplify
import torch.utils.data as torchdata

from utils.general_utils import *

RawResult = collections.namedtuple("RawResult", "unique_id logits loss")
all_ops = [
    "add", "subtract", "multiply", "divide", "exp", "greater", "table_max",
    "table_min", "table_sum", "table_average"
]


def read_txt(input_path):
    """Read a txt file into a list."""
    with open(input_path) as input_file:
        input_data = input_file.readlines()
    items = []
    for line in input_data:
        items.append(line.strip())
    return items


class Examples_dataset(torchdata.Dataset):

    def __init__(self, input_data, tokenizer, retrieve_mode, dataset_type,
                 max_seq_length, program_mode):
        self.input_data = input_data
        self.tokenizer = tokenizer
        self.retrieve_mode = retrieve_mode
        self.dataset_type = dataset_type
        self.max_seq_length = max_seq_length
        self.program_mode = program_mode

    def __getitem__(self, index: int):
        entry = self.input_data[index]
        example = read_mathqa_entry(entry, self.tokenizer, self.retrieve_mode,
                                    self.dataset_type, self.max_seq_length,
                                    self.program_mode)

        return entry, example

    def __len__(self):
        return len(self.input_data)

    @staticmethod
    def collect_fn():

        def sub_collect_fn(features):
            return [feature[0] for feature in features
                    ], [feature[1] for feature in features]

        return sub_collect_fn


def read_examples(input_path, tokenizer, op_list, const_list, retrieve_mode,
                  dataset_type, max_seq_length, program_mode):
    """Read a json file into a list of examples."""
    with open(input_path) as input_file:
        input_data = json.load(input_file)

    d = Examples_dataset(input_data, tokenizer, retrieve_mode, dataset_type,
                         max_seq_length, program_mode)
    train_dataloader = torchdata.DataLoader(d,
                                            batch_size=50,
                                            num_workers=20,
                                            collate_fn=d.collect_fn())
    new_input_data = []
    new_examples = []
    for step, batch in enumerate(tqdm(train_dataloader)):
        new_input_data.extend(batch[0])
        new_examples.extend(batch[1])
    return new_input_data, new_examples, op_list, const_list


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 question,
                 input_ids,
                 input_mask,
                 option_mask,
                 segment_ids,
                 options,
                 answer=None,
                 program=None,
                 program_ids=None,
                 program_weight=None,
                 program_mask=None,
                 example=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.question = question
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.option_mask = option_mask
        self.segment_ids = segment_ids
        self.options = options
        self.answer = answer
        self.program = program
        self.program_ids = program_ids
        self.program_weight = program_weight
        self.program_mask = program_mask
        self.example = example


class MathQAExample(
        collections.namedtuple(
            "MathQAExample",
            "id original_question question_tokens options answer \
        numbers number_indices original_program program table")):

    def convert_single_example(self, *args, **kwargs):
        return convert_single_mathqa_example(self, *args, **kwargs)


def read_mathqa_entry(entry, tokenizer, retrieve_mode, dataset_type,
                      max_seq_length, program_mode):
    if dataset_type == 'finqa':
        question = entry["qa"]["question"]
        this_id = entry["id"]
        context = ""

        if retrieve_mode == "single":
            for ind, each_sent in entry["model_input"]:
                context += each_sent
                context += " "
        elif retrieve_mode == "gold":
            for each_con in entry["qa"]["gold_inds"]:
                context += entry["qa"]["gold_inds"][each_con]
                context += " "
        elif retrieve_mode == "none":
            # no retriever, use longformer
            table = entry["table"]
            table_text = ""
            for row in table[1:]:
                this_sent = table_row_to_text(table[0], row)
                table_text += this_sent

            context = " ".join(
                entry["pre_text"]) + " " + table_text + " " + " ".join(
                    entry["post_text"])

        context = context.strip()
        # process "." and "*" in text
        context = context.replace(". . . . . .", "")
        context = context.replace("* * * * * *", "")
        original_question = question + " " + tokenizer.sep_token + " " + context.strip(
        )

        if "exe_ans" in entry["qa"]:
            options = entry["qa"]["exe_ans"]
        else:
            options = None

        table = entry['table']

        original_question_tokens = original_question.split(' ')
        numbers = []
        number_indices = []
        question_tokens = []
        for i, tok in enumerate(original_question_tokens):
            num = str_to_num(tok)
            if num is not None:
                numbers.append(tok)
                number_indices.append(len(question_tokens))
                if tok[0] == '.':
                    numbers.append(str(str_to_num(tok[1:])))
                    number_indices.append(len(question_tokens) + 1)

            tok_proc = tokenize(tokenizer, tok, dataset_type)
            question_tokens.extend(tok_proc)
            if len(question_tokens) > max_seq_length:
                break

        if "exe_ans" in entry["qa"]:
            answer = entry["qa"]["exe_ans"]
        else:
            answer = None

        # table headers
        for row in entry["table"]:
            tok = row[0]
            if tok and tok in original_question:
                numbers.append(tok)
                tok_index = original_question.index(tok)
                prev_tokens = original_question[:tok_index]
                number_indices.append(
                    len(tokenize(tokenizer, prev_tokens, dataset_type)) + 1)

        if program_mode == "seq":
            if 'program' in entry["qa"]:
                original_program = entry["qa"]['program']
                program = program_tokenization(original_program)
            else:
                program = None
                original_program = None

        elif program_mode == "nest":
            if 'program_re' in entry["qa"]:
                original_program = entry["qa"]['program_re']
                program = program_tokenization(original_program)
            else:
                program = None
                original_program = None
        else:
            program = None
            original_program = None
    else:
        question = " ".join(entry["annotation"]["cur_dial"])
        this_id = entry["id"]
        context = ""

        if retrieve_mode == "single":
            for ind, each_sent in entry["annotation"]["model_input"]:
                context += each_sent
                context += " "
        elif retrieve_mode == "gold":
            for each_con in entry["annotation"]["gold_ind"]:
                context += entry["annotation"]["gold_ind"][each_con]
                context += " "

        elif retrieve_mode == "none":
            # no retriever, use longformer
            table = entry["table"]
            table_text = ""
            for row in table[1:]:
                this_sent = table_row_to_text(table[0], row)
                table_text += this_sent

            context = " ".join(entry["pre_text"]) + " " + \
                " ".join(entry["post_text"]) + " " + table_text

        context = context.strip()
        # process "." and "*" in text
        context = context.replace(". . . . . .", "")
        context = context.replace("* * * * * *", "")

        original_question = question + " " + tokenizer.sep_token + " " + context.strip(
        )

        if "exe_ans" in entry["annotation"]:
            options = entry["annotation"]["exe_ans"]
        else:
            options = None

        original_question_tokens = original_question.split(' ')

        numbers = []
        number_indices = []
        question_tokens = []
        for i, tok in enumerate(original_question_tokens):
            num = str_to_num(tok)
            if num is not None:
                numbers.append(tok)
                number_indices.append(len(question_tokens))
                if tok[0] == '.':
                    numbers.append(str(str_to_num(tok[1:])))
                    number_indices.append(len(question_tokens) + 1)
            tok_proc = tokenize(tokenizer, tok, dataset_type)
            question_tokens.extend(tok_proc)

        if "exe_ans" in entry["annotation"]:
            answer = entry["annotation"]["exe_ans"]
        else:
            answer = None

        # table headers
        for row in entry["table"]:
            tok = row[0]
            if tok and tok in original_question:
                numbers.append(tok)
                tok_index = original_question.index(tok)
                prev_tokens = original_question[:tok_index]
                number_indices.append(
                    len(tokenize(tokenizer, prev_tokens, dataset_type)) + 1)

        if program_mode == "seq":
            if 'cur_program' in entry["annotation"]:
                original_program = entry["annotation"]['cur_program']
                program = program_tokenization(original_program)
            else:
                program = None
                original_program = None
        elif program_mode == "nest":
            if 'program_re' in entry["annotation"]:
                original_program = entry["annotation"]['cur_program_re']
                program = program_tokenization(original_program)
            else:
                program = None
                original_program = None
        else:
            program = None
    return MathQAExample(id=this_id,
                         original_question=original_question,
                         question_tokens=question_tokens,
                         options=options,
                         answer=answer,
                         numbers=numbers,
                         number_indices=number_indices,
                         original_program=original_program,
                         program=program,
                         table=table)


def prog_token_to_indices(prog, numbers, number_indices, max_seq_length,
                          op_list, op_list_size, const_list, const_list_size):
    prog_indices = []
    for i, token in enumerate(prog):
        if token in op_list:
            prog_indices.append(op_list.index(token))
        elif token in const_list:
            prog_indices.append(op_list_size + const_list.index(token))
        else:
            if token in numbers:
                cur_num_idx = numbers.index(token)
            else:
                cur_num_idx = -1
                for num_idx, num in enumerate(numbers):
                    if str_to_num(num) == str_to_num(token):
                        cur_num_idx = num_idx
                        break
            # print(prog)
            # print(token)
            # print(const_list)
            # print(numbers)
            if cur_num_idx == -1:
                print(prog)
                print(numbers)
                print(number_indices)
            # assert cur_num_idx != -1
            prog_indices.append(op_list_size + const_list_size +
                                number_indices[cur_num_idx])
    return prog_indices


def convert_single_mathqa_example(example, is_training, tokenizer,
                                  max_seq_length, max_program_length, op_list,
                                  op_list_size, const_list, const_list_size,
                                  cls_token, sep_token):
    """Converts a single MathQAExample into an InputFeature."""
    features = []
    question_tokens = example.question_tokens
    if len(question_tokens) > max_seq_length - 2:
        print("too long")
        question_tokens = question_tokens[:max_seq_length - 2]
    tokens = [cls_token] + question_tokens + [sep_token]
    query_seq_pos = tokens.index(sep_token)
    segment_ids = [0] * len(tokens)
    # segment_ids[0:query_seq_pos + 1] = [1] * (query_seq_pos + 1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)
    for ind, offset in enumerate(example.number_indices):
        if offset < len(input_mask):
            input_mask[offset] = 2
        else:
            if is_training == True:
                # invalid example, drop for training
                return features

            # assert is_training == False

    padding = [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))
    padding_mask = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding_mask)
    segment_ids.extend(padding_mask)

    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    number_mask = [tmp - 1 for tmp in input_mask]
    for ind in range(len(number_mask)):
        if number_mask[ind] < 0:
            number_mask[ind] = 0
    option_mask = [1, 0, 0, 1] + [1] * (len(op_list) + len(const_list) - 4)
    option_mask = option_mask + number_mask
    option_mask = [float(tmp) for tmp in option_mask]

    for ind in range(len(input_mask)):
        if input_mask[ind] > 1:
            input_mask[ind] = 1

    numbers = example.numbers
    number_indices = example.number_indices
    program = example.program
    if program is not None and is_training:
        program_ids = prog_token_to_indices(program, numbers, number_indices,
                                            max_seq_length, op_list,
                                            op_list_size, const_list,
                                            const_list_size)
        program_mask = [1] * len(program_ids)
        program_ids = program_ids[:max_program_length]
        program_mask = program_mask[:max_program_length]
        if len(program_ids) < max_program_length:
            padding = [0] * (max_program_length - len(program_ids))
            program_ids.extend(padding)
            program_mask.extend(padding)
    else:
        program = ""
        program_ids = [0] * max_program_length
        program_mask = [0] * max_program_length
    assert len(program_ids) == max_program_length
    assert len(program_mask) == max_program_length
    features.append(
        InputFeatures(unique_id=-1,
                      example_index=-1,
                      tokens=tokens,
                      question=example.original_question,
                      input_ids=input_ids,
                      input_mask=input_mask,
                      option_mask=option_mask,
                      segment_ids=segment_ids,
                      options=example.options,
                      answer=example.answer,
                      program=program,
                      program_ids=program_ids,
                      program_weight=1.0,
                      program_mask=program_mask,
                      example=example))
    return features


class Feature_dataset(torchdata.Dataset):

    def __init__(self, examples, is_training, tokenizer, max_seq_length,
                 max_program_length, op_list, op_list_size, const_list,
                 const_list_size, cls_token, sep_token):
        self.examples = examples
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_program_length = max_program_length
        self.op_list = op_list
        self.op_list_size = op_list_size
        self.const_list = const_list
        self.const_list_size = const_list_size
        self.cls_token = cls_token
        self.sep_token = sep_token

    def __getitem__(self, index: int):
        example = self.examples[index]
        features = example.convert_single_example(
            is_training=self.is_training,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            max_program_length=self.max_program_length,
            op_list=self.op_list,
            op_list_size=self.op_list_size,
            const_list=self.const_list,
            const_list_size=self.const_list_size,
            cls_token=self.tokenizer.cls_token,
            sep_token=self.tokenizer.sep_token)
        res = []
        unique_id = 1000000000 + index
        for feature in features:
            feature.unique_id = unique_id
            feature.example_index = index
            res.append(feature)
        return res

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collect_fn():

        def sub_collect_fn(features):
            res = []
            for feature in features:
                res.extend(feature)
            return res

        return sub_collect_fn


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 max_program_length,
                                 is_training,
                                 op_list,
                                 op_list_size,
                                 const_list,
                                 const_list_size,
                                 verbose=True):
    """Converts a list of DropExamples into InputFeatures."""
    feature_dataset = Feature_dataset(examples=examples,
                                      is_training=is_training,
                                      tokenizer=tokenizer,
                                      max_seq_length=max_seq_length,
                                      max_program_length=max_program_length,
                                      op_list=op_list,
                                      op_list_size=op_list_size,
                                      const_list=const_list,
                                      const_list_size=const_list_size,
                                      cls_token=tokenizer.cls_token,
                                      sep_token=tokenizer.sep_token)
    dataloader = torchdata.DataLoader(feature_dataset,
                                      batch_size=50,
                                      num_workers=20,
                                      collate_fn=feature_dataset.collect_fn())
    res = []
    for step, batch in enumerate(tqdm(dataloader)):
        res.extend(batch)
    return res


def write_predictions(all_predictions, output_prediction_file):
    """Writes final predictions in json format."""

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def compute_prog_from_logits(logits,
                             max_program_length,
                             example,
                             template=None):
    pred_prog_ids = []
    op_stack = []
    loss = 0
    for cur_step in range(max_program_length):
        cur_logits = logits[cur_step]
        cur_pred_softmax = _compute_softmax(cur_logits)
        cur_pred_token = np.argmax(cur_logits)
        loss -= np.log(cur_pred_softmax[cur_pred_token])
        pred_prog_ids.append(cur_pred_token)
        if cur_pred_token == 0:
            break
    return pred_prog_ids, loss


def indices_to_prog(program_indices, numbers, number_indices, max_seq_length,
                    op_list, op_list_size, const_list, const_list_size):
    prog = []
    for i, prog_id in enumerate(program_indices):
        if prog_id < op_list_size:
            prog.append(op_list[prog_id])
        elif prog_id < op_list_size + const_list_size:
            prog.append(const_list[prog_id - op_list_size])
        else:
            prog.append(numbers[number_indices.index(prog_id - op_list_size -
                                                     const_list_size)])
    return prog


def compute_predictions(all_examples, all_features, all_results, n_best_size,
                        max_program_length, tokenizer, op_list, op_list_size,
                        const_list, const_list_size):
    """Computes final predictions based on logits."""
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "logits"])

    all_predictions = collections.OrderedDict()
    all_predictions["pred_programs"] = collections.OrderedDict()
    all_predictions["ref_programs"] = collections.OrderedDict()
    all_nbest = collections.OrderedDict()
    for (example_index, example) in tqdm(enumerate(all_examples)):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            logits = result.logits
            prelim_predictions.append(
                _PrelimPrediction(feature_index=feature_index, logits=logits))

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", "options answer program_ids program")

        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            program = example.program
            pred_prog_ids, loss = compute_prog_from_logits(
                pred.logits, max_program_length, example)
            pred_prog = indices_to_prog(pred_prog_ids, example.numbers,
                                        example.number_indices,
                                        conf.max_seq_length, op_list,
                                        op_list_size, const_list,
                                        const_list_size)
            nbest.append(
                _NbestPrediction(options=example.options,
                                 answer=example.answer,
                                 program_ids=pred_prog_ids,
                                 program=pred_prog))

        assert len(nbest) >= 1

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["id"] = example.id
            output["options"] = entry.options
            output["ref_answer"] = entry.answer
            output["pred_prog"] = [str(prog) for prog in entry.program]
            output["ref_prog"] = example.program
            output["question_tokens"] = example.question_tokens
            output["numbers"] = example.numbers
            output["number_indices"] = example.number_indices
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions["pred_programs"][example_index] = nbest_json[0][
            "pred_prog"]
        all_predictions["ref_programs"][example_index] = nbest_json[0][
            "ref_prog"]
        all_nbest[example_index] = nbest_json

    return all_predictions, all_nbest
