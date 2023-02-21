import json
import sys
import tqdm
import collections

from utils.general_utils import *


def read_txt(input_path):
    """Read a txt file into a list."""

    with open(input_path) as input_file:
        input_data = input_file.readlines()
    items = []
    for line in input_data:
        items.append(line.strip())
    return items


def read_examples(input_path, tokenizer, op_list, const_list, dataset_type):
    """Read a json file into a list of examples."""

    with open(input_path) as input_file:
        input_data = json.load(input_file)

    examples = []
    for entry in input_data:
        examples.append(read_mathqa_entry(entry, tokenizer, dataset_type))

    return input_data, examples, op_list, const_list


def read_mathqa_entry(entry, tokenizer, dataset_type):
    filename_id = entry["id"]
    if dataset_type == 'finqa':
        question = entry["qa"]["question"]
        if "gold_inds" in entry["qa"]:
            all_positive = entry["qa"]["gold_inds"]
        else:
            all_positive = []
    else:
        question = " ".join(entry["annotation"]["cur_dial"])
        if "gold_ind" in entry["annotation"]:
            all_positive = entry["annotation"]["gold_ind"]
        else:
            all_positive = []

    pre_text = entry["pre_text"]
    post_text = entry["post_text"]
    table = entry["table"]
    table_sents = entry['qa']['table_sents']
    return MathQAExample(filename_id=filename_id,
                         question=question,
                         all_positive=all_positive,
                         pre_text=pre_text,
                         post_text=post_text,
                         table=table,
                         table_sents=table_sents)


class MathQAExample(
        collections.namedtuple(
            "MathQAExample", "filename_id question all_positive \
        pre_text post_text table table_sents")):

    def convert_single_example(self, *args, **kwargs):
        return convert_single_mathqa_example(self, *args, **kwargs)


def convert_single_mathqa_example(example, is_training, tokenizer,
                                  max_seq_length, cls_token, sep_token,
                                  pretrained_model):
    """Converts a single MathQAExample into Multiple Retriever Features."""
    """ option: tf idf or all"""
    """train: 1:3 pos neg. Test: all"""

    pos_features = []
    features_neg = []

    question = example.question
    all_text = example.pre_text + example.post_text

    if is_training:
        for gold_ind in example.all_positive:
            this_gold_sent = example.all_positive[gold_ind]
            this_input_feature = wrap_single_pair(tokenizer, question,
                                                  this_gold_sent, 1,
                                                  max_seq_length, cls_token,
                                                  sep_token, pretrained_model)

            this_input_feature["filename_id"] = example.filename_id
            this_input_feature["ind"] = gold_ind
            pos_features.append(this_input_feature)

        pos_text_ids = []
        pos_table_ids = []
        for gold_ind in example.all_positive:
            if "text" in gold_ind:
                pos_text_ids.append(int(gold_ind.replace("text_", "")))
            elif "table" in gold_ind:
                pos_table_ids.append(int(gold_ind.replace("table_", "")))

        # test: all negs
        # text
        for i in range(len(all_text)):
            if i not in pos_text_ids:
                this_text = all_text[i]
                this_input_feature = wrap_single_pair(
                    tokenizer, example.question, this_text, 0, max_seq_length,
                    cls_token, sep_token, pretrained_model)
                this_input_feature["filename_id"] = example.filename_id
                this_input_feature["ind"] = "text_" + str(i)
                features_neg.append(this_input_feature)
        # table
        for this_table_id in range(len(example.table_sents)):
            if this_table_id not in pos_table_ids:
                this_input_feature = wrap_single_pair(
                    tokenizer, example.question,
                    example.table_sents[this_table_id], 0, max_seq_length,
                    cls_token, sep_token, pretrained_model)
                this_input_feature["filename_id"] = example.filename_id
                this_input_feature["ind"] = "table_" + str(this_table_id)
                features_neg.append(this_input_feature)

    else:
        pos_features = []
        features_neg = []
        question = example.question

        # set label as -1 for test examples
        for i in range(len(all_text)):
            this_text = all_text[i]
            this_input_feature = wrap_single_pair(tokenizer, example.question,
                                                  this_text, -1,
                                                  max_seq_length, cls_token,
                                                  sep_token, pretrained_model)
            this_input_feature["filename_id"] = example.filename_id
            this_input_feature["ind"] = "text_" + str(i)
            features_neg.append(this_input_feature)
        # table
        for this_table_id in range(len(example.table_sents)):
            this_input_feature = wrap_single_pair(
                tokenizer, example.question,
                example.table_sents[this_table_id], -1, max_seq_length,
                cls_token, sep_token, pretrained_model)
            this_input_feature["filename_id"] = example.filename_id
            this_input_feature["ind"] = "table_" + str(this_table_id)
            features_neg.append(this_input_feature)

    return pos_features, features_neg


def wrap_single_pair(tokenizer, question, context, label, max_seq_length,
                     cls_token, sep_token, pretrained_model):
    '''
    single pair of question, context, label feature
    '''

    question_tokens = tokenize(tokenizer, question, pretrained_model)
    this_gold_tokens = tokenize(tokenizer, context, pretrained_model)

    tokens = [cls_token] + question_tokens + [sep_token]
    segment_ids = [0] * len(tokens)

    tokens += this_gold_tokens
    segment_ids.extend([0] * len(this_gold_tokens))

    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length - 1]
        tokens += [sep_token]
        segment_ids = segment_ids[:max_seq_length]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    this_input_feature = {
        "context": context,
        "tokens": tokens,
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "label": label
    }

    return this_input_feature


def convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    is_training,
    pretrained_model,
):
    """Converts a list of DropExamples into InputFeatures."""
    res = []
    res_neg = []
    for (example_index, example) in tqdm(enumerate(examples)):
        features, features_neg = example.convert_single_example(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            is_training=is_training,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            pretrained_model=pretrained_model)

        res.extend(features)
        res_neg.extend(features_neg)

    return res, res_neg


def retrieve_evaluate(all_logits, all_filename_ids, all_inds,
                      output_prediction_file, ori_file, topn):
    '''
    save results to file. calculate recall
    '''

    res_filename = {}
    res_filename_inds = {}

    for this_logit, this_filename_id, this_ind in zip(all_logits,
                                                      all_filename_ids,
                                                      all_inds):

        if this_filename_id not in res_filename:
            res_filename[this_filename_id] = []
            res_filename_inds[this_filename_id] = []
        if this_ind not in res_filename_inds[this_filename_id]:
            res_filename[this_filename_id].append({
                "score": this_logit[1],
                "ind": this_ind
            })
            res_filename_inds[this_filename_id].append(this_ind)

    with open(ori_file) as f:
        data_all = json.load(f)

    all_recall = 0.0
    all_recall_3 = 0.0
    all_recall_8 = 0.0
    all_recall_10 = 0.0
    all_pre_3 = 0.0
    all_pre_5 = 0.0
    for data in data_all:
        this_filename_id = data["id"]

        this_res = res_filename[this_filename_id]

        sorted_dict = sorted(this_res,
                             key=lambda kv: kv["score"],
                             reverse=True)

        gold_inds = data["qa"]["gold_inds"]

        # table rows
        table_retrieved = []
        text_retrieved = []

        # all retrieved
        table_re_all = []
        text_re_all = []

        correct = 0  # top5
        correct_3 = 0
        correct_8 = 0
        correct_10 = 0

        for tmp in sorted_dict[:topn]:
            if "table" in tmp["ind"]:
                table_retrieved.append(tmp)
            else:
                text_retrieved.append(tmp)

            if tmp["ind"] in gold_inds:
                correct += 1

        for tmp in sorted_dict:
            if "table" in tmp["ind"]:
                table_re_all.append(tmp)
            else:
                text_re_all.append(tmp)

        for tmp in sorted_dict[:3]:
            if tmp["ind"] in gold_inds:
                correct_3 += 1
        for tmp in sorted_dict[:8]:
            if tmp["ind"] in gold_inds:
                correct_8 += 1
        for tmp in sorted_dict[:10]:
            if tmp["ind"] in gold_inds:
                correct_10 += 1

        all_recall += (float(correct) / min(5, len(gold_inds)))
        all_recall_3 += (float(correct_3) / min(3, len(gold_inds)))
        all_recall_8 += (float(correct_8) / min(8, len(gold_inds)))
        all_recall_10 += (float(correct_10) / min(10, len(gold_inds)))
        all_pre_3 += (float(correct_3) / 3)
        all_pre_5 += (float(correct) / 5)

        data["table_retrieved"] = table_retrieved
        data["text_retrieved"] = text_retrieved

        data["table_retrieved_all"] = table_re_all
        data["text_retrieved_all"] = text_re_all

    with open(output_prediction_file, "w") as f:
        json.dump(data_all, f, indent=4)

    res_3 = all_recall_3 / len(data_all)
    res = all_recall / len(data_all)
    res_8 = all_recall_8 / len(data_all)
    res_10 = all_recall_10 / len(data_all)
    pre_3 = all_pre_3 / len(data_all)
    pre_5 = all_pre_5 / len(data_all)

    res = "Top 3: " + str(res_3) + "\n" + "Top 5: " + str(
        res) + "\n" + "Top 8: " + str(res_8) + "\n" + "Top 10: " + str(
            res_10) + "\n" + "precision3" + str(
                pre_3) + "\n" + "precision5" + str(pre_5) + "\n"

    return res


def convert_train(json_in, json_out, topn, max_len, dataset_type):
    with open(json_in) as f_in:
        data = json.load(f_in)

    for each_data in data:
        table_retrieved = each_data["table_retrieved"]
        text_retrieved = each_data["text_retrieved"]

        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]
        all_text = pre_text + post_text

        gold_inds = each_data["qa"]["gold_inds"]

        table = each_data["table"]

        all_retrieved = each_data["table_retrieved"] + \
            each_data["text_retrieved"]

        false_retrieved = []
        for tmp in all_retrieved:
            if tmp["ind"] not in gold_inds:
                false_retrieved.append(tmp)

        sorted_dict = sorted(false_retrieved,
                             key=lambda kv: kv["score"],
                             reverse=True)

        acc_len = 0
        all_text_in = {}
        all_table_in = {}

        for tmp in gold_inds:
            if "table" in tmp:
                all_table_in[tmp] = gold_inds[tmp]
            else:
                all_text_in[tmp] = gold_inds[tmp]

        context = ""
        for tmp in gold_inds:
            context += gold_inds[tmp]

        acc_len = len(context.split(" "))

        for tmp in sorted_dict:
            if len(all_table_in) + len(all_text_in) >= topn:
                break
            this_sent_ind = int(tmp["ind"].split("_")[1])

            if "table" in tmp["ind"]:
                this_sent = table_row_to_text(table[0], table[this_sent_ind])
            else:
                this_sent = all_text[this_sent_ind]

            if acc_len + len(this_sent.split(" ")) < max_len:
                if "table" in tmp["ind"]:
                    all_table_in[tmp["ind"]] = this_sent
                else:
                    all_text_in[tmp["ind"]] = this_sent

                acc_len += len(this_sent.split(" "))
            else:
                break

        this_model_input = []

        # original_order
        sorted_dict_table = sorted(all_table_in.items(),
                                   key=lambda kv: int(kv[0].split("_")[1]))
        sorted_dict_text = sorted(all_text_in.items(),
                                  key=lambda kv: int(kv[0].split("_")[1]))

        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) < len(pre_text):
                this_model_input.append(tmp)

        for tmp in sorted_dict_table:
            this_model_input.append(tmp)

        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) >= len(pre_text):
                this_model_input.append(tmp)

        if dataset_type == 'finqa':
            each_data["qa"]["model_input"] = this_model_input
        else:
            each_data["annotation"]["model_input"] = this_model_input

    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)


def convert_test(json_in, json_out, topn, max_len, dataset_type):
    with open(json_in) as f_in:
        data = json.load(f_in)

    for each_data in data:
        table_retrieved = each_data["table_retrieved"]
        text_retrieved = each_data["text_retrieved"]

        pre_text = each_data["pre_text"]
        post_text = each_data["post_text"]
        all_text = pre_text + post_text

        table_sents = each_data['qa']['table_sents']
        all_retrieved = each_data["table_retrieved"] + \
            each_data["text_retrieved"]

        sorted_dict = sorted(all_retrieved,
                             key=lambda kv: kv["score"],
                             reverse=True)

        acc_len = 0
        all_text_in = {}
        all_table_in = {}

        for tmp in sorted_dict:
            if len(all_table_in) + len(all_text_in) >= topn:
                break
            this_sent_ind = int(tmp["ind"].split("_")[1])

            if "table" in tmp["ind"]:
                this_sent = table_sents[this_sent_ind]
            else:
                this_sent = all_text[this_sent_ind]

            if acc_len + len(this_sent.split(" ")) < max_len:
                if "table" in tmp["ind"]:
                    all_table_in[tmp["ind"]] = this_sent
                else:
                    all_text_in[tmp["ind"]] = this_sent

                acc_len += len(this_sent.split(" "))
            else:
                break

        this_model_input = []

        # original_order
        sorted_dict_table = sorted(all_table_in.items(),
                                   key=lambda kv: int(kv[0].split("_")[1]))
        sorted_dict_text = sorted(all_text_in.items(),
                                  key=lambda kv: int(kv[0].split("_")[1]))

        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) < len(pre_text):
                this_model_input.append(tmp)

        for tmp in sorted_dict_table:
            this_model_input.append(tmp)

        for tmp in sorted_dict_text:
            if int(tmp[0].split("_")[1]) >= len(pre_text):
                this_model_input.append(tmp)

        if dataset_type == 'finqa':
            each_data["qa"]["model_input"] = this_model_input
        else:
            each_data["annotation"]["model_input"] = this_model_input

    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)
