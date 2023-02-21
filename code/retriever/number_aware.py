from utils.general_utils import *
import json
import re
import os
import sys

root_path = "APOLLO/dataset"
dataset_type = 'finqa'
if dataset_type == 'finqa':
    train_file = os.path.join(root_path, "FinQA", "train.json")
    train_file_na = os.path.join(root_path, "FinQA", "train_number_aware.json")
else:
    train_file = os.path.join(root_path, "ConvFinQA", "train.json")
    train_file_na = os.path.join(root_path, "FinQA", "train_number_aware.json")

with open(train_file) as input_file:
    train_data = json.load(input_file)


def check_if_in_table_head(table, text):
    for row in table:
        tok = row[0]
        if tok and tok.lower() in text:
            return True
    return False


def find_in_table(table_sents, sent):
    for i in range(len(table_sents)):
        if table_sents[i].replace(" ", "").lower() == sent.replace(" ", "").lower():
            return i
    return -1


for item in train_data:
    id = item['id']
    pre = item['pre_text']
    post = item['post_text']
    if dataset_type == 'finqa':
        gold = item['qa']['gold_inds']
    else:
        gold = item['annotation']['gold_ind']
    table = item['table']
    pre_text = []
    post_text = []
    gold_inds = {}
    gold_list = []
    for i in range(len(pre)):
        if len(pre[i]) < 2:
            continue
        pre_text.append(pre[i])
    for i in range(len(post)):
        if len(post[i]) < 2:
            continue
        post_text.append(post[i])
    for tmp in gold:
        if bool(re.search(r'\d', gold[tmp])) == False:
            if check_if_in_table_head(table, gold[tmp]):
                gold_inds[tmp] = gold[tmp]
                gold_list.append(gold[tmp])
            else:
                print(id)
        else:
            gold_inds[tmp] = gold[tmp]
            gold_list.append(gold[tmp])
    if dataset_type == 'finqa':
        item['qa']['gold_inds'] = gold_inds
    else:
        item['annotation']['gold_ind'] = gold_inds

print("pre process complete...")

count = 0
for item in train_data:
    id = item['id']
    pre = item['pre_text']
    post = item['post_text']
    if dataset_type == 'finqa':
        gold_inds = item['qa']['gold_inds']
    else:
        gold_inds = item['annotation']['gold_ind']
    table = item['table']
    table_sents = []
    all_text = pre + post
    gold_inds_revise = {}
    for i in range(len(table)):
        this_table_line = table_row_to_text(table[0], table[i])
        table_sents.append(this_table_line)
    table_sents_revise = []
    for i in table_sents:
        if i not in table_sents_revise:
            table_sents_revise.append(i)
    if len(table_sents) != len(table_sents_revise):
        count += 1
    for tmp in gold_inds:
        if "text" in tmp:
            ind = int(tmp.replace("text_", ""))
            index = all_text.index(gold_inds[tmp])
            if index == ind:
                string = "text_" + str(index)
                gold_inds_revise[string] = gold_inds[tmp]
            elif index != ind:
                print("Text id need to be revised:")
                string = "text_" + str(index)
                print("Before revised:{}".format(ind))
                print("After revised:{}".format(
                    int(string.replace("text_", ""))))
                gold_inds_revise[string] = gold_inds[tmp]
        else:
            ind = int(tmp.replace("table_", ""))
            if find_in_table(table_sents_revise, gold_inds[tmp]) != -1:
                index = find_in_table(table_sents_revise, gold_inds[tmp])
                if index == ind:
                    string = "table_" + str(index)
                    gold_inds_revise[string] = gold_inds[tmp]
                elif index != ind:
                    print("Table id need to be revised:")
                    print(id)
                    string = "table_" + str(index)
                    print("Before revised:{}".format(ind))
                    print("After revised:{}".format(
                        int(string.replace("table_", ""))))
                    gold_inds_revise[string] = gold_inds[tmp]
            else:
                print("Wrong revised!")
    if len(gold_inds_revise) == 0:
        print("there is no gold")
        print(id)
    item['table_sents'] = table_sents_revise
    if dataset_type == 'finqa':
        item['qa']['gold_inds'] = gold_inds_revise
    else:
        item['annotation']['gold_ind'] = gold_inds_revise

with open(train_file_na, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=4)
print("--------------------------------------------")
print(count)
