import copy
import itertools as it
import json
import random
import os
from utils import *

dataset_type = 'finqa'
data_ori = "./train_retrieve_output.json"
root_path = "APOLLO/dataset"

with open(data_ori) as input_file:
    data = json.load(input_file)

new_data = []

for item in data:
    if dataset_type == "finqa":
        program = item['qa']['program']
    else:
        program = item['annotation']['cur_program']
    table = item['table']
    count = 0
    if 'add' in program or 'multiply' in program:
        program = program_tokenization(program)
        flag, gold_res = eval_program(program, table)
        for tokens in program:
            if tokens == "add(" or tokens == "multiply":
                count += 1
        program_list = []
        for i in range(int(len(program) / 4 + 1)):
            if i * 4 + 4 < len(program):
                program_list.append(program[i * 4:i * 4 + 4])
            else:
                program_list.append(program[i * 4:])
        program_bool = []
        for i in range(len(program_list)):
            if 'add(' in program_list[i] or 'multiply' in program_list[i]:
                if '#' not in program_list[i][1]:
                    program_bool.append(1)
            else:
                program_bool.append(0)
        program_bool = program_bool[:-1]
        s = list(it.product(range(2), repeat=program_bool.count(1)))[1:]
        for i in range(len(s)):
            program_sec = copy.deepcopy(program_list)
            index = 0
            for j in range(len(program_bool)):
                if program_bool[j] == 1:
                    if s[i][index] == 1:
                        program_sec[j] = change(program_sec[j])
                    index += 1
            prd = program_tokens_to_program(program_sec)
            tar_prog = program_tokenization(prd)
            flag, exe_res = eval_program(tar_prog, table)
            if flag == 1 or exe_res != gold_res:
                print("error!")
            document = copy.deepcopy(item)
            if dataset_type == "finqa":
                document['qa']['program'] = prd
            else:
                document['annotation']['cur_program'] = prd
            new_data.append(document)

print(len(new_data))

if dataset_type == "finqa":
    output_prediction_file = os.path.join(
        root_path, "FinQA", "train_TPA_Switch.json")
else:
    output_prediction_file = os.path.join(
        root_path, "ConvFinQA", "train_TPA_Switch.json")

new_train = new_data + data
random.shuffle(new_train)
with open(output_prediction_file, "w") as f:
    json.dump(new_train, f, indent=4)
