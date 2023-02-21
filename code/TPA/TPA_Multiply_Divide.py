import copy
import json
import random
import os
from utils import *

dataset_type = 'finqa'
data_ori = "./train_retrieve_output.json"
root_path = "APOLLO/dataset"

with open(data_ori) as input_file:
    data = json.load(input_file)

num_list = [
    'const_2', 'const_1', 'const_3', 'const_4', 'const_5', 'const_6',
    'const_7', 'const_8', 'const_9', 'const_10'
]

new_data = []

count = 0
for item in data:
    if dataset_type == "finqa":
        program = item['qa']['program']
    else:
        program = item['annotation']['cur_program']
    table = item['table']
    id = item['id']
    if 'greater' in program or 'table_max' in program or 'table_min' in program or 'table_sum' in program or 'table_average' in program:
        continue
    program = program_tokenization(program)
    flag, gold_res = eval_program(program, table)
    program_list = []
    for i in range(int(len(program) / 4 + 1)):
        if i * 4 + 4 < len(program):
            program_list.append(program[i * 4:i * 4 + 4])
        else:
            program_list.append(program[i * 4:])
    index = len(program_list[:-1]) - 1
    num = random.choice(num_list)
    program_sec = create_program('multiply(', index, num)
    program_list.insert(-1, program_sec)
    program_sec = create_program('divide(', index + 1, num)
    program_list.insert(-1, program_sec)
    prd = program_tokens_to_program(program_list)
    tar_prog = program_tokenization(prd)
    flag, exe_res = eval_program(tar_prog, table)
    if flag == 1 or exe_res != gold_res:
        print("error!")
        count += 1
    elif flag == 0 and exe_res == gold_res:
        document = copy.deepcopy(item)
        if dataset_type == "finqa":
            document['qa']['program'] = prd
        else:
            document['annotation']['cur_program'] = prd
        new_data.append(document)
print(count)
print(len(new_data))

if dataset_type == "finqa":
    output_prediction_file = os.path.join(root_path, "FinQA",
                                          "train_TPA_Multiply_Divide.json")
else:
    output_prediction_file = os.path.join(root_path, "ConvFinQA",
                                          "train_TPA_Multiply_Divide.json")

new_train = new_data + data
random.shuffle(new_train)
with open(output_prediction_file, "w") as f:
    json.dump(new_train, f, indent=4)
