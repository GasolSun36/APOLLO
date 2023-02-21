import sys
import collections
import tqdm
import json
import torch.utils.data as torchdata

from generator.utils import *


def program_tokens_to_program(program_tokens):
    total_program = []
    program_str = ""
    for item in program_tokens[:-1]:
        total_program.extend(item)
    for i in range(len(total_program)):
        if i == 1 or i == 5 or i == 9 or i == 13 or i == 17 or i == 21 or i == 25 or i == 29 or i == 33:
            program_str += total_program[i]
            program_str += ", "
        elif total_program[i] == ')' and i != len(total_program) - 1:
            program_str += total_program[i]
            program_str += ", "
        elif total_program[i] == ')' and i == len(total_program) - 1:
            program_str += total_program[i]
        else:
            program_str += total_program[i]
    return program_str


def change(program_list):
    program_list[1], program_list[2] = program_list[2], program_list[1]
    return program_list


def create_program(op, id, num):
    program = []
    program.append(op)
    program.append('#{}'.format(id))
    program.append('{}'.format(num))
    program.append(')')
    return program
