
# APOLLO: An Optimized Training Approach for Long-form Numerical Reasoning

This repo provides the code of APOLLO. In the paper, we adopt a number-aware negative sampling strategy in retriever to discriminate key numerical facts from others. Moreover, we design consistency-based reinforcement learning with target program augmentation, to increase program diversity and ultimately increase the execution accuracy.
***
This repo is still developing, feel free to report bugs and we will fix them.
## How to cite
If you extend or use this work, please cite the [paper](https://arxiv.org/abs/2212.07249) where it was introduced:

```
@article{sun2022apollo,
  title={APOLLO: An Optimized Training Approach for Long-form Numerical Reasoning},
  author={Sun, Jiashuo and Zhang, Hang and Lin, Chen and Gong, Yeyun and Guo, Jian and Duan, Nan},
  journal={arXiv preprint arXiv:2212.07249},
  year={2022}
}
```
***
## Leaderboard
The FinQA and ConvFinQA challenge leaderboard is on CodaLab [https://codalab.lisn.upsaclay.fr/competitions/4138#results](https://codalab.lisn.upsaclay.fr/competitions/4138#results) and  [https://codalab.lisn.upsaclay.fr/competitions/8582#results](https://codalab.lisn.upsaclay.fr/competitions/8582#results). APOLLO achieves rank 1 on both datasets.
***

## Requirements 

 - pytorch 1.7.1
 - huggingface transformers 4.4.2
 - sympy 1.9

***
## Code
### Datasets
The FinQA and ConvFinQA datasets are in `/APOLLO/dataset/`.In FinQA, there are train,dev and test set. However, in ConvFinQA, there are only train and dev set.
### The Retriever
#### Pre-process
Pre-processing is mainly to process the data of train, dev and test in advance, so that the training and testing stage can directly load these processed data. To pre-process these data, edit data_process.py to set your own project path and some hyperparameters. In particular, if you use number-aware negative sampling during training, edit and run number_aware.py to create the train_number_aware.json file in advance.
Specifically, firstly run:

```bash
python number_aware.py
```
Secondly, run:

```bash
python data_process.py
```
#### Train
To train the retriever, you can run:

```bash
python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=8889 Main.py\
--root_path "/colab_space/sunjiashuo/APOLLO/" \
--model_save_name retriever-deberta \
--pretrained_model deberta \
--model_size microsoft/deberta-v3-large \
--mode train \
--neg_rate 3 \
--max_seq_length 512 --batch_size 8 --gradient_accumulation_steps 1 \
--learning_rate 2e-5 --epoch 50 --report 500 \
--features_dir /colab_space/sunjiashuo/FinQA/dataset/retriever/ \
--examples_dir /colab_space/sunjiashuo/FinQA/dataset/retriever/ \
--tags 1 \
--dataset_type finqa
```
#### Inference
To inference, you can run:

```bash
python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=8899 Main.py\
--root_path "/colab_space/sunjiashuo/APOLLO/" \
--model_save_name retriever-deberta \
--pretrained_model deberta \
--model_size microsoft/deberta-v3-large \
--mode inference \
--features_dir /colab_space/sunjiashuo/FinQA/dataset/retriever/ \
--examples_dir /colab_space/sunjiashuo/FinQA/dataset/retriever/ \
--saved_model_path "the path of your selected checkpoint in the training" \
--dataset_type finqa --tags 2 
```
#### Convert
To convert data for generator training, you can run:
```bash
python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=8999 Main.py\
--root_path "/colab_space/sunjiashuo/APOLLO/" \
--model_save_name retriever-deberta \
--pretrained_model deberta \
--model_size microsoft/deberta-v3-large \
--mode convert \
--dataset_type finqa --tags 3 
```

### The Generator
#### Pre-process
To pre-process the data, you can edit and run:

```bash
python data_process.py
```
#### Train
To train the retriever, you can run:

```bash
python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=7889 Main.py\
--root_path "/colab_space/sunjiashuo/APOLLO/" \
--model_save_name generator-roberta-large \
--pretrained_model roberta \
--model_size roberta-large \
--mode train \
--retrieve_mode single --program_mode seq \
--max_seq_length 512 --batch_size 8 --gradient_accumulation_steps 1 \
--learning_rate 2e-5 --epoch 50 --max_program_length 30 --report 500 \
--features_dir /colab_space/sunjiashuo/FinQA/dataset/generator/ \
--examples_dir /colab_space/sunjiashuo/FinQA/dataset/generator/ \
--tags 1 \
--dataset_type finqa
```
#### Inference
To inference, you can run:

```bash
python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=7899 Main.py\
--root_path "/colab_space/sunjiashuo/APOLLO/" \
--model_save_name generator-roberta-large \
--pretrained_model roberta \
--model_size roberta-large \
--mode inference \
--features_dir /colab_space/sunjiashuo/FinQA/dataset/generator/ \
--examples_dir /colab_space/sunjiashuo/FinQA/dataset/generator/ \
--tags 2 \
--saved_model_path "the path of your selected checkpoint in the training" \
--dataset_type finqa
```
#### Consistency-based Reinforcement learning
In order to adopt reinforcement learning, you need to train a model in the above process. Then, you can run:
```bash
python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=6899 Main.py\
--root_path "/colab_space/sunjiashuo/APOLLO/" \
--model_save_name generator-roberta-large \
--pretrained_model roberta \
--model_size roberta-large \
--mode train \
--features_dir /colab_space/sunjiashuo/FinQA/dataset/generator/ \
--examples_dir /colab_space/sunjiashuo/FinQA/dataset/generator/ \
--tags 3 \
--saved_model_path "the path of your selected checkpoint in the supervised training" \
--dataset_type finqa --rl
```
#### Consistency-based Reinforcement learning
In order to adopt reinforcement learning, you need to train a model in the above process. Then, you can run:
```bash
python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=6899 Main.py\
--root_path "/colab_space/sunjiashuo/APOLLO/" \
--model_save_name generator-roberta-large \
--pretrained_model roberta \
--model_size roberta-large \
--mode train \
--features_dir /colab_space/sunjiashuo/FinQA/dataset/generator/ \
--examples_dir /colab_space/sunjiashuo/FinQA/dataset/generator/ \
--tags 3 \
--saved_model_path "the path of your selected checkpoint in the supervised training" \
--dataset_type finqa --rl
```
The test of reinforcement learning is the same as the inference of supervised training.
#### Target Program Augmentation
In order to adopt target program augmentation, you need to cd in `/TPA` Then, you can run:

```bash
python TPA_Switch.py
python TPA_Add_Subtract.py
python TPA_Multiply_Divide.py
python TPA_Mul-Div.py
```
to create TPA dataset. Then, you need to edit and run data_process.py in `/Generator` to create features:

```bash
python data_process.py
```
In particular, you need to edit:

```python
examples_dir = "/colab_space/sunjiashuo/APOLLO/dataset/generator/"
features_dir = "/colab_space/sunjiashuo/APOLLO/dataset/generator/"
if dataset_type == "finqa":
    train_file = root_path + "dataset/FinQA/train_retrieve_output.json"
else:
    train_file = root_path + "dataset/ConvFinQA/train_retrieve_output.json"
f = open(os.path.join(examples_dir, 'train_examples.pickle'), 'wb')
f = open(os.path.join(examples_dir, 'dev_examples.pickle'), 'wb')
f = open(os.path.join(features_dir, 'train_features.pickle'), 'wb')
f = open(os.path.join(features_dir, 'dev_features.pickle'), 'wb')
f = open(os.path.join(examples_dir, 'test_examples.pickle'), 'wb')
f = open(os.path.join(features_dir, 'test_features.pickle'), 'wb')
```
to 
```python
examples_dir = "/colab_space/sunjiashuo/APOLLO/dataset/generator_tpa/"
features_dir = "/colab_space/sunjiashuo/APOLLO/dataset/generator_tpa/"
if dataset_type == "finqa":
    train_file = root_path + "dataset/FinQA/train_TPA_Switch.json"  # or other TPA methods
else:
    train_file = root_path + "dataset/ConvFinQA/train_TPA_Switch.json"  # or other TPA methods
f = open(os.path.join(examples_dir, 'train_examples_switch.pickle'), 'wb')
f = open(os.path.join(examples_dir, 'dev_examples_switch.pickle'), 'wb')
f = open(os.path.join(features_dir, 'train_features_switch.pickle'), 'wb')
f = open(os.path.join(features_dir, 'dev_features_switch.pickle'), 'wb')
f = open(os.path.join(examples_dir, 'test_examples_switch.pickle'), 'wb')
f = open(os.path.join(features_dir, 'test_features_switch.pickle'), 'wb')
```

Then, you can run this command in `/Generator`:
```bash
python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=5899 Main.py\
--root_path "/colab_space/sunjiashuo/APOLLO/" \
--model_save_name generator-roberta-large \
--pretrained_model roberta \
--model_size roberta-large \
--mode train \
--features_dir /colab_space/sunjiashuo/FinQA/dataset/generator_tpa/ \
--examples_dir /colab_space/sunjiashuo/FinQA/dataset/generator_tpa/ \
--tags 4 \
--saved_model_path "the path of your selected checkpoint in the supervised training" \
--dataset_type finqa --tpa \
--tpa_methods switch
```
The test of TPA is the same as the inference of supervised training.