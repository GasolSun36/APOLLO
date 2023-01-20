import torch
import logging

logger = logging.getLogger(__name__)
import random
from transformers import AdamW
import numpy as np
import torch.distributed as dist


def set_env(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    # Set seed
    set_seed(args)


def is_first_worker():
    return not dist.is_available() or not dist.is_initialized(
    ) or dist.get_rank() == 0


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_optimizer(
    args,
    model: nn.Module,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if p.requires_grad and not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        weight_decay
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if p.requires_grad and any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.0
    }]
    return AdamW(optimizer_grouped_parameters,
                 lr=args.learning_rate,
                 eps=args.adam_epsilon)
