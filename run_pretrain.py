import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
import sklearn.metrics as metrics
import argparse
import importlib
import os
import sys
import numpy as np
import torch
import torch.utils.data

from thop import profile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from dataset import build_dataset
from models import build_model
from trainer import *

from transformers import TrainingArguments, TrainerCallback

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Training Config File")
    args = parser.parse_args()
    config_file = args.cfg.split('/')[-1].split('.')[0]
    config_setting = importlib.import_module("configs." + config_file).Config()
    return config_setting

args = parse_args()

train_dataset = build_dataset(args.dataset, args, split='train')

def collate_fn(examples):

    inputs = np.array([example['clip'] for example in examples])
    labels = np.array([example['label'] for example in examples])
    indexs = np.array([example['index'] for example in examples])

    inputs = torch.tensor(inputs, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    indexs = torch.tensor(indexs, dtype=torch.int)

    return {
        'inputs': inputs,
        'labels': labels,
        'indexs': indexs
    }

num_classes = train_dataset.num_classes
model = build_model(name='BaseModel', args=args, num_classes=num_classes)

class LastEpochSaveCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch == args.num_train_epochs:
            control.should_save = True
        else:
            control.should_save = False

training_args = TrainingArguments(
    output_dir=args.output_dir,
    run_name=args.output_dir,
    per_device_train_batch_size=args.train_batch_size,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    # max_steps=args.max_steps,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    logging_steps=args.print_freq,
    save_strategy='epoch',
    save_total_limit=5,  # always save the last
    dataloader_num_workers=args.workers,
    remove_unused_columns=False,
    report_to='none',
)

sgd_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,)

trainer = MotionTrainerforActionCls(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    optimizers=(sgd_optimizer, None),
    # compute_metrics=compute_metrics,
    data_collator=collate_fn,
    callbacks=[LastEpochSaveCallback()]
)

trainer.train()