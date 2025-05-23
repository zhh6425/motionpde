import numpy as np
import torch
import argparse
import importlib
import os
import sys
import torch.utils.data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from dataset import build_dataset
from models import build_model
from trainer import *

from transformers import TrainingArguments, TrainerCallback

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Training Config File")
    parser.add_argument("--probing", action='store_true', help="Use Probing")
    args = parser.parse_args()
    config_file = args.cfg.split('/')[-1].split('.')[0]
    config_setting = importlib.import_module("configs." + config_file).Config()
    config_setting.probing = args.probing
    return config_setting

args = parse_args()

train_dataset = build_dataset(args.dataset, args, split='train')
valid_dataset = build_dataset(args.dataset, args, split='test')

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

if args.probing:
    print('freeze the encoder')
    args.output_dir += '_probing'
    for param in model.encoder.parameters():
        param.requires_grad = False

class LastEpochEvalAndSaveCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch == args.num_train_epochs:
            control.should_evaluate = True
            control.should_save = True
        else:
            control.should_evaluate = False
            control.should_save = False

training_args = TrainingArguments(
    output_dir=args.output_dir,
    run_name=args.output_dir,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.train_batch_size,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    # max_steps=args.max_steps,
    warmup_ratio=args.warmup_ratio,
    logging_steps=args.print_freq,
    metric_for_best_model='video_acc',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    # eval_steps=args.eval_freq,
    # save_steps=args.eval_freq,
    save_total_limit=1,  # always save the best & the last
    dataloader_num_workers=args.workers,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    report_to='none',
)

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
sgd_optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,)
adam_optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay,)

trainer = MotionTrainerforActionCls(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    optimizers=(adam_optimizer, None) if args.dataset == 'SHREC' else (sgd_optimizer, None),
    # compute_metrics=compute_metrics,
    data_collator=collate_fn,
    # callbacks=[LastEpochEvalAndSaveCallback()] if args.dataset == 'NTURGBD' else None
)

trainer.train()