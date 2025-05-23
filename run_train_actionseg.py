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
    parser.add_argument("--probing", action='store_true', help="Use Probing")
    args = parser.parse_args()
    config_file = args.cfg.split('/')[-1].split('.')[0]
    config_setting = importlib.import_module("configs." + config_file).Config()
    config_setting.probing = args.probing
    return config_setting

args = parse_args()

train_dataset = build_dataset(args.dataset, args, split='train')
val_dataset = build_dataset(args.dataset, args, split='val')

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
model = build_model(name='BaseActSegModel', args=args, num_classes=num_classes)

if args.probing:
    print('freeze the encoder')
    args.output_dir += '_probing'
    for param in model.encoder.parameters():
        param.requires_grad = False

class NoEvalCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_evaluate = False
            

training_args = TrainingArguments(
    output_dir=args.output_dir,
    run_name=args.output_dir,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.train_batch_size,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    #max_steps=args.epochs,
    warmup_ratio=args.warmup_ratio,
    logging_steps=args.print_freq,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,  
    dataloader_num_workers=args.workers,
    metric_for_best_model='mean_acc',
    load_best_model_at_end=True,
    remove_unused_columns=False,
    report_to='none',
)

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
sgd_optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,)

trainer = MotionTrainerforActionSeg(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(sgd_optimizer, None),
    # compute_metrics=compute_metrics,
    data_collator=collate_fn,
    # callbacks=[NoEvalCallback()]
)

trainer.train()

# out = trainer.evaluate()
# if 'eval_mean_acc' in out:
#     print(out['eval_mean_acc'])
# np.save(os.path.join(args.output_dir, 'preds.npy'), out['video_preds'])
