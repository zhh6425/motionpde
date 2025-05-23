import numpy as np
import torch
import argparse
import importlib
import os
import sys
import numpy as np
import torch
import torch.utils.data
import sklearn.metrics as metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from dataset import build_dataset
from models import build_model
from trainer import *
from models import *

from transformers import TrainingArguments

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Training Config File")
    parser.add_argument("--ensemble", type=int, default=1)
    args = parser.parse_args()
    config_file = args.cfg.split('/')[-1].split('.')[0]
    config_setting = importlib.import_module("configs." + config_file).Config()
    config_setting.ensemble = args.ensemble
    return config_setting

args = parse_args()

test_dataset = build_dataset(args.dataset, args, split='test')

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

num_classes = test_dataset.num_classes
#model = build_model(name='BaseActSegModel', args=args, num_classes=num_classes)

def initialize_model(model_name):
    model_config = PretrainedConfig.from_pretrained(model_name)
    model = BaseActSegModel.from_pretrained(
        model_name,
        config=model_config, 
        num_classes=num_classes,
    )
    return model

def average_models_state_dicts(model_state_dicts):
    num_models = len(model_state_dicts)
    averaged_model = copy.deepcopy(models[0])
    averaged_state_dict = averaged_model.state_dict()

    for key in averaged_state_dict.keys():
        layer_weights = sum([model_state_dict[key] for model_state_dict in model_state_dicts]) / num_models
        averaged_state_dict[key] = layer_weights
    averaged_model.load_state_dict(averaged_state_dict)
    return averaged_model

model_names = os.listdir(args.model_init)
model_names.sort()
model_names = [name for name in model_names if name.startswith('checkpoint')][: args.ensemble]

if len(model_names) > 0:
    models = [initialize_model(os.path.join(args.model_init, model_name)) for model_name in model_names]
    model_state_dicts = [model.state_dict() for model in models]

    model = average_models_state_dicts(model_state_dicts)
    model.save_pretrained(os.path.join(args.model_init, f'ensemble_last_{args.ensemble}'))
else:
    model = initialize_model(args.model_init)
    # model = build_model(name='BaseModel', args=args, num_classes=num_classes)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    remove_unused_columns=False,
    report_to='none',
)

trainer = MotionTrainerforActionSeg(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    # compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

out = trainer.evaluate()
if 'eval_mean_acc' in out:
    print(out['eval_mean_acc'])
np.save(os.path.join(args.output_dir, 'preds.npy'), out['video_preds'])




