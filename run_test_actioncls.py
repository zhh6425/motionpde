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

num_classes = valid_dataset.num_classes

def initialize_model(model_name):
    model_config = PretrainedConfig.from_pretrained(model_name)
    model = BaseModel.from_pretrained(
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

try:
    model_names = os.listdir(args.model_init)
    model_names.sort()
    model_names = [name for name in model_names if name.startswith('checkpoint')][: args.ensemble]
except:
    model_names = []

if len(model_names) > 0:
    models = [initialize_model(os.path.join(args.model_init, model_name)) for model_name in model_names]
    model_state_dicts = [model.state_dict() for model in models]

    model = average_models_state_dicts(model_state_dicts)
    model.save_pretrained(os.path.join(args.model_init, f'ensemble_last_{args.ensemble}'))
else:
    model = initialize_model(args.model_init)

print(model)
print(f'testing model from {args.model_init}')

# model = model.cuda()
# model.eval()

# def evaluate(batch_size=32, metric_key_prefix: str = "eval"):

#     probs = []
#     indexs = []
#     labels = []

#     eval_loss = 0.
#     with torch.no_grad():
#          for step in tqdm(range(0, len(valid_dataset), batch_size)):
#             inputs = [valid_dataset[step + idx] for idx in range(0, min(batch_size, len(valid_dataset)-step))]
#             inputs = collate_fn(inputs)

#             labels.append(inputs['labels'].detach().cpu())
#             inputs = {k: v.cuda() for k, v in inputs.items()}
#             outputs = model(**inputs)
#             eval_loss += outputs['loss']
#             probs.append(outputs['logit'].detach().cpu())
#             indexs.append(outputs['indexs'].detach().cpu())

#     eval_loss = eval_loss / len(valid_dataset)

#     probs = torch.cat(probs, dim=0).numpy()
#     preds = probs.argmax(-1)
#     indexs = torch.cat(indexs, dim=0).numpy()
#     labels = torch.cat(labels, dim=0).numpy()
        
#     acc = metrics.accuracy_score(labels, preds)

#     num_video = max(indexs) + 1
#     video_probs = np.zeros((num_video, num_classes), dtype=probs.dtype)
#     video_labels = np.zeros(num_video, dtype=labels.dtype)   

#     for idx in np.unique(indexs):
#         video_probs[idx] = np.sum(probs[indexs == idx], axis=0)
#         video_labels[idx] = labels[indexs == idx][0]
    
#     video_preds = video_probs.argmax(-1)
#     acc_video = metrics.accuracy_score(video_labels, video_preds)

#     class_count = [0] * num_classes
#     class_correct = [0] * num_classes

#     for i, pre in enumerate(video_preds):
#         label = video_labels[i]
#         class_count[label] += 1
#         class_correct[label] += (pre == label)
#     class_acc = [c / float(s) for c, s in zip(class_correct, class_count)]

#     metrics_dict = {}
#     metrics_dict['eval_loss'] = eval_loss.tolist()
#     metrics_dict['acc'] = acc
#     metrics_dict['video_acc'] = acc_video
#     metrics_dict['class_acc'] = class_acc

#     # Prefix all keys with metric_key_prefix + '_'
#     for key in list(metrics_dict.keys()):
#         if not key.startswith(f"{metric_key_prefix}_"):
#             metrics_dict[f"{metric_key_prefix}_{key}"] = metrics_dict.pop(key)

#     return metrics_dict

# print(evaluate())


training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_eval_batch_size=args.train_batch_size,
    dataloader_num_workers=args.workers,
    remove_unused_columns=False,
    report_to='none',
)

trainer = MotionTrainerforActionCls(
    model=model,
    args=training_args,
    eval_dataset=valid_dataset,
    # compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

print(trainer.evaluate())

