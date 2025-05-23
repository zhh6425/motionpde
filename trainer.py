from typing import List, Optional
import os
import numpy as np
import torch
from tqdm import tqdm

from transformers import Trainer
from transformers.utils import logging

import sklearn.metrics as metrics
from collections import defaultdict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logger = logging.get_logger(__name__)


class MotionTrainerforActionCls(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        # if 'index' in inputs:
        #     index = inputs.pop('index')

        outputs = model(**inputs)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, ignore_keys=None, metric_key_prefix: str = "eval"):

        logger.info("*** Training Evaluate ***")

        probs = []
        indexs = []
        labels = []

        args = self.args
        model = self.model.to(args.device)

        model.eval()
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        batch_size = eval_dataloader.batch_size
        num_classes = self.eval_dataset.num_classes
        eval_loss = 0.
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(eval_dataloader)):
                labels.append(inputs['labels'].detach().cpu())
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                outputs = model(**inputs)
                eval_loss += outputs['loss']
                probs.append(outputs['logit'].detach().cpu())
                indexs.append(outputs['indexs'].detach().cpu())

        eval_loss = eval_loss / len(eval_dataloader)

        probs = torch.cat(probs, dim=0).numpy()
        preds = probs.argmax(-1)
        indexs = torch.cat(indexs, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()
        
        acc = metrics.accuracy_score(labels, preds)

        num_video = max(indexs) + 1
        video_probs = np.zeros((num_video, num_classes), dtype=probs.dtype)
        video_labels = np.zeros(num_video, dtype=labels.dtype)   

        for idx in np.unique(indexs):
            video_probs[idx] = np.sum(probs[indexs == idx], axis=0)
            video_labels[idx] = labels[indexs == idx][0]
    
        video_preds = video_probs.argmax(-1)
        acc_video = metrics.accuracy_score(video_labels, video_preds)

        class_count = [0] * num_classes
        class_correct = [0] * num_classes

        for i, pre in enumerate(video_preds):
            label = video_labels[i]
            class_count[label] += 1
            class_correct[label] += (pre == label)
        class_acc = [c / float(s) for c, s in zip(class_correct, class_count)]

        metrics_dict = {}
        metrics_dict['eval_loss'] = eval_loss.tolist()
        metrics_dict['acc'] = acc
        metrics_dict['video_acc'] = acc_video
        metrics_dict['class_acc'] = class_acc

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics_dict.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics_dict[f"{metric_key_prefix}_{key}"] = metrics_dict.pop(key)

        self.log(metrics_dict)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics_dict)

        return metrics_dict


class MotionTrainerforActionSeg(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        # if 'index' in inputs:
        #     index = inputs.pop('index')

        outputs = model(**inputs)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    def get_labels_start_end_time(self, frame_wise_labels, bg_class=["background"]):
        labels = []
        starts = []
        ends = []
        last_label = frame_wise_labels[0]
        if frame_wise_labels[0] not in bg_class:
            labels.append(frame_wise_labels[0])
            starts.append(0)
        for i in range(len(frame_wise_labels)):
            if frame_wise_labels[i] != last_label:
                if frame_wise_labels[i] not in bg_class:
                    labels.append(frame_wise_labels[i])
                    starts.append(i)
                if last_label not in bg_class:
                    ends.append(i)
                last_label = frame_wise_labels[i]
        if last_label not in bg_class:
            ends.append(i)
        return labels, starts, ends
    
    def levenstein(self, p, y, norm=False):
        m_row = len(p)    
        n_col = len(y)
        D = np.zeros([m_row+1, n_col+1], np.float64)
        for i in range(m_row+1):
            D[i, 0] = i
        for i in range(n_col+1):
            D[0, i] = i
        for j in range(1, n_col+1):
            for i in range(1, m_row+1):
                if y[j-1] == p[i-1]:
                    D[i, j] = D[i-1, j-1]
                else:
                    D[i, j] = min(D[i-1, j] + 1, D[i, j-1] + 1, D[i-1, j-1] + 1)
        if norm:
            score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
        else:
            score = D[-1, -1]
        return score
    
    def edit_score(self, recognized, ground_truth, norm=True, bg_class=["background"]):
        P, _, _ = self.get_labels_start_end_time(recognized, bg_class)
        Y, _, _ = self.get_labels_start_end_time(ground_truth, bg_class)
        return self.levenstein(P, Y, norm)

    def evaluate(self, ignore_keys=None, metric_key_prefix: str = "eval"):

        logger.info("*** Training Evaluate ***")

        probs = []
        indexs = []

        args = self.args
        model = self.model.to(args.device)

        num_classes = self.eval_dataset.num_classes
        model.eval()

        BSZ = 16
        video_probs = defaultdict(lambda: np.zeros((150, num_classes), dtype=np.float32))
        eval_loss = 0.
        edit = 0
        step_count = 0
        with torch.no_grad():
            for step in tqdm(range(0, len(self.eval_dataset), BSZ)):
                inputs = [self.eval_dataset[step + idx] for idx in range(0, min(BSZ, len(self.eval_dataset)-step))]
                video_idx = [self.eval_dataset.index_map[step + idx][0] for idx in range(0, min(BSZ, len(self.eval_dataset)-step))]
                start_frame = [self.eval_dataset.index_map[step + idx][1] for idx in range(0, min(BSZ, len(self.eval_dataset)-step))]

                inputs = self.data_collator(inputs)
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                clip_labels = inputs['labels']
                if self.eval_dataset.label is None: 
                    inputs['labels'] = None
                outputs = model(**inputs)
                eval_loss += outputs['loss']

                clip_probs = outputs['logit'].detach().cpu()
                indexs.append(outputs['indexs'].detach().cpu())

                # factor = 150 / self.eval_dataset.frames_per_clip

                for i, probs in enumerate(clip_probs):
                    # video_idx, start_frame = self.eval_dataset.index_map[index_[i]]
                    preds = probs.argmax(-1)
                    # edit += self.edit_score(preds, clip_labels[i])

                    for t, prob in enumerate(probs):
                        frame_idx = start_frame[i] + t
                        np.add(video_probs[video_idx[i]][frame_idx], prob, out=video_probs[video_idx[i]][frame_idx])

                step_count += 1

        eval_loss = eval_loss / step_count

        indexs = torch.cat(indexs, dim=0).numpy()

        num_video = max(indexs) + 1
        final_video_probs = np.zeros((num_video, 150, num_classes), dtype=np.float32)
        for video_idx, probs in video_probs.items():
            final_video_probs[video_idx, :, :] = probs
    
        video_preds = final_video_probs.argmax(-1)

        metrics_dict = {}
        if self.eval_dataset.label is not None:
            metrics_dict['eval_loss'] = eval_loss.tolist()
            metrics_dict['mean_acc'] = np.mean(video_preds == self.eval_dataset.label)
            for pred, tgt in zip(video_preds, self.eval_dataset.label):
                edit += self.edit_score(pred, tgt)
            metrics_dict['edit'] = (1.0 * edit) / len(video_preds)
            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics_dict.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics_dict[f"{metric_key_prefix}_{key}"] = metrics_dict.pop(key)

            self.log(metrics_dict)
            
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics_dict)
        metrics_dict['video_preds'] = video_preds
        
        return metrics_dict


class MotionTrainerforRecons(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        # if 'index' in inputs:
        #     index = inputs.pop('index')

        outputs = model(**inputs)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, ignore_keys=None, metric_key_prefix: str = "eval"):

        logger.info("*** Training Evaluate ***")

        args = self.args

        # os.makedirs(args.output_dir, exist_ok=True)
        try:
            epoch = int(self.state.epoch)
        except:
            epoch = 0
        save_path = os.path.join(args.output_dir, f'epoch_{epoch}.png')

        model = self.model.to(args.device)

        model.eval()
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        batch_size = eval_dataloader.batch_size
        eval_loss = 0.
        save_vis = True
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(eval_dataloader)):
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                outputs = model(**inputs)
                eval_loss += outputs[0]
                if save_vis:
                    idx_ = 27
                    original_frames = inputs["inputs"][idx_].cpu().numpy()
                    reconstructed_frames = outputs[1][idx_].cpu().numpy()
                    self.visualize_point_clouds(original_frames, reconstructed_frames, save_path, [i for i in range(0, 24, 2)])
                    save_vis = False

        eval_loss = eval_loss / len(eval_dataloader)

        metrics_dict = {}
        metrics_dict['eval_loss'] = eval_loss.tolist()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics_dict.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics_dict[f"{metric_key_prefix}_{key}"] = metrics_dict.pop(key)

        self.log(metrics_dict)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics_dict)

        return metrics_dict
    
    def visualize_point_clouds(self, original, reconstructed, save_path, frame_indices):
        fig = plt.figure(figsize=(4 * len(frame_indices), 10))
        for i, frame_index in enumerate(frame_indices):
            ax1 = fig.add_subplot(2, len(frame_indices), i + 1, projection='3d')
            self.plot_point_cloud(ax1, original[frame_index], f'Original Frame {frame_index+1}', 'b')
            ax2 = fig.add_subplot(2, len(frame_indices), i + len(frame_indices) + 1, projection='3d')
            self.plot_point_cloud(ax2, reconstructed[frame_index], f'Reconstructed Frame {frame_index+1}', 'orange')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_point_cloud(self, ax, points, title, color='b'):
        points = points * -1
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c=color, cmap='hot')
        ax.set_title(title)
        ax.axis('off')
        ax.view_init(elev=100, azim=-95, roll=0)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')