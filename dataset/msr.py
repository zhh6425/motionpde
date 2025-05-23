import os
import numpy as np
from torch.utils.data import Dataset
from .build import DATASET_REGISTRY
import h5py
import concurrent.futures

Cross_Subject = [1, 2, 3, 4, 5]

def sample_points(p, num_point=2048):
    replace = True if p.shape[0] < num_point else False
    r = np.random.choice(p.shape[0], size=num_point, replace=replace)
    return p[r, :]

def clip_normalize(clip):
    pc = np.reshape(a=clip, newshape=[-1, 3])
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    clip = (clip - centroid) / m
    return clip

def drop_points(clip, p=0.1, l=0.01):
    T, N, C = clip.shape
    num_points_to_replace = int(N * p)
    replaced_clip = clip.copy()
    for t in range(T):
        indices = np.random.choice(N, num_points_to_replace, replace=False)
        noise = np.random.normal(0, l, (num_points_to_replace, C))
        replaced_clip[t, indices] = noise
    return replaced_clip

def drop_frames(clip, p=0.1, l=0.01):
    T, N, C = clip.shape
    num_frames_to_replace = int(T * p)
    replaced_clip = clip.copy()
    frame_indices = np.random.choice(T, num_frames_to_replace, replace=False)
    noise = np.random.normal(0, l, (num_frames_to_replace, N, C))
    replaced_clip[frame_indices] = noise
    return replaced_clip

@DATASET_REGISTRY.register()
class MSRACTION3D(Dataset):
    def __init__(self, cfg, split='train'):
        super(MSRACTION3D, self).__init__()
        self.frames_per_clip = cfg.frames_per_clip
        self.step_between_clips = cfg.step_between_clips
        self.num_points = cfg.num_points
        self.use_multi_dataset = cfg.use_multi_dataset
        self.name = "MSRACTION3D"

        dataset_idx = cfg.dataset_list.index(self.name)
        self.meta_roots = cfg.meta_list[dataset_idx]
        self.data_roots = cfg.data_roots[dataset_idx]
        DATA_CROSS_SUBJECT = True

        self.video_names = []
        self.labels = []
        self.index_map = []
        index = 0
        self.TRAIN_ID = Cross_Subject

        with open(self.meta_roots, 'r') as f:
            for line in f:
                name, nframes = line.split()
                split_name = int(name.split('_')[1].split('s')[1])  # msr + utd
                if split != 'test':
                    if split_name in self.TRAIN_ID:
                        label = int(name.split('_')[0][1:]) - 1
                        nframes = int(nframes)
                        for t in range(0, nframes - (self.frames_per_clip - 1), self.step_between_clips):
                            self.index_map.append((index, t))
                        index += 1
                        self.labels.append(label)
                        self.video_names.append(name)
                else:
                    if split_name not in self.TRAIN_ID:
                        label = int(name.split('_')[0][1:]) - 1
                        nframes = int(nframes)
                        for t in range(0, nframes - (self.frames_per_clip - 1), self.step_between_clips):
                            self.index_map.append((index, t))
                        index += 1
                        self.labels.append(label)
                        self.video_names.append(name)

        self.train = split != 'test'
        self.num_classes = max(self.labels) + 1

    def load_frame(self, i, video_name):
        return np.load(os.path.join(self.data_roots, video_name, f'{i}.npz'))['arr_0']

    def load_video_clip(self, video_name, start):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            clip = list(
                executor.map(lambda i: self.load_frame(i, video_name), range(start, start + self.step_between_clips * self.frames_per_clip, self.step_between_clips))
                )

        # clip = []
        # end = start + self.frames_per_clip
        # for i in range(start, end):
        #     frame_data = np.load(os.path.join(self.data_roots, video_name, f'{i}.npz'))['arr_0']
        #     clip.append(frame_data)
        # with h5py.File(os.path.join(self.data_roots, f'msr_{video_name[9:11]}.h5'), 'r') as hf:
        #     group = hf[video_name]
        #     for i in range(start, end):
        #         dset_name = f'frame_{i:04d}'
        #         if dset_name in group:
        #             frame_data = group[dset_name][()]
        #             clip.append(frame_data)
        #         else:
        #             break
        assert len(clip) == self.frames_per_clip
        return clip

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        extra_data = {}
        if self.use_multi_dataset:
            extra_data["dataset_name"] = self.name

        label = self.labels[index]
        clip = self.load_video_clip(self.video_names[index], t)

        # for i, p in enumerate(clip):
        #     replace = True if p.shape[0] < self.num_points else False
        #     r = np.random.choice(p.shape[0], size=self.num_points, replace=replace)
        #     clip[i] = p[r, :]
        clip = [sample_points(np.array(frame), self.num_points) for frame in clip]
        clip = clip_normalize(np.array(clip))

        if self.train:
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales
        # else:
        #     clip = drop_points(clip, p=0.7, l=0.05)
        #     # clip = drop_frames(clip, p=0.1)

        return {
            'clip': clip.astype(np.float32),
            'label': label, 
            'index': index, 
            'extra_data': extra_data
        }
     
