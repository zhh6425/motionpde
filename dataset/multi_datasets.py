import os
import sys
import numpy as np
from torch.utils.data import ConcatDataset
from .build import DATASET_REGISTRY, build_dataset

@DATASET_REGISTRY.register()
class MULTIDATASETS(ConcatDataset):
    def __init__(self, cfg, split):
        # construct each datasets
        self.dataset_list = cfg.dataset_list
        self.dataset_replicas = cfg.dataset_replicas

        datasets = []
        self.num_classes = {}
        for num_replica, dataset_name in zip(self.dataset_replicas, self.dataset_list):
            if split != "train":
                num_replica = 1
            for i in range(int(num_replica)):
                dataset = build_dataset(dataset_name, cfg, split)
                datasets.append(dataset)
            self.num_classes[dataset_name] = dataset.num_classes

        super().__init__(datasets)
