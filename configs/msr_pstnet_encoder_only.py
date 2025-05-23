from transformers import PretrainedConfig

class Config(PretrainedConfig):
    model_type = "points"
    def __init__(self, **kwargs,):
        self.dataset = "MSRACTION3D"
        self.dataset_list = ['MSRACTION3D']  # for multi dataset training
        self.data_roots = ['DATA/MSR-Action3D/video']
        self.meta_list = ['DATA/MSR-Action3D/msr.list']
        self.step_between_clips = 1
        self.frames_per_clip = 24
        self.num_points = 2048
        self.model = 'PSTNet'
        self.train_mode = 'encoder_only'  # encoder_only/one_stage/two_stage
        self.pretrain = True if self.train_mode == 'two_stage' else False
        self.use_info = False
        self.add_cross_proj = False
        self.use_multi_dataset = False
        # P4D
        self.radius = 0.3#
        self.nsamples = 9#
        self.dropout_rate = 0.#
       # training
        self.train_batch_size = 32#
        self.epochs = 35
        self.workers = 16#
        self.lr = 0.01
        self.warmup_ratio = 0.1#
        self.weight_decay = 1e-4#
        self.label_smoothing = 0.#
        # output
        self.print_freq = 10
        self.output_dir = f'{self.dataset}_{self.model}_{self.lr}_{self.train_batch_size}_{self.epochs}_{self.train_mode}'
        self.model_init = None
        if self.model_init is not None:
            model_name = self.model_init.split("/")[1]
            self.output_dir = self.output_dir + f'_init_from_{model_name}'
        # Others
        self.seed = 0
        super().__init__(**kwargs)
