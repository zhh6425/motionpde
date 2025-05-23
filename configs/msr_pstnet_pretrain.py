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
        self.train_mode = 'two_stage'  # encoder_only/one_stage/two_stage
        self.pretrain = True if self.train_mode == 'two_stage' else False
        self.use_info = False
        self.add_cross_proj = False
        self.use_multi_dataset = False
        # P4D
        self.radius = 0.1#
        self.nsamples = 9#
        self.dropout_rate = 0.#
        # Project
        self.hidden_dim = 1024#
        # SolvingModule
        self.head = 8#
        self.depth = 1#
        self.num_basis = 16#
        self.num_token = 4#
        self.tem = 0.1#
        self.loss_weight = 0.1#
        # training
        self.train_batch_size = 32
        self.epochs  = 45
        self.workers = 16#
        self.lr = 0.03#
        self.warmup_ratio = 0.1#
        self.weight_decay = 1e-4#
        self.label_smoothing = 0.#
        # output
        self.print_freq = 10
        self.output_dir = f'{self.dataset}_{self.model}_{self.lr}_{self.radius}_{self.train_batch_size}_{self.epochs}_{self.train_mode}-' \
                          f'{self.hidden_dim}-{self.head}-{self.num_basis}-{self.num_token}-{self.tem}-{self.loss_weight}'
        self.model_init = None
        self.resume = None # f'OUTPUT/{self.output_dir}/checkpoint/model_best.pth'
        self.start_epoch = 0
        # Others
        self.seed = 0
        super().__init__(**kwargs)
