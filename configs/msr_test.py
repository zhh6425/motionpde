from transformers import PretrainedConfig

class Config(PretrainedConfig):
    def __init__(self):
        self.dataset = "MSRACTION3D"
        self.dataset_list = ['MSRACTION3D']  # for multi dataset training
        self.data_roots = ['DATA/MSR-Action3D/video']
        self.meta_list = ['DATA/MSR-Action3D/msr.list']
        self.step_between_clips = 1
        self.frames_per_clip = 24
        self.num_points = 2048
        self.use_info = False
        self.add_cross_proj = False
        self.use_multi_dataset = False
        self.train_batch_size = 32
        self.workers = 16
        # output
        self.print_freq = 10
        self.model_init = 'zxh4546/msr-psttransformer-pde-fromscratch'
        self.output_dir = f'{self.dataset}-test_model_{self.model_init.split("/")[-1]}'
        # Others
        self.seed = 0
