# MotionPDE
> Official Repo of paper _Paving the Way for Point Cloud Video Representation Learning Using A PDE Model_
(In progress)

# Install
## install environment
'''
conda env create -f environment.yml
conda activate pde
'''
## install pointnet2 ext
'''
cd utils/cpp/pointnet2_batch/
python setup.py install
'''

# Quickstart
1. Download [MSRAction-3D](https://sites.google.com/view/wanqingli/data-sets/msr-action3d) dataset. You will need the Depth.rar (53.8Mb)

2. unzip file and run
'''
cd dataset/preparedata/msr
python preprocess_msr.py --input_dir [MSR_DATA_PATH] --output_dir [OUT_PATH]
'''

3. run train one-stage
'''
python run_train_actioncls.py --cfg configs/msr_pstnet_one_stage.py
'''
or test from our huggingface checkpoint
'''
python run_test_actioncls.py --cfg configs/msr_test.py
'''


