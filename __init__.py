from __future__ import print_function
import os
import sys
import argparse
import torch
from args import opt
# Read configuration
# config = argparse.ArgumentParser(fromfile_prefix_chars='@')
# config.add_argument('--name', help='input experiment name')
# config.add_argument('--nohup', default=True,
#                     help='if run the experiment nohup, default: True')
# config.add_argument('--cuda', default=False,
#                     help='if enable GPU acceleration, default: False')
# config.add_argument('--mode', default='train',
#                     help='choose {} or {}'.format('training', 'testing'))

# config.add_argument('--N-subimgs', default=190000, help='number of patches')
# config.add_argument('--N-epochs', default=10, help='number of training epoch')
# config.add_argument('--batch-size', default=32, help='batch size')
# config.add_argument('--patch-height', default=48,
#                     help='height of patch, (H, W, C)')
# config.add_argument('--patch-width', default=48,
#                     help='width of patch, (H, W, C)')
# global opt
# opt = vars(config.parse_args(['@configuration.txt']))
# print(opt)
opt = vars(opt)
exp_name = opt['name']
nohup = opt['nohup']
run_gpu = opt['cuda']
mode = opt['mode']


if torch.cuda.is_available() and not run_gpu:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda to enable it")
if run_gpu:
    torch = torch.cuda
    print('\n {} will run on GPU with CUDA acceration'.format(mode))
else:
    print('\n {} will run on CPU'.format(mode))

# Create results directory
result_dir = exp_name
print('\n 1. Create dir for the experimental results (if not already exits')
if os.path.exists(result_dir):
    print('{} already exits'.format(result_dir))
elif os.system == 'win32':
    os.system('mkdir ' + result_dir)
else:
    os.system('mkdir -p ' + result_dir)

# add write_config.py to copy the configuration into experiment folder
if mode == 'training':
    if os.system == 'win32':
        os.system('type nul> ' + os.path.join(result_dir,
                                              result_dir + '_configuration.txt'))
    else:
        os.system('touch ' + os.path.join(result_dir,
                                          result_dir + '_configuration.txt'))

    with open(file=os.path.join(result_dir, result_dir + '_configuration.txt'), mode='w') as f:
        _ = opt
        for key in[*_]:
            f.write('--{}\n{}\n'.format(key, _[key]))

# run the experiment
if nohup:
    print('\n 2. Run the {} with nohup'.format(mode))
    os.system(' nohup python -u ./src/retinaNN_{}.py > '.format(mode) +
              './' + result_dir + '/' + exp_name + '_{}.nohup'.format(mode))
else:
    os.system('python -u ./src/retinNN_{}.py')
    print('\n python ./src/retinaNN_{}.py'.format(mode))

# Now, training/testing is running with another script
