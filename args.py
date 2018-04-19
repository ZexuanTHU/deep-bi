import argparse
config = argparse.ArgumentParser(fromfile_prefix_chars='@')
config.add_argument('--name', help='input experiment name')
config.add_argument('--root-dir', help='root dir')
config.add_argument('--nohup', default=True,
                    help='if run the experiment nohup, default: True')
config.add_argument('--cuda', default=False,
                    help='if enable GPU acceleration, default: False')
config.add_argument('--mode', default='train',
                    help='choose {} or {}'.format('training', 'testing'))

config.add_argument('--N-subimgs', default=190000, help='number of patches')
config.add_argument('--N-epochs', default=10, help='number of training epoch')
config.add_argument('--batch-size', default=32, help='batch size')
config.add_argument('--patch-height', default=48,
                    help='height of patch, (H, W, C)')
config.add_argument('--patch-width', default=48,
                    help='width of patch, (H, W, C)')
opt = config.parse_args(['@configuration.txt'])
