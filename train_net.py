import os, argparse
import os.path as op
import numpy as np
import torch
import training, network, data, utils


### Set manual seed for debug
# seed = 2025
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
device = torch.device('cuda:0')

p = argparse.ArgumentParser(description='Input path to config file')
p.add_argument('-c', '--config_path', required=True, help='Path to config file.')

args = p.parse_args()
opt = {
    'config_path': args.config_path,
    'root_path': op.dirname(op.abspath(__file__)),
}
opt = utils.process_options(opt, mode='train')

### Train Dataset
train_dataloader = data.get_dataloader(opt['dataset'], dataset_mode='train')
opt['training']['train_dataloader'] = train_dataloader

### define model
model = network.define_model(opt['model'])
model.to(device)

opt['training']['train_loss'] = training.config_loss(opt['loss'])
training.train_model(opt['training'], model)

if 'post_epochs' in opt['training']:
    training.optimize_code(opt['training'], model)

# save the current config file to the output folder
src_config_path = opt['config_path']
cp_config_path = op.join(opt['log_path'], 'config.yaml')
os.system(f'cp {src_config_path} {cp_config_path}')