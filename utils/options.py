import os, yaml
import os.path as op

def process_options(opt, mode='train'):
    res = yaml.safe_load(open(opt['config_path']))
    res['config_path'] = opt['config_path']
    res['root_path'] = opt['root_path']
    logging_root = res['logging_root']
    exp_name = res['experiment_name']
    res['log_path'] = op.join(opt['root_path'], logging_root, exp_name)
    if mode == 'train':
        train_opt = res['training']
        train_opt['log_path'] = res['log_path']
        train_opt['loss'] = res['loss']

    # res['dataset']['handle_file'] = res['handle_file']
    # res['model']['handle_file'] = res['handle_file']

    return res

def diffusion_options(opt):
    res = yaml.safe_load(open(opt['config_path']))
    res['config_path'] = opt['config_path']
    res['root_path'] = opt['root_path']
    logging_root = res['logging_root']
    exp_name = res['experiment_name']
    res['log_path'] = op.join(opt['root_path'], logging_root, exp_name)

    return res