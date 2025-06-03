import os, pickle, yaml
import os.path as op
import torch
import numpy as np
from utils import MCGrid
from app import Agent

seed = 2025
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda:0')

agent = Agent()

root_path = op.dirname(op.abspath(__file__))
exp_name = 'demo'
log_path = op.join(root_path, 'results', exp_name)

output_path = op.join(root_path, 'inference', exp_name)
os.makedirs(output_path, exist_ok=True)

agent.load_model(device, log_path, checkpoint='final')
mc_grid = MCGrid({
    'reso': 256,
    'level': 0.,
})
arg = {
    'mc_grid': mc_grid,
    'data_root': '/path/to/your/dataset',
    'output_folder': output_path,
}
agent('ngcnet_inference', arg)