import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calc_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

class LossHandler():
    def __init__(self):
        self.metric_types = ['BCE', 'L1', 'L2', 'cos_sim', 'None']

    def get_metric_fn(self, opt):
        metric = opt['metric']
        if metric not in self.metric_types:
            raise NameError('Not supported metric')

        loss = None
        if metric == 'BCE':
            loss = nn.BCELoss()
        elif metric == 'L1':
            loss = nn.L1Loss()
        elif metric == 'L2':
            loss = nn.MSELoss()
        elif metric == 'cos_sim':
            loss = nn.CosineSimilarity(eps=1e-6)
        elif metric == 'None':
            loss = None
        else:
            raise NotImplementedError('Not implemented metric')

        return loss


    def parse_config(self, loss_schedule):
        self.loss_fn = {}
        for name, loss_config in loss_schedule.items():
            metric_fn = self.get_metric_fn(loss_config)
            self.loss_fn[name] = {
                'metric_fn': metric_fn, 
                'factor': loss_config['factor']
            }

    def sdf_loss(self, output, gt, metric_fn):
        out_sdf = output['sdf']
        gt_sdf = gt['sdf'].view_as(out_sdf)
        return metric_fn(out_sdf, gt_sdf)

    def code_loss(self, output, gt, metric_fn):
        code = output['code']
        reg_loss = torch.sum(torch.pow(code, 2), dim=-1)
        return torch.mean(reg_loss)
    
    def nodes_loss(self, output, gt, metric_fn):
        nodes = output['nodes']
        nodes_gt = gt['nodes']

        return metric_fn(nodes, nodes_gt)
    
    def mat_loss(self, output, gt, metric_fn):
        mat = output['mat']
        mat_gt = gt['mat']

        return metric_fn(mat, mat_gt)
    
    def kl_loss(self, output, gt, metric_fn):
        # already calculated in model
        kl_reg = output['kl']
        return torch.mean(kl_reg)
    
    def edm_loss(self, output, gt, metric_fn):
        loss = output['loss']
        return loss.mean()
    
    def occ_sum_loss(self, output, gt, metric_fn):
        # pred(Nc, Ns), gt(Ns)
        occ_pred = output['occ']
        occ_gt = gt['occ']

        occ_sum = torch.sum(occ_pred, dim=0)
        return metric_fn(occ_sum, occ_gt)
    
    def occ_max_loss(self, output, gt, metric_fn):
        # pred(Nc, Ns), gt(Ns)
        occ_pred = output['occ']
        occ_gt = gt['occ']

        occ_max,_ = torch.max(occ_pred, dim=0)
        return metric_fn(occ_max, occ_gt)
    

    def __call__(self, output, gt):
        res = {}
        for name, loss in self.loss_fn.items():
            if hasattr(self, name):
                func = getattr(self, name)
            else:
                raise NameError('Not defined loss name')
            loss_term = func(output, gt, loss['metric_fn'])
            res[name] = loss['factor']*loss_term

        return res

