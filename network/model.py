import math
import torch
import torch.nn as nn

from mlp import *
from pos_encoding import PosEncoding


class NGCNet(nn.Module):
    """docstring for NGCNet ."""
    def __init__(self, arg):
        super().__init__()
        self.dim_code = arg['dim_code']
        self.dim_feat = arg['dim_feat']
        
        self.embd = nn.Embedding(arg['n_curve'], self.dim_code)
        self.init_embedding(self.embd, self.dim_code)
        
        self.encoder = FeatCurveEncoder(arg)
        if arg['num_pos_encoding'] > 0:
            self.pos_enc = PosEncoding(arg['num_pos_encoding'])
            diff = self.pos_enc.d_out - self.pos_enc.d_in
            arg['decoder_curve']['size'][0] += diff
        self.decoder = MLP(**arg['decoder_curve'])


    def init_embedding(self, embedding, dim_code):
        nn.init.normal_(embedding.weight.data, 0., 1./ math.sqrt(dim_code))

    
    def set_post_mode(self):
        for pname, param in self.core.named_parameters():
            if 'embd' in pname:
                continue

            param.requires_grad = False


    def forward(self, model_input):
        mi = model_input
        # curve_idx:(Nb, Ns); coords:(Nb, Ns); samples(Nb,Ns,3)
        curve_code = self.embd(mi['curve_idx'])
        curve_feats = self.encoder(curve_code, mi['coords'])
        if hasattr(self, 'pos_enc'):
            samples = self.pos_enc(mi['samples'])
        else:
            samples = mi['samples']
        curve_feats = torch.cat([curve_feats, samples], dim=-1)
        
        curve_sdf = self.decoder.forward_simple(curve_feats).squeeze(-1)
        return {
            'sdf': curve_sdf,
            'code': curve_code,
        }
    
    @torch.no_grad()
    def inference(self, model_input):
        curve_input = self.pack_data(model_input)
        ci = curve_input
        curve_code = self.embd(ci['curve_idx'])

        curve_feats = self.encoder(curve_code, ci['coords'])
        if hasattr(self, 'pos_enc'):
            samples = self.pos_enc(ci['samples'])
        else:
            samples = ci['samples']
        curve_feats = torch.cat([curve_feats, samples], dim=-1)

        curve_sdf = self.decoder.forward_simple(curve_feats).squeeze(-1)
        return curve_sdf

    @torch.no_grad()
    def mix_curve(self, model_input):
        # assume only one curve
        mi = model_input
        curve_data = self.pack_data(mi)
        curve_idx = curve_data['curve_idx']
        curve_code = self.embd(curve_idx)
        new_idx = mi['new_idx']* torch.ones_like(curve_idx, dtype=int).to(curve_code.device)
        new_code = self.embd(new_idx)

        curve_feats = self.encoder.mix_feats(
            curve_code, new_code, curve_data['coords'], mi)
        
        if hasattr(self, 'pos_enc'):
            samples = self.pos_enc(curve_data['samples'])
        else:
            samples = curve_data['samples']
        curve_feats = torch.cat([curve_feats, samples], dim=-1)
        curve_sdf = self.decoder.forward_simple(curve_feats).squeeze(-1)
        return curve_sdf


    def pack_data(self, model_input):
        mi = model_input
        device = mi['device']

        n_sample = mi['coords'].shape[0]
        curve_idx = mi['curve_idx']*torch.ones(n_sample, dtype=int)
        res = {
            'samples': torch.from_numpy(mi['samples_local']).float().to(device),
            'coords': torch.from_numpy(mi['coords']).float().to(device),
            'curve_idx': curve_idx.to(device)
        }

        res = {key:val.unsqueeze(0) for key,val in res.items()}
        return res


class FeatCurveEncoder(nn.Module):
    """docstring for FeatCurveEncoder."""
    def __init__(self, arg):
        super(FeatCurveEncoder, self).__init__()
        self.dim_code = arg['dim_code']
        self.dim_feat = arg['dim_feat']

        self.linear1 = nn.Linear(self.dim_code, 2*self.dim_feat)
        self.mlp = MLP(**arg['encoder_curve'])
    
    def forward(self, code, coords):
        # code(Nb, Ns, N_code); coords(Nb, Ns)
        Nb, Ns = code.shape[0], code.shape[1]
        end_feats = self.linear1(code).view(Nb, Ns, self.dim_feat, 2)
        # weights (Nb, Ns, 2)
        weights = torch.stack([coords, 1-coords], dim=-1)
        feat_curve = torch.einsum('bnjk,bnk->bnj', end_feats, weights)
        return self.mlp.forward_simple(feat_curve)
   
    def mix_feats(self, code1, code2, coords, arg):
        func1 = arg['mix_func1']
        func2 = arg['mix_func2']

        ts1, weights1 = func1(coords)
        ts2, weights2 = func2(coords)
        feat1 = self.forward(code1, ts1)
        feat2 = self.forward(code2, ts2)
        feat = feat1*weights1[...,None] + feat2*weights2[...,None]
        return feat


class DeepSDF(nn.Module):
    """docstring for DeepSDF."""
    def __init__(self, arg):
        super(DeepSDF, self).__init__()
        self.dim_code = arg['dim_code']
        self.dim_feat = arg['dim_feat']
        
        self.embd = nn.Embedding(arg['n_shape'], self.dim_code)
        self.init_embedding(self.embd, self.dim_code)
        if arg['num_pos_encoding'] > 0:
            self.pos_enc = PosEncoding(arg['num_pos_encoding'])
            diff = self.pos_enc.d_out - self.pos_enc.d_in
            arg['decoder']['size'][0] += diff
        self.decoder = MLP(**arg['decoder'])

    def init_embedding(self, embedding, dim_code):
        nn.init.normal_(embedding.weight.data, 0., 1./ math.sqrt(dim_code))


    def forward(self, model_input):
        mi = model_input
        idx = mi['idx']
        # codes: (Nb, 256)
        codes = self.embd(idx)

        if hasattr(self, 'pos_enc'):
            samples = self.pos_enc(mi['samples'])
        else:
            samples = mi['samples']

        # samples: (Nb, Ns, 3)
        Ns = samples.shape[1]
        codes = codes.repeat(1, Ns, 1)
        feats = torch.cat([codes, samples], dim=-1)
        sdfs = self.decoder.forward_simple(feats).squeeze(-1)
        return {
            'sdf': sdfs,
            'code': codes,
        }

    def inference(self, model_input):
        mi = model_input
        idx = mi['idx']
        codes = self.embd(idx)

        if hasattr(self, 'pos_enc'):
            samples = self.pos_enc(mi['samples'])
        else:
            samples = mi['samples']

        # samples: (Nb, Ns, 3)
        Ns = samples.shape[1]
        codes = codes[:,None,:].repeat(1, Ns, 1)
        feats = torch.cat([codes, samples], dim=-1)
        sdfs = self.decoder.forward_simple(feats).squeeze(-1)
        return sdfs


if __name__ == "__main__":
    pass