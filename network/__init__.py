import os, sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

import os.path as op
import yaml, torch
import model


def define_model(opt):
    if hasattr(model, opt['type']):
        return getattr(model, opt['type'])(opt)

    return None


def load_model(device, log_path, checkpoint='final', strict=False):
    config_path = op.join(log_path, 'config.yaml')
    opt = yaml.safe_load(open(config_path))

    if checkpoint == 'final' or checkpoint == 'post':
        ckpt_name = f'model_{checkpoint}.pth'
    else:
        ckpt_name = 'model_epoch_%04d.pth' % int(checkpoint)

    model = define_model(opt['model'])
    checkpoint_path = op.join(log_path, f'checkpoints/{ckpt_name}')
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device),
        strict=strict
    )
    model.to(device)
    return model, opt

def save_vae_latents(model, dataset, output_path, device):
    num_shape = len(dataset)

    latent_codes = []
    with torch.no_grad():
        for i in range(num_shape):
            model_input = dataset.get_input(i).to(device)
            # vae: (mean, logvar), shape(2, 1, Nf)
            vae = model.encoder.inference_vae(model_input)
            vae = vae.detach().cpu()
            latent_codes.append(vae)

    latent_codes = torch.stack(latent_codes)
    print('latent_codes: ', latent_codes.shape)
    output_file = op.join(output_path, 'vae_latents.pt')
    torch.save(latent_codes, output_file)