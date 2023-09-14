import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from act.act_builder import ACTBuilder


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        self.model, self.optimizer = ACTBuilder.build(args_override)
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        image = self.custom_normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = self.kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image) # no action, sample from prior
            return a_hat

    def custom_normalize(self, image):
        # Separate the RGB and the keypoint channels
        rgb = image[:, :, :3]
        keypoints = image[:, :, 3:4]

        # Normalize only the RGB channels
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Applying normalization across the num_cam dimension
        normalized_rgb = torch.stack([normalize(cam) for cam in rgb.split(1, dim=1)], dim=1).squeeze(2)

        # Combine normalized RGB with untouched keypoints channel
        return torch.cat([normalized_rgb, keypoints], dim=2)

    def configure_optimizers(self):
        return self.optimizer

    def kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld