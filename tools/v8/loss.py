"""V8 losses: analytic velocity regression only, no score / DMD / GAN."""
import torch.nn.functional as F


def cfm_bfm_loss(v_pred, v_target):
    return F.mse_loss(v_pred.float(), v_target.float())
