import torch


def _b(uncertainty):
    return uncertainty + 0.05

def lrgb(infered, uncertainty, gt):
    u = _b(uncertainty)
    s = torch.square((infered - gt).norm(dim=-1, p=2)) / (2 * torch.square(u)) + (torch.log(u + 3) / 2)
    return torch.mean(s)


class SatNerfLoss:
    def __call__(self, infered, uncertainty, gt):
        return lrgb(infered, uncertainty, gt)  # + lmbdsc * lsc() + lmbdds * lds()
