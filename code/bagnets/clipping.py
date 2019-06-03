import torch

def clip_below(values, a):
    """Clip values below b
        Input:
        - values(torch tensor): values to be clipped
        Output: (torch tensor) clipped values
    """
    thresh_below = torch.full_like(values, a)
    return torch.where(values >= a, values, thresh_below)

def binarize(values, a):
    """Set values below A to 0 and those above to 1
    Input:
    - values(torch tensor): values to be clipped
    Output: (torch tensor) clipped values
    """
    all_ones = torch.ones_like(values)
    all_zeros = torch.zeros_like(values)
    return torch.where(values >= a, all_ones, all_zeros)

def clip_pm1(values, **kwargs):
    """Clip values to [-1, 1]
    Input:
    - values(torch tensor): values to be clipped
    Output: (torch tensor) clipped values
    """
    return torch.clamp_(values, -1., 1.)

def clip_bias(values, b):
    """Clip values to [-1, 1]
    Input:
    - values(torch tensor): values to be clipped
    - b (float): intersection
    Output: (torch tensor) clipped values
    """
    return torch.clamp_(values + b, -1, 1)

def clip_linear(values, a, b):
    """Clip values to [-1, 1]
    Input:
    - values(torch tensor): values to be clipped
    - a (float): coefficient
    - b (float): intersection
    Output: (torch tensor) clipped values
    """
    return torch.clamp_(values * a + b, -1, 1)

def sigmoid_linear(values, a, b):
    """Clip values to [-1, 1]
    Input:
    - values(torch tensor): values to be clipped
    - a (float): coefficient
    - b (float): intersection
    Output: (torch tensor) clipped values
    """
    x_lin = values * a + b
    return torch.sigmoid(x_lin)

def tanh_linear(values, a, b):
    """Clip values to [-1, 1]
    Input:
    - values(torch tensor): values to be clipped
    - a (float): coefficient
    - b (float): intersection
    Output: (numpy array) clipped values
    """
    return torch.tanh(values * a + b)

def clip_logits(bagnet, clip, images, **kwargs):
    """Clip logits returned by patches
    Input:
    - bagnet (pytorch model): Bagnet without average pooling
    - clip (python function): clip function
    - images (pytorch tensor): 
    """
    with torch.no_grad():
        patch_logits = bagnet(images)
    return clip(patch_logits, **kwargs)
