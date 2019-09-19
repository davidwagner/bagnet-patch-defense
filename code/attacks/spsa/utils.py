import torch
import torch.nn as nn
import numpy as np

class PatchAttackWrapper(nn.Module):

    def __init__(self, model, img, size, loc, clip_fn=None, a=None, b=None):
        super(PatchAttackWrapper, self).__init__()
        self.batch_size = img.shape[0]
        self.model = model
        self.x1, self.y1 = loc
        self.x2, self.y2 = self.x1 + size[0], self.y1 + size[1]
        mask = torch.ones((1, 3, 224, 224)).cuda()
        mask[:, :, self.x1:self.x2, self.y1:self.y2] = 0
        self.img = (mask*img)

    def forward(self, subimg):
        sticker = self._make_sticker(subimg)
        attacked_img = self.img + sticker
        logits = self.model(attacked_img)
        return logits

    def _make_sticker(self, subimg):
        sticker = torch.zeros((1, 3, 224, 224)).cuda()
        sticker[:, :, self.x1:self.x2, self.y1:self.y2] = subimg
        return sticker

class ClippedPatchAttackWrapper(PatchAttackWrapper):
    def __init__(self, model, img, size, loc, clip_fn, a=None, b=None):
        super().__init__(model, img, size, loc)
        self.clip_fn = clip_fn
        self.a = a
        self.b = b

    def forward(self, subimg):
        sticker = self._make_sticker(subimg)
        attacked_img = self.img + sticker
        logits = self.clip_fn(self.model(attacked_img), self.a, self.b)
        logits = torch.mean(logits, dim=(1, 2))
        return logits

class DynamicClippedPatchAttackWrapper(ClippedPatchAttackWrapper):
    def __init__(self, model, img, size, loc, clip_fn, a=None, b=None):
        super().__init__(model, img, size, loc, clip_fn, a, b)
    
    def _make_sticker(self, subimg):
        bs, _, _, _ = subimg.shape
        sticker = torch.zeros((bs, 3, 224, 224)).cuda()
        sticker[:, :, self.x1:self.x2, self.y1:self.y2] = subimg
        return sticker

def get_subimgs(images, loc, size):
    """Take an of IMAGES, location LOC and SIZE of stickers, 
        return the sub-image that are for adversarial stickers
    Input:
    - image (numpy array): images, shape = (n, 3, 224, 224)
    - loc (tuple): (x1, y1) coordinate of the upper-left corner of stickers
    - size (tuple): (w, h) width and height of stickers
    Output: 
    -subimage (numpy array): subimage shape = (n, 3, w, h)
    """
    x1, y1 = loc
    x2, y2 = x1 + size[0], y1 + size[1]
    return images[:, :, x1:x2, y1:y2]

class AdamOptimizer:
    """Basic Adam optimizer implementation that can minimize w.r.t.
    a single variable.
    Parameters
    ----------
    shape : tuple
        shape of the variable w.r.t. which the loss should be minimized
    """

    def __init__(self, shape, learning_rate,
                 beta1=0.9, beta2=0.999, epsilon=10e-8):
        """Updates internal parameters of the optimizer and returns
        the change that should be applied to the variable.
        Parameters
        ----------
        shape : tuple
            the shape of the image
        learning_rate: float
            the learning rate in the current iteration
        beta1: float
            decay rate for calculating the exponentially
            decaying average of past gradients
        beta2: float
            decay rate for calculating the exponentially
            decaying average of past squared gradients
        epsilon: float
            small value to avoid division by zero
        """

        self.m = torch.zeros(shape).cuda()
        self.v = torch.zeros(shape).cuda()
        self.t = 0

        self._beta1 = beta1
        self._beta2 = beta2
        self._learning_rate = learning_rate
        self._epsilon = epsilon

    def __call__(self, gradient):
        """Updates internal parameters of the optimizer and returns
        the change that should be applied to the variable.
        Parameters
        ----------
        gradient : `pytorch tensor`
            the gradient of the loss w.r.t. to the variable
        """

        self.t += 1

        self.m = self._beta1 * self.m + (1 - self._beta1) * gradient
        self.v = self._beta2 * self.v + (1 - self._beta2) * gradient**2

        bias_correction_1 = 1 - self._beta1**self.t
        bias_correction_2 = 1 - self._beta2**self.t

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2

        return -self._learning_rate * m_hat / (torch.sqrt(v_hat) + self._epsilon)

def apply_sticker(adv, img, loc, size):
    """
    Input:
    - adv (np array): adversarial sticker, shape=(3, h, w), pixel ~ (0, 1)
    - img (np array): original image, shape=(3, 224, 224), pixel ~ (0, 1)
    - loc (list):
    - size (int):
    """
    x1, y1 = loc
    x2, y2 = x1 + size[0], y1 + size[1]
    sticker = np.zeros_like(img).astype(img.dtype)
    sticker[:, x1:x2, y1:y2] = adv
    mask = np.ones_like(img).astype(img.dtype)
    mask[:, x1:x2, y1:y2] = 0.
    img = img * mask
    return img + sticker

