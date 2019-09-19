import torch
import torch.nn as nn
import numpy as np
import time
from clipping import *
import matplotlib.pyplot as plt
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Reference: https://github.com/FlashTek/foolbox/blob/adam-pgd/foolbox/optimizers.py#L31
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
        gradient : `np.ndarray`
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

def image_partition(seed, total_num, partition_size):
    idx_lst = np.arange(total_num)
    np.random.seed(seed)
    np.random.shuffle(idx_lst)
    num_per_partition = int(total_num/partition_size)
    idx_lst = np.split(idx_lst, num_per_partition)
    return idx_lst

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

class StickerSPSA:
    def __init__(self, model, subimg, label, sticker_size=(20, 20), 
                 delta = 0.01, num_samples=128, step_size=0.01):
        self.model = model
        self.clean_subimg = subimg.clone()
        self.adv_subimg = subimg.clone()
        self.label = label
        self.sticker_size = sticker_size
        self.adv_pertub = None
        self.num_samples = num_samples
        self.delta = delta
        self.adam_optimizer = AdamOptimizer(shape=(1, 3)+sticker_size, learning_rate=step_size)
    
    def run(self):
        # Sample perturbation from Bernoulli +/- 1 distribution
        _samples = torch.sign(torch.empty((self.num_samples//2, 3) + self.sticker_size, dtype=self.adv_subimg.dtype).uniform_(-1, 1))
        _samples = _samples.cuda()
        delta_x = self.delta * _samples
        delta_x = torch.cat([delta_x, -delta_x], dim=0) # so there are 2*num_samples
        _sampled_perturb = self.adv_subimg + delta_x

        with torch.no_grad():
            logits = self.model(_sampled_perturb)

        # calculate the margin logit loss
        label_logit = logits[:, self.label].reshape((-1, ))
        value, indices = torch.topk(logits, k=5, dim=1)
        logits[:, self.label] = float('-inf')
        best_other_logit, _ = torch.max(logits, dim=1)
        ml_loss = label_logit - best_other_logit

        # estimate the gradient
        all_grad = ml_loss.reshape((-1, 1, 1, 1)) / delta_x
        est_grad = torch.mean(all_grad, dim=0)

        # update the sticker
        self.adv_subimg += self.adam_optimizer(est_grad[None])

        # clip the perturbation so that it is in a valid range
        adv_pertub = self.adv_subimg - self.clean_subimg 
        self.adv_pertub = torch.clamp(adv_pertub, -1.8044, 2.2489)
        self.adv_subimg = self.clean_subimg + self.adv_pertub

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

def run_sticker_spsa(data_loader, model, num_iter, id2id,
                     wrapper=DynamicClippedPatchAttackWrapper, sticker_size=(20, 20), stride=20, clip_fn=tanh_linear, a=0.05, b=-1):
    for n, (image, label) in enumerate(data_loader):

        # Move the image to GPU and obtain the top-5 prediction on the clean image.
        image = image.to(device)
        true_label = id2id[label.item()]
        logits = model(image)
        logits = torch.mean(logits, dim=(1, 2))
        _, topk = torch.topk(logits, 5, dim=1)
        topk = topk[0].cpu().numpy()
        print(f'clean topk {(label.item(), topk)}')
        mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        std = np.array([[0.229, 0.224, 0.225]]).reshape((3, 1, 1))
        earlyreturn = False
        tic = time.time()
        for x in range(0, 224 - sticker_size[0] + 1, stride):
            if earlyreturn: break
            for y in range(0, 224 - sticker_size[1] + 1, stride):
                if earlyreturn: break
                print(f'Image {n}, current position: {(x, y)}')
                wrapped_model = wrapper(model, image.clone(), sticker_size, (x, y), clip_fn, a, b)
                subimg = get_subimgs(image, (x, y), sticker_size)
                
                spsa_attack = StickerSPSA(wrapped_model, subimg, label, step_size=0.01) # TODO: adjust step size
                for i in range(num_iter):
                    spsa_attack.run()

                # evaluate the sticker
                logits = wrapped_model(spsa_attack.adv_subimg)
                values, topk = torch.topk(logits, 5, dim=1)
                topk = topk[0].cpu().numpy()
                if true_label+100 not in topk: #TODO: remove 100
                    earlyreturn = True
                    print(f"Successfully attack at {(x, y)}")
                    adv_img = image[0].cpu().numpy()
                    adv_subimg = spsa_attack.adv_subimg[0].cpu().numpy()
                    adv_img = apply_sticker(adv_subimg, adv_img, (x, y), sticker_size)
                    adv_img = (adv_img*std) + mean
                    adv_img = adv_img.transpose([1, 2, 0])
                    print(adv_img.min(), adv_img.max())
                    plt.imsave(f'./{n}.png', adv_img)
                else:
                    print(f"Fail to attack at {(x, y)}")
                print(f"label: {true_label}, topk: {topk}")
        tac = time.time()
        print('Time duration for one position: {:.2f} min.'.format((tac - tic)/60))


