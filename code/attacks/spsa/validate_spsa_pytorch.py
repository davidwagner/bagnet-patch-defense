import foolbox
import foolbox_alpha
from foolbox_alpha.criteria import TargetClass
import torch
from foolbox_alpha.attacks import AdamRandomPGD
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from StickerSPSAObj import *

class AdamOptimizer:
    """Basic Adam optimizer implementation that can minimize w.r.t.
    a single variable.
    Parameters
    ----------
    shape : tuple
        shape of the variable w.r.t. which the loss should be minimized
    """

    def __init__(self, shape, data_type, learning_rate,
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

        self.m = torch.zeros(shape, dtype=data_type).cuda()
        self.v = torch.zeros(shape, dtype=data_type).cuda()
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
        #TODO: remove print
        #print(f"self.m: min: {torch.min(self.m)}, max: {torch.max(self.m)}")
        #print(f"self.v: min: {torch.min(self.v)}, max: {torch.max(self.v)}")
        
        bias_correction_1 = 1 - self._beta1**self.t
        bias_correction_2 = 1 - self._beta2**self.t

        #TODO:
        #print(f"bias_correction_1: {bias_correction_1}")
        #print(f"bias_correction_2: {bias_correction_2}")

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2
        #TODO: remove print
        #print(f"m_hat: min: {torch.min(m_hat)}, max: {torch.max(m_hat)}")
        #print(f"v_hat: min: {torch.min(v_hat)}, max: {torch.max(v_hat)}")

        return -self._learning_rate * m_hat / (torch.sqrt(v_hat) + self._epsilon)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    print(torch.cuda.get_device_name(0))


image_size = (2, 2)

#TODO: Finish implementation
def spsa_run(model, input_image, label, image_size, num_samples=2049,  delta=0.01):
    # Sample perturbation from Bernoulli +/- 1 distribution
    _samples = torch.sign(torch.empty((num_samples//2, 3) + image_size, dtype=input_image.dtype).uniform_(-1, 1))
    _samples = _samples.cuda()
    delta_x = delta * _samples
    delta_x = torch.cat([delta_x, -delta_x], dim=0) 
    #print(f'delta_x: {delta_x}')
    _sampled_perturb = input_image + delta_x
    #print(f'sampled_perturb: {_sampled_perturb}')

    with torch.no_grad():
        logits = model(_sampled_perturb)

    # calculate the margin logit loss
    label_logit = logits[:, label].reshape((-1, )).clone()
    #print(f'{(label_logit.min().item(), label_logit.max().item())}')
    logits[:, label] = float('-inf')
    #print(f'{(label_logit.min().item(), label_logit.max().item())}')
    best_other_logit, _ = torch.max(logits, dim=1)
    ml_loss = label_logit - best_other_logit
    #print(f'ml_loss: {ml_loss}')

        # estimate the gradient
    all_grad = ml_loss.reshape((-1, 1, 1, 1)) / delta_x
    #print(f'all_grad: {all_grad}')
    est_grad = torch.mean(all_grad, dim=0) 
        #TODO: REMOVE PRINT
    print(f'SPSA: gradient: \n{est_grad}')
    adam_grad = adam_optimizer(est_grad[None])
    print(f'SPSA: adam gradient: \n{adam_grad}')
    output_image = input_image + adam_grad
    return output_image

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
    def forward(self, image):
        logitA = torch.sum(image, dim = (1, 2, 3))
        logitB = torch.sum(-image, dim = (1, 2, 3))
        logits = torch.stack((logitA, logitB), dim=1)
        return logits

model = ToyModel()
model.to(device)
model.eval()

base_image = np.zeros((3, )+image_size)
small_image = base_image.copy()
small_image[:, :, :] = 1
print(f'image:\n {small_image}')

with torch.no_grad():
    clean_logits = model(torch.from_numpy(small_image[None]))
    clean_logits
print(f'clean logits: {clean_logits}')


fmodel = foolbox_alpha.models.PyTorchModel(model, bounds=(-1, 1), num_classes=2)
criterion = TargetClass(1)
attack = AdamRandomPGD(fmodel, criterion=criterion, distance=foolbox.distances.Linfinity)
adversarial = attack(small_image, 0, iterations = 200, epsilon=1, stepsize= 0.01, random_start=False, return_early=True, binary_search=False)

#print(f'adversarial image by pgd:\n {adversarial}')


image = torch.from_numpy(small_image[None]).to(device)
adam_optimizer = AdamOptimizer(shape=(1, 3)+image_size, data_type=image.dtype, learning_rate=0.01)

for _ in range(10):
    image = spsa_run(model, image, 0, image_size)
#print(f'adversarial image by spsa:\n {image}')
