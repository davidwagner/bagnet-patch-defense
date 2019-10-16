import foolbox
import foolbox_alpha
from foolbox_alpha.criteria import TargetClass
import torch
from foolbox_alpha.attacks import AdamRandomPGD
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from StickerSPSAObj import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    print(torch.cuda.get_device_name(0))


#TODO: Finish implementation
def spsa_run(model, input_image, label, image_size, num_samples=128,  delta=0.01):
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

image_size = (2, 2)

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

#print(f'adversarial image:\n {adversarial}')


image = torch.from_numpy(small_image[None]).to(device)
spsa_run(model, image, 0, image_size)
