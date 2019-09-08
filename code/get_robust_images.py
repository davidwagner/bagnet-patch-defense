import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from bagnets.utils import *
from bagnets.clipping import *
from bagnets.security import *
import torchvision.transforms as transforms
import foolbox_alpha
import foolbox
from foolbox.attacks import ProjectedGradientDescentAttack as PGD
from foolbox.criteria import TopKMisclassification
from foolbox.criteria import TargetClass
from foolbox.distances import Distance
from foolbox.adversarial import Adversarial
import logging
from advertorch.attacks import LinfPGDAttack

def foolbox_get_robust(model, wrapper, data_loader, attack_size, 
                        clip_fn, a=None, b=None, stride=1, k=5, 
                        attack_alg=PGD, targeted=False, case='avg',
                        max_iter=40, eps=1., stepsize=1/40, return_early=True, random_start=True,
                        mean=np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)), 
                        std=np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)), 
                        output_root = './foolbox_results'):
    """ Take in a pytorch data loader, use the ATTACK_ALG to
        calculate the security upper bound.
    Input:
    - model (pytorch model): model wrapped for sticker attack 
    - wrapper (pytorch model): model wrapper for sticker attack
    - data_loader (pytorch DataLoader): batchsize=1, pytorch dataloader in CPU, without normalization, pixel ~ [0, 1]
    - attack_size (iterable): sticker size
    - stride (int): stride of how stickers move
    - k (int): top-k misclassification criteria
    - clip_fn (callable): clipping function (None if don't apply)
    - attack_alg (foolbox.attack): attack algorithm implemented in foolbox
    - targeted (boolean): whether apply a targeted attack,
    - case (string): ['avg', 'worst', 'best']: sampling strategy for targeted attacks
    - max_iter (int): number of max iterations
    - eps (float): from 0 to 1, percentage of perturbation
    - stepsize (float): optimizer stepsize
    - return_early (boolean): apply early return in attack
    - random_start (boolean): apply random initialization in attack
    - mean, std (numpy array): mean and standard devication of dataset
    - output_path (str): directory for saving resulting plots
    Output:
    - succ_prob (float): fraction of images whose prediction are not affected by the attack
    """
    num_robust_imgs, total_images = 0, 0
    imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view((3, 1, 1))
    imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view((3, 1, 1))
    for images, labels in data_loader:
        prep_images = (images - imagenet_mean) / imagenet_std
        _images = prep_images.cuda()
        if targeted: # if targeted, calculate logits for sampling targeted class
            with torch.no_grad():
                if clip_fn is None:
                    logits = model(_images)
                else:
                    logits = clip_fn(model(_images), a, b)
                    logits = torch.mean(logits, dim=(1, 2))
                
        image, label = images[0].numpy(), labels[0].item()
        c, h, w = image.shape
        total_images += 1
        is_robust = True
        for x in range(0, h - attack_size[0] + 1, stride):
            for y in range(0, w - attack_size[1] + 1, stride):
                wrapped_model = wrapper(model, prep_images, attack_size, (x, y), clip_fn, a, b)
                wrapped_model.eval()
                print('image {}, current location: {}'.format(total_images, (x, y)))
                fmodel = foolbox.models.PyTorchModel(wrapped_model, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
                if not targeted:
                    criterion = TopKMisclassification(k)
                else:
                    targeted_class = pick_targeted_classes(logits, k=1, case=case)[0].item()
                    criterion = TargetClass(targeted_class)
                attack = attack_alg(fmodel, criterion=criterion, distance=foolbox.distances.Linfinity)
                subimg = get_subimg(image, (x, y), attack_size)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    adversarial = attack(subimg, label, iterations = max_iter, epsilon=eps, stepsize=stepsize, random_start=random_start, return_early=return_early, binary_search=False)
                    
                if adversarial is not None:
                    msg = 'Image {}, attack successfully, location {}'.format(total_images, (x, y))
                    print(msg)
                    is_robust = False
                    break
                else:
                    print('Image {}: fail to find an adversarial sticker at {}'.format(total_images, (x, y)))
            if not is_robust:
                break
        if is_robust:
            num_robust_imgs += 1
            save_path = os.path.join(output_root, str(label))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            image = image.transpose([1, 2, 0])
            plt.imsave(os.path.join(save_path, '{}-{}.png'.format(label, total_images)), image)
    succ_prob = num_robust_imgs / total_images
    return succ_prob
