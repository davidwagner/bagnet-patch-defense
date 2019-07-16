from advertorch.attacks import LinfPGDAttack
from bagnets.clipping import*
from bagnets.security import*
import foolbox
from foolbox.utils import samples
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import time
import os
import pickle
import logging
from absl import flags

flags.DEFINE_integer('seed', 42, 'random seed for sampling images from ImageNet')
flags.DEFINE_string('clip_fn', 'tanh_linear', 'clipping function')
flags.DEFINE_float('a', 0.05, 'clipping parameter A')
flags.DEFINE_float('b', -1, 'clipping parameter B')
flags.DEFINE_float('eps', 5., 'range of perturbation')
flags.DEFINE_integer('nb_iter', 40, 'number of iterations for PGD')
flags.DEFINE_string('output_root', './results', 'directory for storing results')
FLAGS = flags.FLAGS

clip_fn_dic = {"tanh_linear":tanh_linear, 
               "binarize":binarize}

clip_fn = clip_fn_dic[FLAGS.clip_fn]

OUTPUT_PATH = os.path.join(FLAGS.output_root, '{}-{}-{}-{}-{}-{}-{}'.format(FLAGS.seed, FLAGS.clip_fn, FLAGS.a, FLAGS.b, FLAGS.eps, FLAGS.nb_iter, FLAGS.stepsize)) 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


if __name__ == "__main__":
    if use_cuda:
        print(torch.cuda.get_device_name(0))

    seed = 42
    clip_fn, a, b = tanh_linear, 0.05, -1

    N = 50
    meta_batchsize = 35
    attack_size = (20, 20)
    stride = 20
    eps, nb_iter, stepsize = 5., 100, 0.5
    max_iter = ((224 - attack_size[0]) // stride + 1) * ((224 - attack_size[1]) // stride + 1)
    print('max iter: {}'.format(max_iter))

    with torch.no_grad():
    count = 0
    for image, label in val_subset_loader:
	image = image.to(device)
        logit = bagnet33(image)
        if clip_fn:
            logit = clip_fn(logit, a, b)
        logit = torch.mean(logit, dim=(1, 2))
        _, topk = torch.topk(logit, k=5, dim=1)
        topk, label = topk.cpu().numpy()[0], label.numpy()[0]
        if label in topk:
            count += 1
    print(count / N)

    data_iter = iter(val_subset_loader)
    metabatch = MetaBatch(data_iter, meta_batchsize, max_iter)

    tic = time.time()
    succ_prob = batch_upper_bound(bagnet33, metabatch, 
                                  clip_fn, a, b,
                                  attack_size=attack_size, eps=eps, nb_iter=nb_iter, stepsize=stepsize, 
                                  stride=stride, k=5)
    tac = time.time()
    print("Success probability: {}, Time: {:.3f}s or {:.3f}hr(s)".format(succ_prob, tac - tic, (tac-tic)/3600))

    for key, value, in metabatch.fail_list.items(): # (image, label, attack location)
        image, label, loc = value
        x1, y1 = loc
        x2, y2 = x1 + 20, y1 + 20
        adv = metabatch.adv[key].clone()
        image = image[None].clone()
        image[:, :, x1:x2, y1:y2] = adv[None]
        logits = bagnet33(image.to(device))
        if clip_fn:
            logits = clip_fn(logits, a, b)
        logits = torch.mean(logits, dim=(1, 2))
        _, topk = torch.topk(logits, k=5, dim=1)
        image = undo_imagenet_preprocess(image[0]).cpu().numpy() # undo preprocessing
        image = convert2channel_last(image)
    #     print(image.min(), image.max())
        if not os.path.exists('./results/advertorch-attack/{}-{}-{}-{}-{}-{}/{}'.format(seed, a, b, eps, nb_iter, stepsize, label)):
            os.mkdir('./results/advertorch-attack/{}-{}-{}-{}-{}-{}/{}'.format(seed, a, b, eps, nb_iter, stepsize, label))
        
        plt.imsave('./results/advertorch-attack/{}-{}-{}-{}-{}-{}/{}/{}.png'.format(seed, a, b, eps, nb_iter, stepsize, label, key), image)
        plt.imshow(image)
        plt.title('{} {}'.format(label, idx2label[label]))
        plt.show()
        print(topk, label)
    
    metabatch.waitlist = None
    with open('./results/advertorch-attack/metabatch-{}_images-{}_{}-{}-{}-{}-{}.mtb'.format(N, seed, a, b, eps, nb_iter, stepsize), 'wb') as file:
        pickle.dump(metabatch, file)
