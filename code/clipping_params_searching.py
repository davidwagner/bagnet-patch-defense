from advertorch.attacks import LinfPGDAttack
import bagnets
from bagnets.clipping import*
from bagnets.security import*
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import time
import os
import pickle
import logging
from absl import app, flags
from time import gmtime, strftime
import json

FLAGS = flags.FLAGS
flags.DEFINE_integer('N', 50, 'number of images')
flags.DEFINE_integer('seed', 42, 'random seed for sampling images from ImageNet')
flags.DEFINE_multi_integer('attack_size', [20, 20], 'size of sticker')
flags.DEFINE_integer('stride', 20, 'stride of sticker')
flags.DEFINE_string('model', 'bagnet33', 'model being evaluated')
flags.DEFINE_string('clip_fn', 'tanh_linear', 'clipping function')
flags.DEFINE_string('param', 'a', 'clip(a*x + b). Which parameter are going to be tested')
flags.DEFINE_multi_integer('param_list', None, 'list of parameters to be tested')
flags.DEFINE_float('fixed_param', None, 'the other fixed parameter')
flags.DEFINE_string('data_path', '/mnt/data/imagenet', 'directory where data are stored')
flags.DEFINE_string('output_root', '/mnt/data/clipping_params_searching/', 'directory for storing results')

def main(argv):
    """
    FLAGS.output_root/
        [NAME]/
            [NAME].log
            [Name].lst
    """
    assert FLAGS.param in ['a', 'b'], 'FLAGS.param must be either a or b'
    NAME = '{}-{}-{}-{}x{}-{}-{}-{}'.format(FLAGS.model, FLAGS.N, FLAGS.seed, FLAGS.attack_size[0], FLAGS.attack_size[1], FLAGS.stride, FLAGS.clip_fn, FLAGS.param)
    LOG_NAME = NAME+'.log'
    LOG_PATH = os.path.join(FLAGS.output_root, LOG_NAME)

    print("log to {}".format(LOG_PATH))

    logger = logging.basicConfig(filename=LOG_PATH, level=logging.INFO)
    logging.info(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    logging.info("Setting:\n model: {}, N: {}, seed: {}, attack_size: {}x{}, stride: {}, clip_fn: {}, param: {}".format(FLAGS.model, FLAGS.N, FLAGS.seed, FLAGS.attack_size[0], FLAGS.attack_size[1], FLAGS.stride, FLAGS.clip_fn, FLAGS.param))
    logging.info("List of evaluated parameter values: {}".format(FLAGS.param_list))

    ###################################
    # Model and data preparation
    ###################################

    # GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        print("GPU: {}".format(torch.cuda.get_device_name(0)))
        logging.info("GPU: {}".format(torch.cuda.get_device_name(0)))
    else:
        print(device)
        logging.info(device)
    
    # ImageNet validation set
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    imagenet_transform = transforms.Compose([transforms.Resize(256), 
                                             transforms.CenterCrop(224), 
                                             transforms.ToTensor(), 
                                             normalize])
    imagenet_val = datasets.ImageNet(FLAGS.data_path, split='val', download=False, 
                                         transform=imagenet_transform)

    np.random.seed(FLAGS.seed)
    val_subset_indices = np.random.choice(np.arange(50000), size=FLAGS.N, replace=False)
    val_subset_loader = torch.utils.data.DataLoader(imagenet_val, 
                                                    batch_size=1,
                                                    num_workers=4,
                                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(val_subset_indices))

    # load pretrained model
    clip_fn_dic = {"tanh_linear":tanh_linear, 
                   "sigmoid_linear":sigmoid_linear,
                   "binarize":binarize,
                   "None": None}

    model_dic = {"bagnet9":bagnets.pytorch.bagnet9(pretrained=True, avg_pool=False),
                 "bagnet17":bagnets.pytorch.bagnet17(pretrained=True, avg_pool=False),
                 "bagnet33":bagnets.pytorch.bagnet33(pretrained=True, avg_pool=False)}

    clip_fn = clip_fn_dic[FLAGS.clip_fn]
    model = model_dic[FLAGS.model]
    model = model.to(device)
    model.eval()

    acc_list = []

    if FLAGS.param = 'a':
        for param in FLAGS.param_list:
            msg = "Threshold: a = {}".format(param)
            print(msg)
            logging.info(msg)
            step_acc = validate(val_subset_loader, model, device, clip = clip_fn, a = param, b = FLAGS.fixed_param)
            acc_list.append(step_acc)

    else:
        for param in FLAGS.param_list:
            msg = "Threshold: b = {}".format(param)
            print(msg)
            logging.info(msg)
            step_acc = validate(val_subset_loader, model, device, clip = clip_fn, a = FLAGS.fixed_param, b = param)
            acc_list.append(step_acc)

    with open(os.path.join(OUTPUT_PATH, NAME+'.lst'), 'wb') as file:
        pickle.dump(acc_list, file)

if __name__ == "__main__":
    app.run(main)
