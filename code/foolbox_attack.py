from advertorch.attacks import LinfPGDAttack
import bagnets
from bagnets.clipping import*
from bagnets.security import*
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import time
import os
import pickle
import logging
from foolbox.attacks import ProjectedGradientDescentAttack as PGD
from foolbox_alpha.attacks import AdamRandomPGD
from absl import app, flags
from time import gmtime, strftime
import json

FLAGS = flags.FLAGS
flags.DEFINE_integer('N', 500, 'number of images')
flags.DEFINE_boolean('rand_init', True, 'randomly init stickers')
flags.DEFINE_boolean('return_early', True, 'early return')
flags.DEFINE_integer('chunkid', 10, 'index of partition chunk')
flags.DEFINE_multi_integer('attack_size', [20, 20], 'size of sticker')
flags.DEFINE_integer('stride', 20, 'stride of sticker')
flags.DEFINE_string('model', 'bagnet33', 'model being evaluated')
flags.DEFINE_string('clip_fn', 'tanh_linear', 'clipping function')
flags.DEFINE_float('a', 0.05, 'clipping parameter A')
flags.DEFINE_float('b', -1, 'clipping parameter B')
flags.DEFINE_string('attack_alg', 'PGD', 'attack algorithm')
flags.DEFINE_float('eps', 1., 'percentage of perturbation')
flags.DEFINE_integer('nb_iter', 40, 'number of iterations for PGD')
flags.DEFINE_float('stepsize', 1/40, 'stepsize of PGD')
flags.DEFINE_string('data_path', '/mnt/data/imagenet', 'data directory')
flags.DEFINE_string('output_root', '/mnt/data/results/foolbox_results', 'directory for storing results')

def main(argv):
    """
    FLAGS.output_root/
        [NAME]/
            [NAME].mtb
            [NAME].log
            dataset/
    """
    NAME = '{}-{}-{}x{}-{}-{}-{}-{}-{}-{}-{}'.format(FLAGS.N, FLAGS.chunkid, FLAGS.attack_size[0], FLAGS.attack_size[1], FLAGS.stride, FLAGS.model, FLAGS.clip_fn, FLAGS.attack_alg, FLAGS.eps, FLAGS.nb_iter, FLAGS.stepsize)
    OUTPUT_PATH = os.path.join(FLAGS.output_root, NAME)
    print(OUTPUT_PATH)

    if not os.path.exists(OUTPUT_PATH):
                os.mkdir(OUTPUT_PATH)
    
    LOG_PATH = os.path.join(OUTPUT_PATH, NAME+'.log')
    print("log to {}".format(LOG_PATH))

    logger = logging.basicConfig(filename=LOG_PATH, level=logging.INFO)
    logging.info(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    logging.info("Setting:\n N: {}, chunk id: {}, attack_size: {}x{}, stride: {}, model: {}, clip_fn: {}, attack_alg: {}, eps: {}, nb_iter: {}, stepsize: {}".format(FLAGS.N, FLAGS.chunkid, FLAGS.attack_size[0], FLAGS.attack_size[1], FLAGS.stride, FLAGS.model, FLAGS.clip_fn, FLAGS.attack_alg, FLAGS.eps, FLAGS.nb_iter, FLAGS.stepsize))
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
    imagenet_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor()])
    imagenet_val = datasets.ImageNet(FLAGS.data_path, split='val', download=False,
                                         transform=imagenet_transform)

    val_subset_indices = image_partition(42, FLAGS.N)[FLAGS.chunkid]
    val_subset_loader = torch.utils.data.DataLoader(imagenet_val,
                                                    batch_size=1,
                                                    num_workers=4,
                                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(val_subset_indices))

    # load pretrained model
    clip_fn_dic = {"tanh_linear":tanh_linear, 
                   "sigmoid_linear":sigmoid_linear,
                   "binarize":binarize,
                   "None": None}
    clip_fn = clip_fn_dic[FLAGS.clip_fn]

    model_dic = {"resnet18":models.resnet18(pretrained=True),
                "alexnet":models.alexnet(pretrained=True),
                "vgg16":models.vgg16(pretrained=True),
                "resnet34":models.resnet34(pretrained=True),
                "bagnet9":bagnets.pytorch.bagnet9(pretrained=True, avg_pool=False),
                "bagnet17":bagnets.pytorch.bagnet17(pretrained=True, avg_pool=False),
                "bagnet33":bagnets.pytorch.bagnet33(pretrained=True, avg_pool=False),
                "resnet50":models.resnet50(pretrained=True),
                "resnet101":models.resnet101(pretrained=True),
                "resnet152":models.resnet152(pretrained=True),
                "densenet":models.densenet161(pretrained=True),
                "inception":models.inception_v3(pretrained=True)}
    model = model_dic[FLAGS.model]

    attack_alg_dic = {"PGD": PGD,
            "AdamRandomPGD": AdamRandomPGD}
    attack_alg = attack_alg_dic[FLAGS.attack_alg]

    if clip_fn is None:
        wrapper = PatchAttackWrapper
        if FLAGS.model in ["bagnet9", "bagnet17", "bagnet33"]:
            model.avg_pool = True
    else:
        wrapper = ClippedPatchAttackWrapper

    model = model.to(device)
    model.eval()

    #####################################
    # Start attacking
    #####################################
    imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view((3, 1, 1))
    imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view((3, 1, 1))
    print("Start attacking...")
    with torch.no_grad():
        count = 0
        for image, label in val_subset_loader:
            image = image - imagenet_mean
            image = image / imagenet_std
            image = image.to(device)
            logit = model(image)
            if FLAGS.model in ["bagnet9", "bagnet17", "bagnet33"] and model.avg_pool == False: # if apply clipping function
                logit = clip_fn(logit, FLAGS.a, FLAGS.b)
                logit = torch.mean(logit, dim=(1, 2))
            _, topk = torch.topk(logit, k=5, dim=1)
            topk, label = topk.cpu().numpy()[0], label.numpy()[0]
            if label in topk:
                count += 1
        clean_acc = count / FLAGS.N
        print("Accuracy before attack: {}".format(clean_acc))
        logging.info("Accuracy before attack: {}".format(clean_acc))

    tic = time.time()
    succ_prob = foolbox_upper_bound(model, wrapper, val_subset_loader, FLAGS.attack_size, 
                                    clip_fn=clip_fn, a=FLAGS.a, b=FLAGS.b, stride=FLAGS.stride, 
                                    max_iter=FLAGS.nb_iter, attack_alg=attack_alg, eps=FLAGS.eps, stepsize=FLAGS.stepsize,
                                    return_early=FLAGS.return_early, output_root=OUTPUT_PATH)
    tac = time.time()
    print("Success probability: {}, Time: {:.3f}s or {:.3f}hr(s)".format(succ_prob, tac - tic, (tac-tic)/3600))
    logging.info("Success probability: {}, Time: {:.3f}s or {:.3f}hr(s)".format(succ_prob, tac - tic, (tac-tic)/3600))

if __name__ == "__main__":
    app.run(main)

