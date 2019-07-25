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
flags.DEFINE_float('eps', 5., 'range of perturbation')
flags.DEFINE_integer('nb_iter', 40, 'number of iterations for PGD')
flags.DEFINE_boolean('rand_init', True, 'random initial point in attack')
flags.DEFINE_float('stepsize', 0.5, 'stepsize of PGD')
flags.DEFINE_integer('metabatch_size', 10, 'metabatch size')
flags.DEFINE_string('data_path', '/mnt/data/imagenet', 'directory where data are stored')
flags.DEFINE_string('output_root', '/mnt/data/results/', 'directory for storing results')

def main(argv):
    """
    FLAGS.output_root/
        [NAME]/
            [NAME].mtb
            [NAME].log
            dataset/
    """
    NAME = 'scheduled-rang_init_{}-{}-{}-{}-{}x{}-{}-{}-{}-{}'.format(FLAGS.rand_init, FLAGS.model, FLAGS.N, FLAGS.seed, FLAGS.attack_size[0], FLAGS.attack_size[1], FLAGS.stride, FLAGS.eps, FLAGS.nb_iter, FLAGS.stepsize)
    OUTPUT_PATH = os.path.join(FLAGS.output_root, NAME)

    if not os.path.exists(OUTPUT_PATH):
                os.mkdir(OUTPUT_PATH)
    if not os.path.exists(os.path.join(OUTPUT_PATH, "dataset")):
                os.mkdir(os.path.join(OUTPUT_PATH, "dataset"))

    LOG_PATH = os.path.join(OUTPUT_PATH, NAME+'.log')
    print("log to {}".format(LOG_PATH))

    logger = logging.basicConfig(filename=LOG_PATH, level=logging.INFO)
    logging.info(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    logging.info("Setting:\n Scheduled clipping parameters, random init: {},  model: {}, N: {}, seed: {}, attack_size: {}x{}, stride: {}, eps: {}, nb_iter: {}, stepsize: {}".format(FLAGS.rand_init, FLAGS.model, FLAGS.N, FLAGS.seed, FLAGS.attack_size[0], FLAGS.attack_size[1], FLAGS.stride, FLAGS.eps, FLAGS.nb_iter, FLAGS.stepsize))
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
    clip_fn, a, b = binarize, 25, None
    model_dic = {"bagnet9":bagnets.pytorch.bagnet9(pretrained=True, avg_pool=False),
                 "bagnet17":bagnets.pytorch.bagnet17(pretrained=True, avg_pool=False),
                 "bagnet33":bagnets.pytorch.bagnet33(pretrained=True, avg_pool=False)}

    model = model_dic[FLAGS.model]
    model = model.to(device)
    model.eval()

    #####################################
    # Start attacking
    #####################################
    max_iter = ((224 - FLAGS.attack_size[0]) // FLAGS.stride + 1) * ((224 - FLAGS.attack_size[1]) // FLAGS.stride + 1)
    print('max iter: {}'.format(max_iter))

    with torch.no_grad():
        count = 0
        for image, label in val_subset_loader:
            image = image.to(device)
            logit = model(image)
            if clip_fn:
                logit = clip_fn(logit, a, b)
            logit = torch.mean(logit, dim=(1, 2))
            _, topk = torch.topk(logit, k=5, dim=1)
            topk, label = topk.cpu().numpy()[0], label.numpy()[0]
            if label in topk:
                count += 1
        clean_acc = count / FLAGS.N
        print("Accuracy before attack: {}".format(clean_acc))
        logging.info("Accuracy before attack: {}".format(clean_acc))

    # instanciate metabatch
    data_iter = iter(val_subset_loader)
    metabatch = MetaBatch(data_iter, FLAGS.metabatch_size, max_iter)
    metabatch.clean_acc = clean_acc

    tic = time.time()
    succ_prob = scheduled_upper_bound(model, metabatch, 
                                      attack_size=FLAGS.attack_size, 
                                      eps=FLAGS.eps, nb_iter=FLAGS.nb_iter, stepsize=FLAGS.stepsize, 
                                      stride=FLAGS.stride, k=5, rand_init=FLAGS.rand_init)
    tac = time.time()
    print("Success probability: {}, Time: {:.3f}s or {:.3f}hr(s)".format(succ_prob, tac - tic, (tac-tic)/3600))
    logging.info("Success probability: {}, Time: {:.3f}s or {:.3f}hr(s)".format(succ_prob, tac - tic, (tac-tic)/3600))

    for key, value, in metabatch.fail_list.items(): # (image, label, attack location)
        print("image {}:".format(key))
        logging.info("image {}:".format(key))
        image, label, loc = value
        x1, y1 = loc
        x2, y2 = x1 + 20, y1 + 20
        adv = metabatch.adv[key].clone()
        image = image[None].clone()
        image[:, :, x1:x2, y1:y2] = adv[None]
        with torch.no_grad():
            logits = model(image.to(device))
        if clip_fn:
            logits = clip_fn(logits, a, b)
        logits = torch.mean(logits, dim=(1, 2))
        _, topk = torch.topk(logits, k=5, dim=1)
        image = undo_imagenet_preprocess(image[0]).cpu().numpy() # undo preprocessing
        image = convert2channel_last(image)
    #     print(image.min(), image.max())
        class_path = os.path.join(OUTPUT_PATH, 'dataset', '{}'.format(label))
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        plt.imsave(os.path.join(class_path, '{}.png'.format(key)), image)
        topk = topk[0].cpu().numpy()
        print(label, topk)
        logging.info((label, topk))
    
    metabatch.waitlist = None
    with open(os.path.join(OUTPUT_PATH, NAME+'.mtb'), 'wb') as file:
        pickle.dump(metabatch, file)

if __name__ == "__main__":
    app.run(main)
