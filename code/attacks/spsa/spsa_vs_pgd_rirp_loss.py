import os
import warnings
from utils import *
from clipping import *
import pytorchnet
import torch
import numpy as np
import time
import pickle
import foolbox
import torch.nn.functional as F
from foolbox_alpha.attacks import AdamRandomPGD
from foolbox.criteria import TopKMisclassification
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from absl import app, flags
import os

flags.DEFINE_integer('N', 500, 'number of images')
flags.DEFINE_integer('chunkid', 0, 'partition chunk id')
flags.DEFINE_integer('nb_iter', 500, 'number of iterations for PGD')
flags.DEFINE_float('stepsize', 0.1, 'stepsize of spsa')
flags.DEFINE_string('data_path', '/mnt/data/imagenet', 'data directory')
flags.DEFINE_string('output_root', "/mnt/data/results/spsa_results/spsa_vs_pgd/", "output root")
FLAGS = flags.FLAGS

def main(argv):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    gpu_count = torch.cuda.device_count()
    
    NAME = '{}-{}-{}-{}'.format(FLAGS.N, FLAGS.chunkid, FLAGS.nb_iter, FLAGS.stepsize)

    OUTPUT_PATH = os.path.join(FLAGS.output_root, NAME)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    #output_root = "."
    # load pretrained model
    model = pytorchnet.bagnet33(pretrained=True, avg_pool=False).to(device)
    model.eval()
    
    # Prepare dataloader
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([[0.229, 0.224, 0.225]]).reshape((3, 1, 1))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    imagenet_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             normalize])

    imagenet_transform_foolbox = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor()])

    # Dataset for spsa
    folder = datasets.ImageNet(FLAGS.data_path, split='val', transform=imagenet_transform)
    
    # Dataset for foolbox: no normalization
    folder_foolbox = datasets.ImageNet(FLAGS.data_path, split='val', transform=imagenet_transform_foolbox)
    
    imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view((3, 1, 1))
    imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view((3, 1, 1))
    img_idx_list = image_partition(42, FLAGS.N)[FLAGS.chunkid]
    for img_idx in img_idx_list:
        print(f'image {img_idx}')
        val_subset_indices = [img_idx]
    
        val_loader = torch.utils.data.DataLoader(folder,
                                                batch_size=1,
                                                num_workers=4,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(val_subset_indices))
    
        val_loader_foolbox = torch.utils.data.DataLoader(folder_foolbox,
                                                batch_size=1,
                                                num_workers=4,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(val_subset_indices))
        px, py = np.random.choice([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200], 2, replace=True)
        print(f"attack at {(px, py)}")
    
    # Section 1. SPSA
        # Get clean prediction and logits
        for image, label in val_loader:
            image = image.to(device)
            label = label[0].item()
            with torch.no_grad():
                logits = model(image)
            #cfd = F.softmax(torch.mean(logits, dim=(1, 2)))
            #value_cfd, _ = torch.topk(cfd, k=5, dim=1)
            #print(f'top cfd before clipping: {value_cfd[0][0]}')
            logits = tanh_linear(logits, a=0.05, b=-1)
            logits = torch.mean(logits, dim=(1, 2))
            #cfd = F.softmax(logits)
            #value_cfd, _ = torch.topk(cfd, k=5, dim=1)
            #print(f'top cfd after clipping: {value_cfd[0][0]}')
            _, topk = torch.topk(logits, k=5, dim=1)
            topk = topk.cpu().numpy()[0]
    
                # calculate the margin logit loss
            init_label_logit = logits[:, label].reshape((-1, )).clone().item()
                #print(f'{(label_logit.min().item(), label_logit.max().item())}')
            logits[:, label] = float('-inf')
                #print(f'{(label_logit.min().item(), label_logit.max().item())}')
            value, indices = torch.topk(logits, k=5, dim=1)
            value = value[0]
            #init_best_other_logits = value[0].item()
            init_worst_other_logits = value[-1].item()
            init_loss = init_label_logit - init_worst_other_logits
    
                #print(f'label_logit: {label_logit}, best_other: {best_other_logit}, worst_other: {worst_other_logit}')
        print(f"label: {label}, topk: {topk}")
        if label not in topk:
            print("Not running the attack because the original input is already misclassified.")
            continue
    
        # Init logit list for spsa and pgd
        spsa_loss_list = [init_loss]
        pgd_loss_head = [init_loss]
    
        # SPSA configuration
        wrapper=DynamicClippedPatchAttackWrapper
        sticker_size = (20, 20)
        step_size = 0.1
        clip_fn=tanh_linear
        a=0.05
        b=-1
    
        subimg = get_subimgs(image, (px, py), sticker_size)
    
        wrapped_model = wrapper(model, image.clone(), sticker_size, (px, py), clip_fn, a, b)
    
        if gpu_count > 1:
            wrapped_model = nn.DataParallel(wrapped_model)
    
        spsa_attack = StickerSPSAEval(wrapped_model, subimg, label, sticker_size=sticker_size, step_size=step_size, random_init=True)
        
        # SPSA attack
        for j in range(FLAGS.nb_iter):
            print(f"spsa: iter{j}")
            spsa_attack.run()
            spsa_loss_list.append(spsa_attack.current_loss)
    
    # Section 2. PGD
        
        attack_size = (20, 20)
        max_iter = FLAGS.nb_iter
        mean=np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)), 
        std=np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    
        for image, label in val_loader_foolbox:
            label = label[0].item()
            prep_image = (image - imagenet_mean) / imagenet_std
            prep_image = prep_image.cuda()
            image = image[0].numpy()
    
            # PGD configuration
            wrapped_model = ClippedPatchAttackWrapper(model, prep_image, attack_size, (px, py), clip_fn, a, b)
            wrapped_model.eval()
            print('image {}, current location: {}'.format(img_idx, (px, py)))
            fmodel = foolbox.models.PyTorchModel(wrapped_model, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
            criterion = TopKMisclassification(5)
            attack = AdamRandomPGD(fmodel, criterion=criterion, distance=foolbox.distances.Linfinity)
            attack.loss_list = []
    
            subimg = get_subimg(image, (px, py), attack_size)
            adversarial = attack(subimg, label, iterations = max_iter, epsilon=1., stepsize=0.01, random_start=True, return_early=False, binary_search=False)
            pgd_loss_list = pgd_loss_head + attack.loss_list
        
        # pickle dump
        spsa_pgd_loss = np.array([spsa_loss_list, pgd_loss_list, f'{img_idx}'])

        file = open(os.path.join(OUTPUT_PATH, f"image{img_idx}.npy"), 'wb')
        pickle.dump(spsa_pgd_loss, file)
        file.close()

if __name__ == "__main__":
    app.run(main)
