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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
gpu_count = torch.cuda.device_count()

output_root = "/mnt/data/results/spsa_results/spsa_vs_pgd/cross_entropy_loss_stepsize0.25_iter500"
#output_root = "."
def get_subimg(image, loc, size):
    x1, y1 = loc
    x2, y2 = x1 + size[0], y1 + size[1]
    return image[:, x1:x2, y1:y2]

class StickerSPSA:
    def __init__(self, model, subimg, label, sticker_size=(20, 20), 
                 delta = 0.01, num_samples=128, step_size=0.01, epsilon=1e-10):
        self.model = model
        self.clean_subimg = subimg.clone()
        #self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).cuda()
        #self.std = torch.tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1)).cuda()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).to(subimg.get_device())
        self.std = torch.tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1)).to(subimg.get_device())

        self.clean_undo_subimg = self.undo_imagenet_preprocess_pytorch(subimg)
        self.adv_subimg = subimg.clone()
        self.label = label
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sticker_size = sticker_size
        self.adv_pertub = None
        self.num_samples = num_samples
        self.delta = delta
        self.epsilon = epsilon
        self.label_cfd = 0
        self.best_other_cfd = 0
        self.worst_other_cfd = 0
        self.adam_optimizer = AdamOptimizer(shape=(1, 3)+sticker_size, learning_rate=step_size)

    def undo_imagenet_preprocess_pytorch(self, subimg):
        return (subimg*self.std) + self.mean
    
    def run(self):
        # Sample perturbation from Bernoulli +/- 1 distribution
        _samples = torch.sign(torch.empty((self.num_samples//2, 3) + self.sticker_size, dtype=self.adv_subimg.dtype).uniform_(-1, 1))
        _samples = _samples.cuda()
        delta_x = self.delta * _samples
        delta_x = torch.cat([delta_x, -delta_x], dim=0) 
        _sampled_perturb = self.adv_subimg + delta_x

        with torch.no_grad():
            logits = self.model(_sampled_perturb)
        cfd = F.softmax(logits, dim=1)

            # calculate the margin logit loss
        self.label_cfd = cfd[:, true_label].reshape((-1, )).clone()
        self.label_cfd = torch.mean(self.label_cfd).item()
        cfd[:, true_label] = float('-inf')
        #print(f'{(label_logit.min().item(), label_logit.max().item())}')
        value, indices = torch.topk(cfd, k=5, dim=1)
        self.best_other_cfd = torch.mean(value[:, 0]).item()
        self.worst_other_cfd = torch.mean(value[:, -1]).item()

        """ margin-based loss
        # calculate the margin logit loss
        self.label_logit = logits[:, self.label].reshape((-1, )).clone()
        #print(f'{(label_logit.min().item(), label_logit.max().item())}')
        value, indices = torch.topk(logits, k=5, dim=1)
        logits[:, self.label] = float('-inf')
        #print(f'{(label_logit.min().item(), label_logit.max().item())}')
        self.best_other_logit, _ = torch.max(logits, dim=1)
        values, _ = torch.topk(logits, 5, dim=1)
        self.worst_other_logit = values[-1]
        ml_loss = self.label_logit - self.best_other_logit
        """

        """cross-entropy"""
        label = torch.full((self.num_samples,), self.label, dtype=torch.long).cuda()
        loss = self.loss_fn(logits, label)
        # estimate the gradient
        all_grad = loss.reshape((-1, 1, 1, 1)) / (delta_x + self.epsilon)
        est_grad = torch.mean(all_grad, dim=0)
        
        adam_grad = self.adam_optimizer(est_grad[None])
        
        self.adv_subimg += adam_grad

        # clip the pixel to the valid range
        adv_undo_subimg = self.undo_imagenet_preprocess_pytorch(self.adv_subimg)
        
        adv_undo_subimg = torch.clamp(adv_undo_subimg, 0, 1)

        self.adv_subimg = (adv_undo_subimg - self.mean) / self.std

# load pretrained model
model = pytorchnet.bagnet33(pretrained=True, avg_pool=False).to(device)
model.eval()
print()

# Prepare dataloader
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([[0.229, 0.224, 0.225]]).reshape((3, 1, 1))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
imagenet_transform = transforms.Compose([transforms.ToTensor(), 
                                         normalize])
# Dataset for spsa
folder = datasets.ImageFolder("./spsa_validation_images/vulnerable", transform=imagenet_transform)

# Dataset for foolbox: no normalization
folder_foolbox = datasets.ImageFolder("./spsa_validation_images/vulnerable", transform=transforms.ToTensor())

id2id = {value:int(key) for key, value in folder.class_to_idx.items()}

idx2loc = {
        0: (140, 40),
        1:(60, 120),
        2:(40, 100),
        3:(20, 20),
        4:(100, 80),
        6:(140, 80),
        7:(160, 60),
        8:(40, 120),
        10:(60, 100),
        11:(40, 160),
        12:(80, 80)}
#idx2loc = {0: (140, 40)}
imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view((3, 1, 1))
imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view((3, 1, 1))

for img_idx in [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12]:
#for img_idx in [11]:
    val_subset_indices = [img_idx]

    val_loader = torch.utils.data.DataLoader(folder,
                                            batch_size=1,
                                            num_workers=4,
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(val_subset_indices))

    val_loader_foolbox = torch.utils.data.DataLoader(folder_foolbox,
                                            batch_size=1,
                                            num_workers=4,
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(val_subset_indices))


# Section 1. SPSA
    # Get clean prediction and logits
    for image, label in val_loader:
        true_label = id2id[label.item()]
        image = image.to(device)
        with torch.no_grad():
            logits = model(image)
        logits = tanh_linear(logits, a=0.05, b=-1)
        logits = torch.mean(logits, dim=(1, 2))
        _, topk = torch.topk(logits, k=5, dim=1)
        topk = topk.cpu().numpy()[0]
        cfd = F.softmax(logits, dim=1)

            # calculate the margin logit loss
        init_label_cfd = cfd[:, true_label].reshape((-1, )).clone().item()
            #print(f'{(label_logit.min().item(), label_logit.max().item())}')
        cfd[:, true_label] = float('-inf')
            #print(f'{(label_logit.min().item(), label_logit.max().item())}')
        value, indices = torch.topk(cfd, k=5, dim=1)
        value = value[0]
        init_best_other_cfd = value[0].item()
        init_worst_other_cfd = value[-1].item()

            #print(f'label_logit: {label_logit}, best_other: {best_other_logit}, worst_other: {worst_other_logit}')
    print(f"label: {true_label}, topk: {topk}")

    # Init logit list for spsa and pgd
    label_cfd_list = [init_label_cfd]
    best_cfd_list = [init_best_other_cfd]
    worst_cfd_list = [init_worst_other_cfd]
    pgd_label_cfd_list_head = [init_label_cfd]
    pgd_best_cfd_list_head = [init_best_other_cfd]
    pgd_worst_cfd_list_head = [init_worst_other_cfd]

    # SPSA configuration
    wrapper=DynamicClippedPatchAttackWrapper
    sticker_size = (20, 20)
    print(f"attack at {idx2loc[img_idx]}")
    step_size = 0.25
    clip_fn=tanh_linear
    a=0.05
    b=-1
    number_iter = 500
    x, y = idx2loc[img_idx]

    subimg = get_subimgs(image, (x, y), sticker_size)

    wrapped_model = wrapper(model, image.clone(), sticker_size, (x, y), clip_fn, a, b)

    if gpu_count > 1:
        wrapped_model = nn.DataParallel(wrapped_model)

    spsa_attack = StickerSPSA(wrapped_model, subimg, true_label, sticker_size=sticker_size, step_size=step_size)
    
    # SPSA attack
    for j in range(number_iter):
        print(f"spsa: iter{j}")
        spsa_attack.run()
        label_cfd_list.append(spsa_attack.label_cfd)
        best_cfd_list.append(spsa_attack.best_other_cfd)
        worst_cfd_list.append(spsa_attack.worst_other_cfd)

    spsa_list_collector = np.array([label_cfd_list, best_cfd_list, worst_cfd_list])

# Section 2. PGD
    wrapper = ClippedPatchAttackWrapper
    attack_alg = AdamRandomPGD
    attack_size = (20, 20)
    max_iter = number_iter
    x, y = idx2loc[img_idx]
    mean=np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)), 
    std=np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    for image, label in val_loader_foolbox:
        label = id2id[label.item()]
        prep_image = (image - imagenet_mean) / imagenet_std
        prep_image = prep_image.cuda()
        image = image[0].numpy()

        # PGD configuration
        wrapped_model = wrapper(model, prep_image, attack_size, (x, y), clip_fn, a, b)
        wrapped_model.eval()
        print('image {}, current location: {}'.format(img_idx, (x, y)))
        fmodel = foolbox.models.PyTorchModel(wrapped_model, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
        criterion = TopKMisclassification(5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            attack = attack_alg(fmodel, criterion=criterion, distance=foolbox.distances.Linfinity)

        subimg = get_subimg(image, (x, y), attack_size)
        adversarial = attack(subimg, label, iterations = max_iter, epsilon=1., stepsize=0.01, random_start=True, return_early=False, binary_search=False)
        pgd_label_cfd_list = pgd_label_cfd_list_head + attack.label_cfd_list
        pgd_best_cfd_list = pgd_best_cfd_list_head + attack.best_cfd_list
        pgd_worst_cfd_list = pgd_worst_cfd_list_head + attack.worst_cfd_list

        pgd_list_collector = np.array([pgd_label_cfd_list, pgd_best_cfd_list, pgd_worst_cfd_list])
    
    # pickle dump
    spsa_pgd_collector = np.array([spsa_list_collector, pgd_list_collector])
    file = open('{}/vulnerable{}.npy'.format(output_root, img_idx), 'wb')
    pickle.dump(spsa_pgd_collector, file)
    file.close()
