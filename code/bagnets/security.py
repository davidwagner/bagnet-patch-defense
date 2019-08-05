import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from bagnets.utils import *
from bagnets.clipping import *
import foolbox
from foolbox.criteria import TopKMisclassification
from foolbox.distances import Distance
from foolbox.adversarial import Adversarial
import logging
from advertorch.attacks import LinfPGDAttack

#########################################
# Security Lower Bound
#########################################

def is_invariant(class_lower, class_upper, label, target=None, k=5):
    """Given the upper bound and lower bound of class logits of one image, determine
        whether the classifier could possibly misclassify this image in top-K prediction.
    Input:
    - class_upper (pytorch tensor): shape = (1000,) in CPU
    - class_lower (pytorch tensor): shape = (1000,) in CPU
    - label (int): image's label
    - targets (int): targeted class. None means untargeted attack
    - k (int): top-K prediction
    Output (output): whether the classifier is robust against the attack on this image
    """
    label_lower = class_lower[label].item()
    top6_values, top6_classes = torch.topk(class_upper, k=k+1)
    top6 = top6_values.numpy()
    if target is None:
        return label_lower >= top6_values[-1].item()
    else:
        return label_lower >= top6_values[-1].item() or target not in list(top6[:k])


def get_affected_patches(attack_size, location, ps=33):
    """
    """
    assert ps in [9, 17, 33]
    ps2bound = {9:27-1, 17:26-1, 33:24-1}
    boundary = ps2bound[ps] # max index in logit space
    x1, y1 = location # top-left coordinate of the sticker
    dx, dy = attack_size
    x2, y2 = x1 + dx - 1, y1 + dy - 1 # bottom-right coordinate of the sticker
    
    p1, q1 = max(0, x1 - ps + 1), max(0, y1 - ps + 1)
    p1, q1 = (p1 // 8) * 8, (q1 // 8) * 8 # align to the receptive field's top-left coordinate in image space
    p2, q2 = p1 + ps - 1, q1 + ps - 1
    while not x1 <= p2 and x2 >= p1:
        p1 += ps - 1
        p2 += ps - 1
    while not y1 <= q2 and y2 >= q1:
        q2 += ps - 1
        q1 += ps - 1
    tx, ty = p1 // 8, q1 // 8
    
    # repeat the procedure above for bottom right coordinate
    p1, q1 = min(8*boundary, x2 + ps - 1), min(8*boundary, y2 + ps - 1)
    p1, q1 = (p1 // 8) * 8, (q1 // 8) * 8
    p2, q2 = p1 + ps - 1, q1 + ps - 1
    while not p1 <= x2 and p2 >= x1:
        p1 -= ps + 1
        p2 -= ps + 1
    while not q1 >= y2 and q2 <= y1:
        q2 -= ps + 1
        q1 -= ps + 1
        
    bx, by = p1 // 8, q1 // 8
    return (tx, ty, bx, by)

def bound_patch_attack(patch_logits, affected_patch, bound=(-1, 1)):
    """Bound the effect of patch attack given affected patches
    Input:
    - patch_logits (pytorch tensor): patch logits returned by BagNet33
    - affected_patch (tuple): (x1, y1, x2, y2), where (x1, y1) is the top-left coordinate of top-left patch,
                                and (x2, y2) is the bottom-right coordinate of bottom-right patch.
    - bound (tuple): (lower bound, upper bound)
    Output:
    - class_logits_upper (pytorch tensor): upper bound of class logits, shape = (1000,)
    - class_logits_lower (pytorch tensor): lower bound of class logits, shape = (1000,)
    """
    x1, y1, x2, y2 = affected_patch
    upper, lower = patch_logits.clone(), patch_logits.clone()
    upper[x1:x2+1, y1:y2+1, :] = bound[1]
    lower[x1:x2+1, y1:y2+1, :] = bound[0]
    class_logits_upper = torch.mean(upper, dim=(0, 1))
    class_logits_lower = torch.mean(lower, dim=(0, 1))
    return (class_logits_lower, class_logits_upper)

def get_position_buffer(h, w, attack_size, stride, ps=33):
    """Buffer of all possible affected patch positions
    """
    buffer = set()
    last_affected_patches = (0, 0, 0, 0)
    for i in range(0, h - attack_size[0]+1, stride):
        for j in range(0, w - attack_size[1]+1, stride):    
            affected_patches = get_affected_patches(attack_size, (i, j), ps=ps)
            buffer.add(affected_patches)
    return buffer

def pick_targeted_classes(logits, k=5, case='avg'):
    """
    Input:
    - logits (pytorch tensor): clipped logits (clipped patch logits after avg). Shape (N, 1000)
    - k (int): top-k prediction
    - case (string): ['best', 'avg', 'worst']
            'best': the class ranked at (k+1)-th
            'avg': randomly sample one class outside of the top-k predictions
            'worst': the class ranked the last
    """
    assert case in ['best', 'avg', 'worst'], 'case must be "best", "avg", or "worst"'
    if case == 'best':
        _, topk = torch.topk(logits, k=k+1, dim=1) # shape (N, k+1) 
        return topk[:, -1]
    if case == 'avg':
        _, topk = torch.topk(logits, k=1000, dim=1) # shape (N, 1000) 
        indices = torch.randint(low=5, high=999, size=(1,))
        return torch.index_select(topk, 1, indices.cuda()).reshape((-1,))
    else: # if case == 'worst'
        _, topk = torch.topk(logits, k=1000, dim=1) # shape (N, 1000) 
        return topk[:, -1]

# pytorch data loader version
def get_security_lower_bound(bagnet, data_loader, attack_size, clip, stride=1, bound=(-1, 1), **kwargs):
    """ Take in a pytorch data loader and the size of adversarial sticker, loop over each of  
            the possible position of the sticker to count how many images in the batch 
            can still be correctly classified if the sticker is at that position.
    Input:
    - bagnet (pytorch model): BagNet-33 without avgerage pooling
    - data_loader (pythorch DataLoader): dataloader in CPU
    - attack_size (tuple): (h, w), height and width of the adversarial sticker
    - clip (function): clipping function
    - stride (int): stride of stickers
    - bound (tuple): tanh_linear: (-1, 1), sigmoid_linear: (-1, 1)
    - **kwargs: a, b, clipping parameters
    Output:
    - succ_prob (float): fraction of images whose prediction are invariant in the worst case
    """
    positions = get_position_buffer(224, 224, attack_size, stride)
    positions = sorted(positions)
    num_invariant, total_images = 0, 0
    for images, targets in data_loader:
        images, targets = images.cuda(), targets.numpy()
        with torch.no_grad():
            logits = bagnet(images)
        logits = clip_fn(logits, a, b).cpu()
        avg_logits = torch.mean(logits, dim=(1, 2))
        if targeted:
            targets = pick_targeted_classes(avg_logits, k=k, case=case)
            targets = targetes.numpy()
        else:
            targets = None
        n, c, h, w = images.shape
        total_images += n
        for k in range(n):
            img_patch_logits = logits[k] # shape (24, 24, 1000)
            label = labels[k]
            if targeted: target = targets[k]
            flag = True
            for p in positions:
                class_lower, class_upper = bound_patch_attack(img_patch_logits, p, bound=bound)
                if not is_invariant(class_lower, class_upper, label, target):
                    flag = False
                    break
            if flag:
                num_invariant += 1
    succ_prob = num_invariant / total_images
    return succ_prob

#########################################
# Security Upper Bound (Foolbox)
#########################################

def foolbox_upper_bound(model, wrapper, data_loader, attack_size, 
                        clip_fn, a=None, b=None, stride=1, k=5, 
                        max_iter=40, eps=1, stepsize=0.5, return_early=True, random_start=True,
                        mean=np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)), 
                        std=np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)), 
                        output_root = './foolbox_results'):
    """ Take in a pytorch data loader, use the ATTACK_ALG to
        calculate the security upper bound.
    Input:
    - model (pytorch model): model wrapped for sticker attack 
    - data_loader (pytorch DataLoader): batchsize=1, pytorch dataloader in CPU, without normalization, pixel ~ [0, 1]
    - attack_alg (foolbox.attack): attack algorithm implemented in foolbox
    - stride (int): stride of how stickers move
    - k (int): top-k misclassification criteria
    - mean, std (numpy array): mean and standard devication of dataset
    - output_path (str): directory for saving resulting plots
    Output:
    - succ_prob (float): fraction of images whose prediction are not affected by the attack
    """
    affected_imgs, total_images = 0, 0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for images, labels in data_loader:
        prep_images = normalize(images[0])[None]
        image, label = images[0].numpy(), labels[0].item()
        c, h, w = image.shape
        total_images += 1
        flag = True
        for x in range(0, h - attack_size[0] + 1, stride):
            for y in range(0, w - attack_size[1] + 1, stride):
                wrapped_model = wrapper(model, prep_images, attack_size, (x, y), clip_fn, a, b)
                wrapped_model.eval()
                print('image {}, current location: {}'.format(total_images, (x, y)))
                fmodel = foolbox.models.PyTorchModel(wrapped_model, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
                attack = PGD(fmodel, criterion=TopKMisclassification(k), distance=foolbox.distances.Linfinity)
                subimg = get_subimg(image, (x, y), attack_size)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    adversarial = attack(subimg, label, iterations = max_iter, epsilon=eps, stepsize=stepsize, random_start=random_start, return_early=return_early, binary_search=False)
                    
                if adversarial is not None:
                    msg = 'Image {}, attack successfully, location {}'.format(total_images, (x, y))
                    print(msg)
                    logging.info(msg)
                    plt_name = '{}.png'.format(total_images)
                    save_path = os.path.join(output_root, str(label))
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    attacked_img = apply_sticker(adversarial, image, (x, y), attack_size).transpose([1, 2, 0])
                    plt.imsave(os.path.join(save_path, plt_name), attacked_img)
                    affected_imgs += 1
                    flag = False
                    break
                else:
                    print('Image {}: fail to find an adversarial sticker at {}'.format(total_images, (x, y)))
            if not flag:
                break
    num_invariant = total_images - affected_imgs
    succ_prob = num_invariant / total_images
    return succ_prob

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

class PatchAttackWrapper(nn.Module):
    
    def __init__(self, model, img, size, loc, clip_fn=None, a=None, b=None):
        super(PatchAttackWrapper, self).__init__()
        self.batch_size = img.shape[0]
        self.model = model
        self.x1, self.y1 = loc
        self.x2, self.y2 = self.x1 + size[0], self.y1 + size[1]
        mask = torch.ones((1, 3, 224, 224))
        mask[:, :, self.x1:self.x2, self.y1:self.y2] = 0
        self.img = (mask*img).cuda()
    
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


######################################################
# Advertorch Metabatch Upper Bound
######################################################
class AdverTorchWrapper(nn.Module):
    def __init__(self, model, img, size, loc, clip, a, b):
        super(AdverTorchWrapper, self).__init__()
        self.batch_size = img.shape[0]
        self.model = model
        self.clip = clip
        self.a = a
        self.b = b
        self.x1, self.y1 = loc
        self.x2, self.y2 = self.x1 + size[0], self.y1 + size[1]
        self.img = img.contiguous().cuda(async=False)

    def forward(self, subimg):
        img = self.img.clone()
        img[:, :, self.x1:self.x2, self.y1:self.y2] = subimg
        clipped_patch_logits = self._clip_logits(img, a = self.a, b = self.b)
        clipped_class_logits = torch.mean(clipped_patch_logits, dim=(1, 2))
        return clipped_class_logits

    def _clip_logits(self, attacked_img, **kwargs):
        patch_logits = self.model(attacked_img)
        if self.clip:
            return self.clip(patch_logits, **kwargs)
        return patch_logits

class ScheduledParamAdverTorchWrapper(AdverTorchWrapper):
    def __init__(self, model, img, size, loc, nb_iter, clip=sigmoid_linear, a=1, b=-25, init_factor=0.1, max_factor=100.5):
        super().__init__(model, img, size, loc, clip, a, b)
        param_stepsize = (max_factor-init_factor) / nb_iter
        self.factor_list = np.arange(init_factor, max_factor, param_stepsize)
        self.curr_step = 0
        
    def forward(self, subimg):
        img = self.img.clone()
        img[:, :, self.x1:self.x2, self.y1:self.y2] = subimg
        curr_factor = self.factor_list[self.curr_step]
        clipped_patch_logits = self._clip_logits(img, a = curr_factor*self.a, b = curr_factor*self.b)
        self.curr_step += 1
        clipped_class_logits = torch.mean(clipped_patch_logits, dim=(1, 2))
        return clipped_class_logits

class UndefendedAdverTorchWrapper(nn.Module):
    def __init__(self, model, img, size, loc):
        super(UndefendedAdverTorchWrapper, self).__init__()
        self.batch_size = img.shape[0]
        self.model = model
        self.x1, self.y1 = loc
        self.x2, self.y2 = self.x1 + size[0], self.y1 + size[1]
        self.img = img.contiguous().cuda(async=False)

    def forward(self, subimg):
        img = self.img.clone()
        img[:, :, self.x1:self.x2, self.y1:self.y2] = subimg
        logits = self.model(img)
        return logits

class MetaBatch:
    def __init__(self, data_iter, size, max_iter):
        self.clean_acc = 1
        self.global_step = 0
        self.waitlist = data_iter # images waiting for attack, pytorch data iterator, batch size=1, in CPU
        self.exhaust = False
        self.size = size # size of the metabatch
        self.images = torch.zeros((size, 3, 224, 224)) # images under attack, in CPU
        self.record = np.zeros((self.size, 2), dtype=np.int64) # [(image ID 0, # duration 0), (image ID 1, # duration 1), ...]
        self.labels = torch.zeros((self.size, ), dtype=torch.int64)
        assert self.size <= len(data_iter), "Size of MetaBatch shouldn't larger than data iterator's size"
        self.orig_list = {} # {id: (image, label)}
        for idx in range(self.size):
            image, label = next(self.waitlist)
            self.orig_list[self.global_step] = (image[0].clone(), label.item())
            self.record[idx, 0] = self.global_step
            self.global_step += 1
            self.images[idx] = image[0].clone()
            self.labels[idx] = label.item()
        self.fail_list = {} # {id: (image, label, loc), ...}
        self.adv = {} # {id: adv_sticker, ...}
        self.targeted = {} # {id: targeted_class_index, ...}
        self.succ_list = [] # [id, ...]
        self.max_iter = max_iter
        self.location = None
    
    def update(self, indices, adv, targeted_labels=None):
        """ - Replace images that generate adversatial stickers successfully with new images from 
                the wait list.
        Input:
        - indices (python list): 
        - adv (pytorch tensor):
        - targeted_labels (python list)
        """
        # Replace failure cases with new images
        for idx in indices:
            if self.record[idx, 0] not in self.adv.keys(): # don't update images that are already failed.
                self.fail_list[self.record[idx, 0]] = (self.images[idx].clone(), self.labels[idx].item(), self.location) # (image, label, attack location)
                self.adv[self.record[idx, 0]] = adv[idx].clone()
                if targeted_labels is not None:
                    self.targeted[self.record[idx, 0]] = targeted_labels[idx]
                try:
                    image, label = next(self.waitlist)
                    self.orig_list[self.global_step] = (image[0].clone(), label.item())
                    self.images[idx] = image[0].clone()
                    self.record[idx, 0] = self.global_step # update image ID
                    self.record[idx, -1] = 0 # reset counter
                    self.labels[idx] = label.item()
                    self.global_step += 1
                except StopIteration:
                    self.exhaust = True
                    continue
            
        # Replace successful cases with new images
        for k in [i for i in range(self.size) if i not in indices]:
            if self.record[k, 0] not in self.adv.keys() and self.record[k, -1] < self.max_iter+1: self.record[k, -1] += 1 
            if self.record[k, -1] == self.max_iter:
                self.succ_list.append(self.record[k, 0])
                try:
                    image, label = next(self.waitlist)
                    self.orig_list[self.global_step] = (image[0].clone(), label.item())
                    self.images[k] = image[0].clone()
                    self.labels[k] = label.item()
                    self.record[k, 0] = self.global_step # update image ID
                    self.record[k, -1] = 0 # reset counter
                    self.global_step += 1
                except StopIteration:
                    self.exhaust = True
                    continue
        print('metabatch record:\n {}'.format(self.record))
        logging.info('metabatch record:\n {}'.format(self.record))
                    
    def get_succ_prob(self):
        return 1 - len(self.fail_list) / len(self.orig_list)

    def display_stickers(self, n=20):
        for i, (label, sticker) in enumerate(self.adv.items()):
            sticker = undo_imagenet_preprocess(sticker)
            plt.imshow(convert2channel_last(sticker.numpy()))
            plt.title('label {}'.format(label))
            plt.show()
            if i == n-1:
                break

    def display_failures(self, bagnet, idx2label, clip, a, b, n=20):
        label_list, topk_list = [], []
        for i, (key, value) in enumerate(self.fail_list.items()):
            image, label, loc = value
            x1, y1 = loc
            x2, y2 = x1+20, y1+20
            adv = self.adv[key].clone()
            image = image[None].clone()
            image[:, :, x1:x2, y1:y2] = adv[None]
            logits = bagnet(image.cuda())
            if clip:
                logits = clip(logits, a, b)
            logits = torch.mean(logits, dim=(1, 2))
            _, topk = torch.topk(logits, k=5, dim=1)
            image = undo_imagenet_preprocess(image[0]).cpu().numpy()
            image = convert2channel_last(image)
            plt.imshow(image)
            plt.title('{} {}'.format(label, idx2label[label]))
            plt.axis('off')
            plt.show()
            topk = topk[0].cpu().numpy()
            print('top-5 prediction: {}'.format(topk))
            topk_list.append(topk)
            label_list.append(label)
            if i == n-1:
                break
        topk_list = np.array(topk_list)
        label_list = np.array(label_list)
        return topk_list, label_list

    def display_successes(self, bagnet, idx2label, clip, a, b, n=20):
        label_list, topk_list = [], []
        for i, (key, value) in enumerate(self.orig_list.items()):
            if key not in self.adv.keys():
                image, label = value
                image = image[None].clone()
                with torch.no_grad():
                    logits = bagnet(image.cuda())
                if clip:
                    logits = clip(logits, a, b)
                logits = torch.mean(logits, dim=(1, 2))
                _, topk = torch.topk(logits, k=5, dim=1)
                image = undo_imagenet_preprocess(image[0])
                plt.imshow(convert2channel_last(image.cpu().numpy()))
                plt.title('{} {}'.format(label, idx2label[label]))
                plt.axis('off')
                plt.show()
                topk = topk[0].cpu().numpy()
                print('top-5 prediction: {}'.format(topk))
                label_list.append(label)
                topk_list.append(topk)
                if i == n-1:
                    break
        label_list = np.array(label_list)
        topk_list = np.array(topk_list)
        return topk_list, label_list

def undefended_batch_upper_bound(model, metabatch, 
                      attack_size, stride, k=5,
                      attack_alg=LinfPGDAttack, eps=5., nb_iter=40, stepsize=0.5, rand_init=False):
    """ 
    Input:
    - model (pytorch model): bagnet model without avgerage pooling
    - images (pytorch tensor): a batch of preprocessed images, same context as model's
    - labels (pytorch tensor): image labels, same context as model's
    - clip (python function): clipping function
    - a, b (float): clipping parameters
    Output:
    - succ_prob (float):
    """
    earlystop = False
    while not earlystop:
        for x in range(0, 224 - attack_size[0] + 1, stride):
            for y in range(0, 224 - attack_size[1] + 1, stride):
                print('current location {}'.format((x, y)))
                logging.info('current location {}'.format((x, y)))
                metabatch.location = (x, y)
                adv_images = metabatch.images.clone()
                subimg = get_subimgs(adv_images, (x, y), attack_size)
                subimg = subimg.cuda()
                labels = metabatch.labels.cuda()
                adver_model = UndefendedAdverTorchWrapper(model, adv_images, 
                                                attack_size, (x, y)).cuda()
                adver_model.eval()
                adversary = attack_alg(adver_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                              	       nb_iter=nb_iter, eps_iter=stepsize, rand_init=rand_init, 
				       clip_min=-1.8044, clip_max=2.2489, # only for ImageNet data set.
                                       targeted=False)
                adv = adversary.perturb(subimg, labels).cpu()
                # apply stickers to image
                adv_images[:, :, adver_model.x1:adver_model.x2, adver_model.y1:adver_model.y2] = adv
                # evaluate attack
                with torch.no_grad():
                    logits = model(adv_images.cuda())
                    _, topk = torch.topk(logits, k=k, dim=1)
                    l, topk = labels.cpu().numpy(), topk.cpu().numpy()
                    mis_indices = [idx for idx in range(len(l)) if l[idx] not in topk[idx]]
                    print('misclassified indices: {}'.format(mis_indices))
                    logging.info('misclassified indices: {}'.format(mis_indices))
                    metabatch.update(mis_indices, adv)
                print("Defense (total: {}): \n succeed: {}, fail: {}".format(len(metabatch.waitlist), len(metabatch.succ_list), len(metabatch.fail_list)))
                if len(metabatch.succ_list) + len(metabatch.fail_list) == len(metabatch.waitlist): # Early stop if already determine success or failure of all the images
                    earlystop = True
                    break
            if earlystop:
                break
    return metabatch.get_succ_prob()

def batch_upper_bound(model, metabatch, clip, a, b, 
                      attack_size, stride, k=5,
                      attack_alg=LinfPGDAttack, eps=5., nb_iter=40, stepsize=0.5, rand_init=False):
    """ 
    Input:
    - model (pytorch model): bagnet model without avgerage pooling
    - images (pytorch tensor): a batch of preprocessed images, same context as model's
    - labels (pytorch tensor): image labels, same context as model's
    - clip (python function): clipping function
    - a, b (float): clipping parameters
    Output:
    - succ_prob (float):
    """
    earlystop = False
    while not earlystop:
        for x in range(0, 224 - attack_size[0] + 1, stride):
            for y in range(0, 224 - attack_size[1] + 1, stride):
                print('current location {}'.format((x, y)))
                logging.info('current location {}'.format((x, y)))
                metabatch.location = (x, y)
                adv_images = metabatch.images.clone()
                subimg = get_subimgs(adv_images, (x, y), attack_size)
                subimg = subimg.cuda()
                labels = metabatch.labels.cuda()
                adver_model = AdverTorchWrapper(model, adv_images, 
                                                attack_size, (x, y), 
                                                clip, a, b).cuda()
                adver_model.eval()
                adversary = attack_alg(adver_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                              	       nb_iter=nb_iter, eps_iter=stepsize, rand_init=rand_init, 
				       clip_min=-1.8044, clip_max=2.2489, # only for ImageNet data set.
                                       targeted=False)
                adv = adversary.perturb(subimg, labels).cpu()
                # apply stickers to image
                adv_images[:, :, adver_model.x1:adver_model.x2, adver_model.y1:adver_model.y2] = adv
                # evaluate attack
                with torch.no_grad():
                    logits = model(adv_images.cuda())
                    if clip:
                        logits = clip(logits, a, b)
                    logits = torch.mean(logits, dim=(1, 2))
                    _, topk = torch.topk(logits, k=k, dim=1)
                    l, topk = labels.cpu().numpy(), topk.cpu().numpy()
                    mis_indices = [idx for idx in range(len(l)) if l[idx] not in topk[idx]]
                    print('misclassified indices: {}'.format(mis_indices))
                    logging.info('misclassified indices: {}'.format(mis_indices))
                    metabatch.update(mis_indices, adv)
                print("Defense (total: {}): \n succeed: {}, fail: {}".format(len(metabatch.waitlist), len(metabatch.succ_list), len(metabatch.fail_list)))
                if len(metabatch.succ_list) + len(metabatch.fail_list) == len(metabatch.waitlist): # Early stop if already determine success or failure of all the images
                    earlystop = True
                    break
            if earlystop:
                break
    return metabatch.get_succ_prob()

def scheduled_upper_bound(model, metabatch, 
                          attack_size, stride, k=5,
                          clip=binarize, a=25, b=None,
                          attack_alg=LinfPGDAttack, eps=5., nb_iter=40, stepsize=0.5, rand_init=False):
    """ 
    Input:
    - model (pytorch model): bagnet model without avgerage pooling
    - images (pytorch tensor): a batch of preprocessed images, same context as model's
    - labels (pytorch tensor): image labels, same context as model's
    - clip (python function): clipping function
    - a, b (float): clipping parameters
    Output:
    - succ_prob (float):
    """
    earlystop = False
    while not earlystop:
        for x in range(0, 224 - attack_size[0] + 1, stride):
            for y in range(0, 224 - attack_size[1] + 1, stride):
                print('current location {}'.format((x, y)))
                logging.info('current location {}'.format((x, y)))
                metabatch.location = (x, y)
                adv_images = metabatch.images.clone()
                subimg = get_subimgs(adv_images, (x, y), attack_size)
                subimg = subimg.cuda()
                labels = metabatch.labels.cuda()
                msg = "labels: {}".format(labels)
                print(msg)
                logging.info(msg)
                adver_model = ScheduledParamAdverTorchWrapper(model, adv_images, attack_size, (x, y), nb_iter=nb_iter).cuda()
                adver_model.eval()
                adversary = attack_alg(adver_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                              	       nb_iter=nb_iter, eps_iter=stepsize, rand_init=rand_init, 
				       clip_min=-1.8044, clip_max=2.2489, # only for ImageNet data set.
                                       targeted=False)
                adv = adversary.perturb(subimg, labels).cpu()
                # apply stickers to image
                adv_images[:, :, adver_model.x1:adver_model.x2, adver_model.y1:adver_model.y2] = adv
                # evaluate attack
                with torch.no_grad():
                    logits = model(adv_images.cuda())
                    if clip:
                        logits = clip(logits, a, b)
                    logits = torch.mean(logits, dim=(1, 2))
                    _, topk = torch.topk(logits, k=k, dim=1)
                    l, topk = labels.cpu().numpy(), topk.cpu().numpy()
                    print("topk prediction with stickers: \n {}".format(topk))
                    mis_indices = [idx for idx in range(len(l)) if l[idx] not in topk[idx]]
                    print('misclassified indices: {}'.format(mis_indices))
                    logging.info('misclassified indices: {}'.format(mis_indices))
                    metabatch.update(mis_indices, adv)
                if len(metabatch.succ_list) + len(metabatch.fail_list) == len(metabatch.waitlist): # Early stop if already determine success or failure of all the images
                    earlystop = True
                    break
            if earlystop:
                break
    return metabatch.get_succ_prob()


def get_targeted_classes(model, images, clip, a, b, k=5, case='avg'):
    """ 
    Input:
    - model (pytorch model): bagnet model without avgerage pooling
    - images (pytorch tensor): a batch of preprocessed images, same context as model's images 
    - k (int): get targeted classes based on top-K prediction
    - case (string): ['best', 'avg', 'worst']
        'best': the class ranked at (k+1)-th
        'avg': randomly sample one class outside of the top-k predictions
        'worst': the class ranked the last
    Output:
    - targeted class (pytorch tensor): targeted class for targeted attack, same context as model's.
    """
    assert case in ['best', 'avg', 'worst'], 'case must be "best", "avg", or "worst"'
    logits = model(images)
    if clip:
        logits = clip(logits, a, b)
    logits = torch.mean(logits, dim=(1, 2)) # shape (N, 1000)
    if case == 'best':
        _, topk = torch.topk(logits, k=k+1, dim=1) # shape (N, k+1) 
        return topk[:, -1]
    if case == 'avg':
        _, topk = torch.topk(logits, k=1000, dim=1) # shape (N, 1000) 
        indices = torch.randint(low=5, high=999, size=(1,))
        return torch.index_select(topk, 1, indices.cuda()).reshape((-1,))
    else: # if case == 'worst'
        _, topk = torch.topk(logits, k=1000, dim=1) # shape (N, 1000) 
        return topk[:, -1]

def targeted_batch_upper_bound(model, metabatch, clip, a, b, 
                               attack_size, stride, k=5, targeted=True, case='avg',
                               attack_alg=LinfPGDAttack, eps=5., nb_iter=40, stepsize=0.5, rand_init=True):
    """ 
    Input:
    - model (pytorch model): bagnet model without avgerage pooling
    - images (pytorch tensor): a batch of preprocessed images, same context as model's
    - labels (pytorch tensor): image labels, same context as model's
    - clip (python function): clipping function
    - a, b (float): clipping parameters
    Output:
    - succ_prob (float):
    """
    earlystop = False
    while not earlystop:
        for x in range(0, 224 - attack_size[0] + 1, stride):
            for y in range(0, 224 - attack_size[1] + 1, stride):
                print('current location {}'.format((x, y)))
                logging.info('current location {}'.format((x, y)))
                metabatch.location = (x, y)
                adv_images = metabatch.images.clone()
                orig_labels = metabatch.labels
                if targeted:
                    with torch.no_grad():
                        labels = get_targeted_classes(model, adv_images.cuda(), clip, a, b, k=k, case=case)
                        msg1, msg2 = "targeted labels: {}".format(labels), "original labels: {}".format(metabatch.labels)
                        print(msg1)
                        print(msg2)
                        logging.info(msg1)
                        logging.info(msg2)
                else:
                    labels = metabatch.labels.cuda()
                subimg = get_subimgs(adv_images, (x, y), attack_size)
                subimg = subimg.cuda()
                adver_model = AdverTorchWrapper(model, adv_images, 
                                                attack_size, (x, y), 
                                                clip, a, b).cuda()
                adver_model.eval()
                adversary = attack_alg(adver_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                              	       nb_iter=nb_iter, eps_iter=stepsize, rand_init=rand_init, 
				       clip_min=-1.8044, clip_max=2.2489, # only for ImageNet data set.
                                       targeted=True)
                adv = adversary.perturb(subimg, labels).cpu()
                # apply stickers to image
                adv_images[:, :, adver_model.x1:adver_model.x2, adver_model.y1:adver_model.y2] = adv
                # evaluate attack
                with torch.no_grad():
                    logits = model(adv_images.cuda())
                    if clip:
                        logits = clip(logits, a, b)
                    logits = torch.mean(logits, dim=(1, 2))
                    _, topk = torch.topk(logits, k=k, dim=1)
                    l, topk = orig_labels.numpy(), topk.cpu().numpy()
                    print("topk prediction with stickers: \n {}".format(topk))
                    if targeted:
                        tl = labels.cpu().numpy()
                        mis_indices = [idx for idx in range(len(l)) if l[idx] not in topk[idx] and tl[idx] in topk[idx]]
                    else:
                        mis_indices = [idx for idx in range(len(l)) if l[idx] not in topk[idx]]
                    print('misclassified indices: {}'.format(mis_indices))
                    logging.info('misclassified indices: {}'.format(mis_indices))
                    metabatch.update(mis_indices, adv, targeted_labels=labels.cpu().numpy())
                if len(metabatch.succ_list) + len(metabatch.fail_list) == len(metabatch.waitlist): # Early stop if already determine success or failure of all the images
                    earlystop = True
                    break
            if earlystop:
                break
    return metabatch.get_succ_prob()
