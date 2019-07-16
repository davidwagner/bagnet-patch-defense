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

def is_invariant(class_lower, class_upper, target, k=6):
    """Given the upper bound and lower bound of class logits of one image, determine
        whether the classifier could possibly misclassify this image in top-K prediction.
    Input:
    - class_upper (pytorch tensor): shape = (1000,) in CPU
    - class_lower (pytorch tensor): shape = (1000,) in CPU
    - targets (int): image's label
    - k (int): top-K prediction
    Output (output): whether the classifier is robust against the attack on this image
    """
    target_lower = class_lower[target].item()
    top6_values, top6_classes = torch.topk(class_upper, k=k)
    top6 = top6_values.numpy()
    return target_lower > top6_values[-1].item()


def get_affected_patches(attack_size, position):
    """Given the size and region of an attack, return the coordinate of affected patch
    Input:
    - attack_size (int tuple),: (h, w), hight and width of attack
    - position (int tuple): top-left pixel's coordinate of attacking sticker in (224*224)
    Output (tuple): (x1, y1, x2, y2), where (x1, y1) is the coordinate of top-left patch in logit space,
                                and (x2, y2) is the coordinate of bottom-right patch in logit space.
    """
    ax, ay = attack_size
    p1, q1 = position
    p2, q2 = p1 + ax - 1, q1 + ay - 1
    assert p2 <= 223 and q2 <= 223, "sticker shouldn't exceed image"
    assert p2 >= p1 and q2 >= q1, "attack size shouldn't be zero"
    x1, y1 = max((p1 - 33 + 1) // 8, 0), max((q1 - 33 + 1)// 8, 0)
    x2, y2 = min((p2 + 33 - 1) // 8, 23), min((q2 + 33 - 1) // 8, 23)
    return (x1, y1, x2, y2)

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

def get_position_buffer(h, w, attack_size, stride):
    """Buffer of all possible affected patch positions
    """
    buffer = set()
    last_affected_patches = (0, 0, 0, 0)
    for i in range(0, h - attack_size[0]+1, stride):
        for j in range(0, w - attack_size[1]+1, stride):    
            affected_patches = get_affected_patches(attack_size, (i, j))
            buffer.add(affected_patches)
    return buffer

def get_security_lower_bound(bagnet, images, targets, attack_size, clip, stride=1, bound=(-1, 1), **kwargs):
    """ Given a batch of images and the size of adversarial sticker, loop over each of  
            the possible position of the sticker to count how many images in the batch 
            can still be correctly classified if the sticker is at that position.
    Input:
    - bagnet (pytorch model): BagNet-33 without avgerage pooling
    - images (pytorch tensor): images given by foolbox.sample, after preprocessing, in GPU
    - targets (numpy array): targets
    - attack_size (tuple): (h, w), height and width of the adversarial sticker
    - clip (function): clipping function
    - stride (int): stride of sticker
    - bound (tuple): tanh_linear: (-1, 1), sigmoid_linear: (-1, 1)
    Output:
    - succ_prob (float): fraction of images whose prediction are invariant in the worst case
    """
    clipped_patch_logits = clip_logits(bagnet, clip, images, **kwargs)
    clipped_patch_logits = clipped_patch_logits.cpu()
    n, c, h, w = images.shape
    num_invariant = 0
    positions = get_position_buffer(h, w, attack_size, stride)
    for k in range(n):
        img_patch_logits = clipped_patch_logits[k, :] # shape (24, 24, 1000)
        target = targets[k]
        flag = True
        for p in positions:
            class_lower, class_upper = bound_patch_attack(img_patch_logits, p, bound=bound)
            if not is_invariant(class_lower, class_upper, target):
                flag = False
                break
        if flag:
            num_invariant += 1
    succ_prob = num_invariant / n
    return succ_prob

# pytorch data loader version
def get_security_lower_bound2(bagnet, data_loader, attack_size, clip, stride=1, bound=(-1, 1), **kwargs):
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
        clipped_patch_logits = clip_logits(bagnet, clip, images, **kwargs)
        clipped_patch_logits = clipped_patch_logits.cpu()
        n, c, h, w = images.shape
        total_images += n
        for k in range(n):
            img_patch_logits = clipped_patch_logits[k] # shape (24, 24, 1000)
            target = targets[k]
            flag = True
            for p in positions:
                class_lower, class_upper = bound_patch_attack(img_patch_logits, p, bound=bound)
                if not is_invariant(class_lower, class_upper, target):
                    flag = False
                    break
            if flag:
                num_invariant += 1
    succ_prob = num_invariant / total_images
    return succ_prob

#########################################
# Security Upper Bound
#########################################

class PatchAttackWrapper(nn.Module):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view((3, 1, 1))
    std = torch.Tensor([0.229, 0.224, 0.225]).view((3, 1, 1))
    
    def __init__(self, model, img, size, loc, clip, a, b):
        super(PatchAttackWrapper, self).__init__()
        self.batch_size = img.shape[0]
        self.model = model
        self.clip = clip
        self.a = a
        self.b = b
        self.x1, self.y1 = loc
        self.x2, self.y2 = self.x1 + size[0], self.y1 + size[1]
        self.mask = torch.ones((1, 3, 224, 224))
        self.mask[:, :, self.x1:self.x2, self.y1:self.y2] = 0
        self.img = (self.mask*img).cuda()
        self.img.requires_grad = False
    
    def forward(self, subimg):
        sticker = self._make_sticker(subimg)
        attacked_img = self.img + sticker
        clipped_patch_logits = self._clip_logits(attacked_img, a = self.a, b = self.b)
        clipped_class_logits = torch.mean(clipped_patch_logits, dim=(1, 2))
        return clipped_class_logits
    
    def undo_preprocess(self):
        undo_img = self.img[0].cpu() * self.std
        undo_img += self.mean
        undo_img *= self.mask[0]
        return undo_img
    
    def apply_sticker(self, adversarial):
        """ Apply an adversarial sticker to the original image
        Input:
        - adversarial (numpy array): shape = (3, (sticker size)) pixel values in [0, 1]
        """
        sticker = np.zeros((3, 224, 224))
        sticker[:, self.x1:self.x2, self.y1:self.y2] = adversarial
        # undo preprocessing
        attacked_img = self.img[0].cpu().numpy()
        attacked_img *= self.std.numpy()
        attacked_img += self.mean.numpy()
        attacked_img *= self.mask[0].numpy()
        # apply sticker
        attacked_img += sticker
        return attacked_img
    
    def _clip_logits(self, attacked_img, **kwargs):
        patch_logits = self.model(attacked_img)
        return self.clip(patch_logits, **kwargs)
        
    def _make_sticker(self, subimg):
        sticker = torch.zeros((1, 3, 224, 224)).cuda()
        sticker[:, :, self.x1:self.x2, self.y1:self.y2] = subimg
        return sticker
    
def get_subimgs(images, loc, size):
    """Take an of IMAGES, location LOC and SIZE of stickers, 
        return the sub-image that are for adversarial stickers
    Input:
    - image (numpy array): images, shape = (n, 3, 224, 224)
    - loc (tuple): (x1, y1) coordinate of the upper-left corner of stickers
    - size (tuple): (w, h) width and height of stickers
    Output: 
    -subimage (numpy array): subimage shape = (n, 3, w, h)
    """
    x1, y1 = loc
    x2, y2 = x1 + size[0], y1 + size[1]
    return images[:, :, x1:x2, y1:y2]

def get_subimg(image, loc, size):
    """Take an of IMAGES, location LOC and SIZE of stickers, 
        return the sub-image that are for adversarial stickers
    Input:
    - image (numpy array): images, shape = (3, 224, 224)
    - loc (tuple): (x1, y1) coordinate of the upper-left corner of stickers
    - size (tuple): (w, h) width and height of stickers
    Output: 
    -subimage (numpy array): subimage shape = (3, w, h)
    """
    x1, y1 = loc
    x2, y2 = x1 + size[0], y1 + size[1]
    return image[:, x1:x2, y1:y2]

def print_attack_result(wrapper, subimg, adversarial, attacked_img, output_path):
    """Print the masked image, subimage, and adversarial patch
    Input:
    - wrapper (an instance of PatchAttackWrapper): a model wrapper
    - subimg (numpy array): subimg, shape = (3, h, w)
    - adversarial (numpy array): adversarial patch, shape = (3, h, w)
    """
    plt.imshow(convert2channel_last(attacked_img))
    plt.axis('off')
    plt.show()
    plt.imsave(output_path+'_att.png', convert2channel_last(attacked_img))
    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot(141)
    ax.set_title('masked image')
    plt.imshow(convert2channel_last(wrapper.undo_preprocess().cpu().numpy()))
    ax = plt.subplot(142)
    ax.set_title('original')
    plt.imshow(convert2channel_last(subimg))
    ax = plt.subplot(143)
    ax.set_title('adversarial')
    plt.imshow(convert2channel_last(adversarial))
    ax = plt.subplot(144)
    ax.set_title('difference (x500)')
    plt.imshow(convert2channel_last(np.abs(subimg - adversarial)))
    plt.savefig(output_path+'.png')
    plt.show()

class FakeZeroDistance(Distance):
    def _calculate(self):
        return 0, None

def get_security_upper_bound(bagnet33, images, labels, 
                             attack_alg, attack_size, clip, a, b, 
                             wrapper=PatchAttackWrapper, stride=1, k = 5, 
                             binary_search=True, epsilon=0.3, stepsize=0.01, max_iter = 40, 
                             mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)), 
                             std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)), 
                             output_path = './results', log_path='./log'):
    """ Given a batch of IMAGES and corresponding LABELS, use the ATTACK_ALG to
        calculate the security upper bound.
    Input:
    - bagnet33 (pytorch model): Bagnet-33 without average pooling
    - images (numpy array): sampled images given by the foolbox from ImageNet 
        without preprocessing, pixel values 0 ~ 255
    - labels (numpy array): labels of images
    - attack_alg (foolbox.attack): attack algorithm implemented in foolbox
    - clip (function): clipping function in clipping.py
    - a, b (float): clipping parameters
    - stride (int): stride of how stickers move
    - k (int): top-k misclassification criteria
    - mean, std (numpy array): mean and standard devication of dataset
    - output_path (str): directory for saving resulting plots
    - log_path (str): directory for logging file
    Output:
    - succ_prob (float): fraction of images whose prediction are not affected by the attack
    """
    assert os.path.isdir(output_path) and os.path.exists(output_path), 'Please make sure that the output directory exists.'
    n, c, h, w = images.shape
    images_tensor = imagenet_preprocess(images)
    images_tensor = torch.from_numpy(images_tensor)
    images = images / 255.  # because our model expects values in [0, 1]
    affected_imgs  = 0
    distance = FakeZeroDistance
    criterion = TopKMisclassification(k)
    for i in range(n):
        label = labels[i]
        flag = True
        for x in range(0, h - attack_size[0] + 1, stride):
            for y in range(0, w - attack_size[1] + 1, stride):
                if log_path:
                    assert os.path.isdir(log_path) and os.path.exists(log_path), 'Please make sure that the logging directory exists.'
                    INFO_LOG_FILENAME = os.path.join(log_path, 'img-{}_info-logging.log'.format(i))
                    logging.basicConfig(filename=INFO_LOG_FILENAME, level=logging.INFO)
                wrapped_bagnet33 = wrapper(bagnet33, images_tensor[i][None], 
                                                      attack_size, (x, y), 
                                                      clip, a, b).cuda()
                wrapped_bagnet33.eval()
                fbagnet33 = foolbox.models.PyTorchModel(wrapped_bagnet33, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
                subimg = get_subimg(images[i], (x, y), attack_size)
                adv = Adversarial(fbagnet33, criterion, subimg, label, distance=distance)
                attack_alg(adv, binary_search=binary_search, stepsize=0.01, iterations = max_iter)
                adversarial = adv.perturbed
                if adversarial is not None:
                    print('Image {}, attack successfully, location {}'.format(i, (x, y)))
                    plt_name = 'img_{}_size_{}-{}_loc_{}-{}'.format(i, attack_size[0], attack_size[1], x, y)
                    attacked_img = wrapped_bagnet33.apply_sticker(adversarial)
                    print_attack_result(wrapped_bagnet33, subimg, adversarial, attacked_img, os.path.join(output_path, plt_name))
                    affected_imgs += 1
                    flag = False
                    break
                else:
                    print('Image {}: fail to find an adversarial sticker at {}'.format(i, (x, y)))
            if not flag:
                break
    num_invariant = n - affected_imgs
    succ_prob = num_invariant / n
    return succ_prob
# pytorch loader version
def get_security_upper_bound2(bagnet33, data_loader, 
                             attack_alg, attack_size, clip, a, b, 
                             stride=1, k = 5, max_iter = 40,
                             mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)), 
                             std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)), 
                             output_path = './results'):
    """ Take in a pytorch data loader, use the ATTACK_ALG to
        calculate the security upper bound.
    Input:
    - bagnet33 (pytorch model): Bagnet-33 without average pooling
    - data_loader (pytorch DataLoader): pytorch dataloader in CPU, without normalization, pixel ~ [0, 1]
    - attack_alg (foolbox.attack): attack algorithm implemented in foolbox
    - clip (function): clipping function in clipping.py
    - a, b (float): clipping parameters
    - stride (int): stride of how stickers move
    - k (int): top-k misclassification criteria
    - mean, std (numpy array): mean and standard devication of dataset
    - output_path (str): directory for saving resulting plots
    Output:
    - succ_prob (float): fraction of images whose prediction are not affected by the attack
    """
    assert os.path.isdir(output_path) and os.path.exists(output_path), 'Please make sure that the output directory exists.'
    affected_imgs, total_images = 0, 0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
    distance = FakeZeroDistance
    criterion = TopKMisclassification(k)
    for images, labels in data_loader:
        n, c, h, w = images.shape
        total_images += n
        numpy_images = images.numpy()
        labels = labels.numpy()
        for i in range(n):
            label = labels[i]
            flag = True
            norm_image = normalize(images[i])
            for x in range(0, h - attack_size[0] + 1, stride):
                for y in range(0, w - attack_size[1] + 1, stride):
                    wrapped_bagnet33 = PatchAttackWrapper(bagnet33, norm_image[None], 
                                                          attack_size, (x, y), 
                                                          clip, a, b).cuda()
                    wrapped_bagnet33.eval()
                    fbagnet33 = foolbox.models.PyTorchModel(wrapped_bagnet33, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
                    subimg = get_subimg(numpy_images[i], (x, y), attack_size)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        adv = Adversarial(fbagnet33, criterion, subimg, label, distance=distance)
                        attack_alg(adv, iterations = max_iter)
                        adversarial = adv.perturbed
                    if adversarial is not None:
                        print('Image {}, attack successfully, location {}'.format(i, (x, y)))
                        plt_name = 'img_{}_size_{}-{}_loc_{}-{}.png'.format(i, attack_size[0], attack_size[1], x, y)
                        print_attack_result(wrapped_bagnet33, subimg, adversarial, os.path.join(output_path, plt_name))
                        affected_imgs += 1
                        flag = False
                        break
                    else:
                        print('Image {}: fail to find an adversarial sticker at {}'.format(i, (x, y)))
                if not flag:
                    break
    num_invariant = total_images - affected_imgs
    succ_prob = num_invariant / total_images
    return succ_prob

######################################################
# Advertorch Metabatch Upper Bound
######################################################
class AdverTorchWrapper(nn.Module):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view((3, 1, 1))
    std = torch.Tensor([0.229, 0.224, 0.225]).view((3, 1, 1))

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

    def undo_preprocess(self):
        undo_img = self.img[0].cpu() * self.std
        undo_img += self.mean
        return undo_img

    def _clip_logits(self, attacked_img, **kwargs):
        patch_logits = self.model(attacked_img)
        if self.clip:
            return self.clip(patch_logits, **kwargs)
        return patch_logits

class MetaBatch:
    def __init__(self, data_iter, size, max_iter):
        self.global_step = 0
        self.waitlist = data_iter # images waiting for attack, pytorch data iterator, batch size=1, in CPU
        self.exhaust = False
        self.size = size # size of the metabatch
        self.images = torch.zeros((size, 3, 224, 224)) # images under attack, in CPU
        self.record = np.zeros((self.size, 2), dtype=np.int64) # [(image ID 0, # duration 0), (image ID 1, # duration 1), ...]
        self.labels = torch.zeros((self.size, ), dtype=torch.int64)
        assert self.size <= len(data_iter), "Size of MetaBatch shouldn't larger than data iterator's size"
        self.orig_list = {}
        for idx in range(self.size):
            image, label = next(self.waitlist)
            self.orig_list[self.global_step] = (image[0].clone(), label.item())
            self.record[idx, 0] = self.global_step
            self.global_step += 1
            self.images[idx] = image[0].clone()
            self.labels[idx] = label.item()
        self.fail_list = {}
        self.adv = {}
        self.succ_list = []
        self.max_iter = max_iter
        self.location = None
    
    def update(self, indices, adv):
        """ - Replace images that generate adversatial stickers successfully with new images from 
                the wait list.
        Input:
        - indices (list): 
        - adv ():
        """
        # Replace failure cases with new images
        for idx in indices:
            if self.record[idx, 0] not in self.adv.keys(): # don't update images that are already failed.
                self.fail_list[self.record[idx, 0]] = (self.images[idx].clone(), self.labels[idx].item(), self.location) # (image, label, attack location)
                self.adv[self.record[idx, 0]] = adv[idx].clone()
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
        for i, (key, value) in enumerate(self.orig_list):
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
                if len(metabatch.succ_list) + len(metabatch.fail_list) == len(metabatch.waitlist): # Early stop if already determine success or failure of all the images
                    earlystop = True
                    break
            if earlystop:
                break
    return metabatch.get_succ_prob()