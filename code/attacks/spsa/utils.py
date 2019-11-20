import os
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from StickerSPSAObj import *
from StickerSPSAModel import *
from clipping import *
import matplotlib.pyplot as plt
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
gpu_count = torch.cuda.device_count()
print(f"number of gpu(s): {gpu_count}")

class PatchAttackWrapper(nn.Module):

    def __init__(self, model, img, size, loc, clip_fn=None, a=None, b=None):
        super(PatchAttackWrapper, self).__init__()
        self.batch_size = img.shape[0]
        self.model = model
        self.x1, self.y1 = loc
        self.x2, self.y2 = self.x1 + size[0], self.y1 + size[1]
        mask = torch.ones((1, 3, 224, 224)).cuda()
        mask[:, :, self.x1:self.x2, self.y1:self.y2] = 0
        self.img = (mask*img)

    def forward(self, subimg):
        sticker = self._make_sticker(subimg)
        attacked_img = self.img + sticker
        logits = self.model(attacked_img)
        logits = torch.mean(logits, dim=(1, 2))
        return logits

    def _make_sticker(self, subimg):
        sticker = torch.zeros((1, 3, 224, 224)).to(subimg.get_device())
        sticker[:, :, self.x1:self.x2, self.y1:self.y2] = subimg
        return sticker

class DynamicPatchAttackWrapper(PatchAttackWrapper):
    def __init__(self, model, img, size, loc, clip_fn, a=None, b=None):
        super().__init__(model, img, size, loc, clip_fn, a, b)
    
    def _make_sticker(self, subimg):
        bs, _, _, _ = subimg.shape
        sticker = torch.zeros((bs, 3, 224, 224)).cuda()
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
        img = self.img.to(sticker.get_device())
        attacked_img = img + sticker
        logits = self.clip_fn(self.model(attacked_img), self.a, self.b)
        logits = torch.mean(logits, dim=(1, 2))
        return logits

class DynamicClippedPatchAttackWrapper(ClippedPatchAttackWrapper):
    def __init__(self, model, img, size, loc, clip_fn, a=None, b=None):
        super().__init__(model, img, size, loc, clip_fn, a, b)
    
    def _make_sticker(self, subimg):
        bs, _, _, _ = subimg.shape
        sticker = torch.zeros((bs, 3, 224, 224)).to(subimg.get_device())
        sticker[:, :, self.x1:self.x2, self.y1:self.y2] = subimg
        return sticker

def image_partition(seed, total_num, partition_size):
    idx_lst = np.arange(total_num)
    np.random.seed(seed)
    np.random.shuffle(idx_lst)
    num_per_partition = int(total_num/partition_size)
    idx_lst = np.split(idx_lst, num_per_partition)
    return idx_lst

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

def undo_imagenet_preprocess_pytorch(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()
    return (image*std) + mean

def run_sticker_spsa(data_loader, model, num_iter, id2id,
                     wrapper=DynamicClippedPatchAttackWrapper, sticker_size=(20, 20), step_size=0.01, stride=20, clip_fn=tanh_linear, a=0.05, b=-1, output_root='./'):
    for n, (image, label) in enumerate(data_loader):

        # Move the image to GPU and obtain the top-5 prediction on the clean image.
        # TODO: if applicable, apply clipping function
        image = image.cuda()
        true_label = id2id[label.item()]
        logits = model(image)
        if clip_fn:
            logits = clip_fn(logits, a=a, b=b)
        logits = torch.mean(logits, dim=(1, 2))
        values, topk = torch.topk(logits, 5, dim=1)
        topk = topk[0].cpu().numpy()
        print(f'clean topk {(true_label, topk)}')
        print(f'clean top-5 logits: {values}')
        mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        std = np.array([[0.229, 0.224, 0.225]]).reshape((3, 1, 1))
        earlyreturn = False
        
        for x in range(0, 224 - sticker_size[0] + 1, stride):
            if earlyreturn: break
            for y in range(0, 224 - sticker_size[1] + 1, stride):
                tic = time.time()
                if earlyreturn: break
                print(f'Image {n}, current position: {(x, y)}')
                wrapped_model = wrapper(model, image.clone(), sticker_size, (x, y), clip_fn, a, b)
                if gpu_count > 1:
                    wrapped_model = nn.DataParallel(wrapped_model)
                subimg = get_subimgs(image, (x, y), sticker_size)
                
                spsa_attack = StickerSPSA(wrapped_model, subimg, true_label, sticker_size=sticker_size, step_size=step_size) # TODO: adjust step size
                for i in range(num_iter):
                    # TODO: remove print
                    #print(f'iteration {i}')
                    spsa_attack.run()
                tac = time.time()
                print('Time duration for one position: {:.2f} min.'.format((tac - tic)/60))

                # evaluate the sticker
                logits = wrapped_model(spsa_attack.adv_subimg)
                values, topk = torch.topk(logits, 5, dim=1)
                topk = topk[0].cpu().numpy()
                if true_label not in topk: #TODO: add not
                    earlyreturn = True
                    print(f"Successfully attack at {(x, y)}")
                    adv_img = image[0].cpu().numpy()
                    adv_subimg = spsa_attack.adv_subimg[0].cpu().numpy()
                    adv_img = apply_sticker(adv_subimg, adv_img, (x, y), sticker_size)
                    adv_img = (adv_img*std) + mean
                    adv_img = adv_img.transpose([1, 2, 0])
                    adv_img = np.clip(adv_img, 0, 1)
                    plt.imsave(os.path.join(output_root, f"{n}.png"), adv_img)
                else:
                    print(f"Fail to attack at {(x, y)}")
                    #TODO: no need to save failure picture
                    adv_img = image[0].cpu().numpy()
                    adv_subimg = spsa_attack.adv_subimg[0].cpu().numpy()
                    adv_img = apply_sticker(adv_subimg, adv_img, (x, y), sticker_size)
                    adv_img = (adv_img*std) + mean
                    adv_img = adv_img.transpose([1, 2, 0])
                    adv_img = np.clip(adv_img, 0, 1)
                    plt.imsave(os.path.join(output_root, f"{n}-{x}-{y}.png"), adv_img)
                print(f"label: {true_label}, topk: {topk}")
                print(f"top-5 logits: {values}")

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

def image_partition(seed, partition_size):
    idx_lst = np.arange(50000)
    np.random.seed(seed)
    np.random.shuffle(idx_lst)
    num_per_partition = int(50000/partition_size)
    idx_lst = np.split(idx_lst, num_per_partition)
    return idx_lst


def get_subimg(image, loc, size):
    x1, y1 = loc
    x2, y2 = x1 + size[0], y1 + size[1]
    return image[:, x1:x2, y1:y2]

class StickerSPSAEval:
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
        self.current_loss = 0
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
        #cfd = F.softmax(logits, dim=1)

        #    # calculate the margin logit loss
        #self.label_cfd = cfd[:, self.label].reshape((-1, )).clone()
        #self.label_cfd = torch.mean(self.label_cfd).item()
        #cfd[:, self.label] = float('-inf')
        ##print(f'{(label_logit.min().item(), label_logit.max().item())}')
        #value, indices = torch.topk(cfd, k=5, dim=1)
        #self.best_other_cfd = torch.mean(value[:, 0]).item()
        #self.worst_other_cfd = torch.mean(value[:, -1]).item()

        """ margin-based loss """
        # calculate the margin logit loss
        self.label_logit = logits[:, self.label].reshape((-1, )).clone()
        #print(f'{(label_logit.min().item(), label_logit.max().item())}')
        value, indices = torch.topk(logits, k=5, dim=1)
        logits[:, self.label] = float('-inf')
        #print(f'{(label_logit.min().item(), label_logit.max().item())}')
        values, _ = torch.topk(logits, 5, dim=1)
        self.best_other_logit = values[:, 0]
        self.worst_other_logit = values[:, -1]
        #loss = self.label_logit - self.best_other_logit
        loss = self.label_logit - self.worst_other_logit
        self.current_loss = torch.mean(loss).item()

        """cross-entropy
        label = torch.full((self.num_samples,), self.label, dtype=torch.long).cuda()
        loss = self.loss_fn(logits, label)
        """
        
        # estimate the gradient
        all_grad = loss.reshape((-1, 1, 1, 1)) / (delta_x + self.epsilon)
        est_grad = torch.mean(all_grad, dim=0)
        
        adam_grad = self.adam_optimizer(est_grad[None])
        
        self.adv_subimg += adam_grad

        # clip the pixel to the valid range
        adv_undo_subimg = self.undo_imagenet_preprocess_pytorch(self.adv_subimg)
        
        adv_undo_subimg = torch.clamp(adv_undo_subimg, 0, 1)

        self.adv_subimg = (adv_undo_subimg - self.mean) / self.std
