import os
import torch
import torch.nn as nn
import numpy as np
import time
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
