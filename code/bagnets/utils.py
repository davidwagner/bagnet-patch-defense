import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, transform
from bagnets.pytorch import Bottleneck
from keras.preprocessing import image as KImage
import time
import torch
import torch.nn as nn
import math
import logging
from torch.utils import model_zoo

model_urls = {
            'bagnet9': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet8-34f4ccd2.pth.tar',
            'bagnet17': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet16-105524de.pth.tar',
            'bagnet33': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet32-2ddd53ed.pth.tar',
                            }

def image_partition(seed, partition_size):
    idx_lst = np.arange(50000)
    np.random.seed(seed)
    np.random.shuffle(idx_lst)
    num_per_partition = int(50000/partition_size)
    idx_lst = np.split(idx_lst, num_per_partition)
    return idx_lst

def plot_heatmap(heatmap, original, ax, cmap='RdBu_r', 
                 percentile=99, dilation=0.5, alpha=0.25):
    """
    Plots the heatmap on top of the original image 
    (which is shown by most important edges).
    
    Parameters
    ----------
    heatmap : Numpy Array of shape [X, X]
        Heatmap to visualise.
    original : Numpy array of shape [X, X, 3]
        Original image for which the heatmap was computed.
    ax : Matplotlib axis
        Axis onto which the heatmap should be plotted.
    cmap : Matplotlib color map
        Color map for the visualisation of the heatmaps (default: RdBu_r)
    percentile : float between 0 and 100 (default: 99)
        Extreme values outside of the percentile range are clipped.
        This avoids that a single outlier dominates the whole heatmap.
    dilation : float
        Resizing of the original image. Influences the edge detector and
        thus the image overlay.
    alpha : float in [0, 1]
        Opacity of the overlay image.
    
    """
    if len(heatmap.shape) == 3:
        heatmap = np.mean(heatmap, 0)
    
    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, heatmap.shape[1], dx)
    yy = np.arange(0.0, heatmap.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_original = plt.get_cmap('Greys_r')
    cmap_original.set_bad(alpha=0)
    overlay = None
    if original is not None:
        # Compute edges (to overlay to heatmaps later)
        original_greyscale = original if len(original.shape) == 2 else np.mean(original, axis=-1)
        in_image_upscaled = transform.rescale(original_greyscale, dilation, mode='constant', 
                                              multichannel=False, anti_aliasing=True)
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges
    
    abs_max = np.percentile(np.abs(heatmap), percentile)
    abs_min = abs_max
    
    ax.imshow(heatmap, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        ax.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_original, alpha=alpha)
        

def generate_heatmap_pytorch(model, image, target, patchsize):
    """
    Generates high-resolution heatmap for a BagNet by decomposing the
    image into all possible patches and by computing the logits for
    each patch.
    
    Parameters
    ----------
    model : Pytorch Model
        This should be one of the BagNets.
    image : Numpy array of shape [1, 3, X, X]
        The image for which we want to compute the heatmap.
    target : int
        Class for which the heatmap is computed.
    patchsize : int
        The size of the receptive field of the given BagNet.
    
    """
    import torch
    
    with torch.no_grad():
        # pad with zeros
        _, c, x, y = image.shape
        padded_image = np.zeros((c, x + patchsize - 1, y + patchsize - 1))
        padded_image[:, (patchsize-1)//2:(patchsize-1)//2 + x, (patchsize-1)//2:(patchsize-1)//2 + y] = image[0]
        image = padded_image[None].astype(np.float32)
        
        # turn to torch tensor
        input = torch.from_numpy(image).cuda()
        
        # extract patches
        patches = input.permute(0, 2, 3, 1)
        patches = patches.unfold(1, patchsize, 1).unfold(2, patchsize, 1)
        num_rows = patches.shape[1]
        num_cols = patches.shape[2]
        patches = patches.contiguous().view((-1, 3, patchsize, patchsize))

        # compute logits for each patch
        logits_list = []

        for batch_patches in torch.split(patches, 1000):
            logits = model(batch_patches)
            logits = logits[:, target][:, 0]
            logits_list.append(logits.data.cpu().numpy().copy())

        logits = np.hstack(logits_list)
        return logits.reshape((224, 224))


##################################################
# Helper functions from 2019-5-22 notebook
# Reference: https://github.com/wielandbrendel/bag-of-local-features-models/blob/master/bagnets/utils.py
##################################################
def pad_image(image, patchsize):
    _, c, x, y = image.shape
    padded_image = np.zeros((c, x + patchsize - 1, y + patchsize - 1))
    padded_image[:, (patchsize-1)//2 : (patchsize-1)//2 + x, (patchsize-1)//2 : (patchsize-1)//2 + y] = image[0]
    return padded_image[None].astype(np.float32)

def convert2channel_last(image):
    """Convert the format of image to channel-last format
    Input:
    -image (numpy array): image, shape = (3, h, w)
    """
    return image.transpose([1, 2, 0])

def imagenet_preprocess(image):
    # preprocess sample image before training
    image = image / 255.
    image -= np.array([0.485, 0.456, 0.406])[:, None, None]
    image /= np.array([0.229, 0.224, 0.225])[:, None, None]
    return image

def extract_patches(image, patchsize, stride=1):
    patches = image.permute(0, 2, 3, 1)
    patches = patches.unfold(1, patchsize, stride).unfold(2, patchsize, stride)
    patches = patches.contiguous().view((-1, 3, patchsize, patchsize))
    return patches

def bagnet_predict(model, images, k=1, clip=None, a=1e-2, b=-0.78, return_class=True):
    """ Make top-K prediction on IMAGES by MODEL
    Inputs:
    - model: pytorch model. model for prediction
    - images: pytorch tensor. images to be predicted on
    - k: number of classes to return for each image (top-k most possible ones)
    - clip (clipping): clipping function
    - a, b (double): clipping parameters
    - return_class: If True, then return classes as prediction; otherwise, return logits and probability of prediction classes
    Return:
    - indices.cpu().numpy(): numpy array at CPU. prediction K classes
    - l.cpu().numpy(): numpy array at CPU. top-K prediction logits
    - p.cpu().numpy(): numpy array at CPU. top-K prediction probability
    """
    with torch.no_grad():
        logits = model(images)
        if clip:
            assert not model.avg_pool, 'Bagnet should apply clipping before taking average.'
            logits = clip(logits, a, b)
            logits = torch.mean(logits, dim=(1, 2))
        p = torch.nn.Softmax(dim=1)(logits)
        p, indices = torch.topk(p, k, dim=1)
        l, _ = torch.topk(logits, k, dim=1)
        if return_class:
            return indices.cpu().numpy()
        else:
            return l.cpu().numpy(), p.cpu().numpy()

def compare_heatmap(bagnet, patches, gt, target, original, batch_size=1000):
    with torch.no_grad():
        gt_logits_list, target_logits_list = [], []
        for batch_patches in torch.split(patches, batch_size):
            logits = bagnet(batch_patches)
            gt_logits = logits[:, gt][:, 0]
            target_logits = logits[:, target][:, 0]
            gt_logits_list.append(gt_logits.data.cpu().numpy().copy())
            target_logits_list.append(target_logits.data.cpu().numpy().copy())
        gt_logits = np.hstack(gt_logits_list)
        target_logits = np.hstack(target_logits_list)
        gt_heatmap = gt_logits.reshape((224, 224))
        target_heatmap = target_logits.reshape((224, 224))
    fig = plt.figure(figsize=(8, 4))
    original_image = original[0].transpose([1, 2, 0])
    ax = plt.subplot(131)
    ax.set_title('original')
    plt.imshow(original_image / 255.)
    plt.axis('off')
    
    ax = plt.subplot(132)
    ax.set_title('ground true class')
    plot_heatmap(gt_heatmap, original_image, ax, dilation=0.5, percentile=99, alpha=0.25)
    plt.axis('off')
    
    ax = plt.subplot(133)
    ax.set_title('target class')
    plot_heatmap(target_heatmap, original_image, ax, dilation=0.5, percentile=99, alpha=0.25)
    plt.axis('off')
    plt.show()
###################################################

###################################################
# Helper functions from 2019-5-23 notebook
###################################################
def class_patch_logits(bagnet, patches, device, batch_size=1000, num_classes=1000):
    """ Obtain the logits of all the classes across all the patches
    """
    logits_list = []
    with torch.no_grad():
        for batch_patches in torch.split(patches, batch_size):
            logits = bagnet(batch_patches)
            for i in range(num_classes):
                class_logits = logits[:, i]
                logits_list.append(class_logits.data.cpu().numpy().copy())
    return np.hstack(logits_list)
###################################################

###################################################
# Helper functions from 2019-5-24 notebook
###################################################
def attack_patch(image, patchsize, num_patches, seed=None):
    c, x, y = image.shape
    if seed is not None:
        np.random.seed(seed)
    attacked_x = np.random.choice(range(x), size=num_patches, replace=True)
    attacked_y = np.random.choice(range(y), size=num_patches, replace=True)
    for xi, yi in zip(attacked_x, attacked_y):
        c, h, w = image[:, (xi - (patchsize-1)//2): (xi + (patchsize-1)//2), (yi - (patchsize-1) // 2): (yi + (patchsize-1) // 2)].shape
        image[:, (xi - (patchsize-1)//2): (xi + (patchsize-1)//2), (yi - (patchsize-1) // 2): (yi + (patchsize-1) // 2)] = np.random.rand(c, h, w)
    return image
###################################################

###################################################
# Helper function from 5-26 notebook
###################################################
def compute_saliency_map(images, labels, model, criterion, device):
    """Compute saliency map accroding to criterion. This implementation is based on the description in https://arxiv.org/abs/1312.6034
    Input:
    - images (numpy array): images
    - labels (numpy array): labels
    - model (pytorch model): model
    - criterion (pytorch loss function): loss function to compute gradients
    - device: context
    Output:
    - saliency (numpy array): saliency map
    """
    images, labels = torch.from_numpy(images).to(device), torch.from_numpy(labels).to(device)
    images.requires_grad_(True)
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    saliency = np.amax(np.absolute(images.grad.cpu().numpy()), axis=1)
    return saliency

def plot_saliency(images, saliency, alpha=0):
    """Plot saliency map
    Input:
    - images (numpy array): images
    - saliency (numpy array): saliency map
    - alpha (float): transparency
    """
    for i in range(len(saliency)):
        fig = plt.figure(figsize=(8, 4))
        ax = plt.subplot(121)
        ax.set_title('original')
        plt.imshow(convert2channel_last(images[i]))
        plt.axis('off')

        ax = plt.subplot(122)
        ax.set_title('saliency map')
        plt.imshow((saliency[i]), cmap=plt.cm.hot)
        if alpha:
            plt.imshow(convert2channel_last(images[i]), alpha=alpha)
        plt.axis('off')
        plt.show()
       
class BottleneckDebug(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(BottleneckDebug, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=0, bias=False) # changed padding from (kernel_size - 1) // 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            print('Bottleneck: shape before downsampling {}'.format(x.shape))
            residual = self.downsample(x)
        
        if residual.size(-1) != out.size(-1):
            print('Bottleneck: shape after downsampling {}'.format(residual.shape))
            print('Bottleneck: shape of out {}'.format(out.shape))
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:,:,:-diff,:-diff]
        
        out += residual
        out = self.relu(out)

        return out

class BagNetDebug(nn.Module):

    def __init__(self, block, layers, strides=[1, 2, 2, 2], kernel3=[0, 0, 0, 0], num_classes=1000, avg_pool=True):
        self.inplanes = 64
        super(BagNetDebug, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], kernel3=kernel3[0], prefix='layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], kernel3=kernel3[1], prefix='layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], kernel3=kernel3[2], prefix='layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], kernel3=kernel3[3], prefix='layer4')
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avg_pool = avg_pool
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0, prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        print('BagNet: shape after conv1 {}\n'.format(x.shape))
        x = self.conv2(x)
        print('BagNet: shape after conv2 {}\n'.format(x.shape))
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        print('BagNet: shape after layer1 {}\n'.format(x.shape))
        x = self.layer2(x)
        print('BagNet: shape after layer2 {}\n'.format(x.shape))
        x = self.layer3(x)
        print('BagNet: shape after layer3 {}\n'.format(x.shape))
        x = self.layer4(x)
        print('BagNet: shape after layer4 {}\n'.format(x.shape))

        if self.avg_pool:
            print('BagNet: kernel size of AvgPool2d: {}'.format(x.size()[2]))
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            x = x.view(x.size(0), -1)
            print('BagNet: shape after flattening: {}'.format(x.shape))
            x = self.fc(x)
            print('BagNet: shape of final output: {}'.format(x.shape))
        else:
            print('BagNet: return logits of patches')
            x = x.permute(0,2,3,1)
            print('BagNet: shape after transpose: {}'.format(x.shape))
            x = self.fc(x)
            print('BagNet: shape of final output: {}'.format(x.shape))
        return x

def bagnet33_debug(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-33 model (Debugging mode).
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNetDebug(BottleneckDebug, [3, 4, 6, 3], strides=strides, kernel3=[1,1,1,1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet33']))
    return model
###################################################

###################################################
# Helper function from 5-27 notebook
###################################################
class BottleneckRF(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(BottleneckRF, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=0, bias=False) # changed padding from (kernel_size - 1) // 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:,:,:-diff,:-diff]
        
        out += residual
        out = self.relu(out)

        return out

class MaskLayer(nn.Module):
    
    def __init__(self, shape, coordinate, device):
        super(MaskLayer, self).__init__()
        X, Y = coordinate
        self.mask = torch.zeros(shape).to(device)
        for x in X:
            for y in Y:
                self.mask[:, :, x, y] = 1
        
    def forward(self, x):
        return x*self.mask


class BagNetRF(nn.Module):

    def __init__(self, block, mask, layers, strides=[1, 2, 2, 2], kernel3=[0, 0, 0, 0], num_classes=1000, avg_pool=True):
        self.inplanes = 64
        super(BagNetRF, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], kernel3=kernel3[0], prefix='layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], kernel3=kernel3[1], prefix='layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], kernel3=kernel3[2], prefix='layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], kernel3=kernel3[3], prefix='layer4')
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avg_pool = avg_pool
        self.block = block
        self.mask = mask

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0, prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        print('BagNet: shape after layer4 {}\n'.format(x.shape))
        x = self.mask(x)

        if self.avg_pool:
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = x.permute(0,2,3,1)
            x = self.fc(x)

        return x

def bagnet33_RF(batch_size, coordinate, device, pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-33 model (Receptive field mode).
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    mask = MaskLayer((batch_size, 2048, 24, 24), coordinate, device)
    
    model = BagNetRF(BottleneckRF, mask, [3, 4, 6, 3], strides=strides, kernel3=[1,1,1,1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet33']))
    return model
###################################################

###################################################
# Helper function from 2019-5-29 notebook
##################################################
def get_topk_acc(y_hat, y):
    """ Compute top-k accuracy
    Input:
    - y_hat: numpy array with shape (batchsize, K). top-k prediction classes
    - y: numpy array with shape(batchsize, ). target classes
    Return: top-k accuracy
    """
    is_correct = [y[i] in y_hat[i] for i in range(y.size)]
    is_correct = np.array(is_correct)
    return is_correct.sum()/y.size

def validate(val_loader, model, device, k=5, clip=None, **kwargs):
    """Validate model's top-k accuracy
    Input:
    - val_loader: pytorch data loader.
    - model: pytorch model
    - device: context
    - k (int): top-k accuracy
    - clip (function): If not None, apply clipping on patch logits
    Return:
    val_acc: float
    """
    # switch to evaluate mode
    model.eval()
    total_iter = len(val_loader)
    cum_acc = 0
    val_time = 0
    with torch.no_grad():
        start = time.time()
        for i, (images, target) in enumerate(val_loader):
            images, target = images.to(device), target.to(device)
            tic = time.time()
            if clip:
                logits = model(images)
                logits = clip(logits, **kwargs)
                logits = torch.mean(logits, dim=(1, 2))
            else:
                logits = model(images)
            tac = time.time()
            # measure accuracy
            _, y_hat = torch.topk(logits, k=k, dim=1)
            y_hat, target = y_hat.cpu().numpy(), target.cpu().numpy()
            acc = sum([target[i] in y_hat[i] for i in range(target.size)]) / target.size
            cum_acc += acc
            val_time += tac-tic
            msg = 'Iteration {}, validation accuracy: {:.3f}, time: {}s'.format(i, acc, tac-tic)
            print(msg)
            logging.info(msg)
    end = time.time()
    val_acc = cum_acc / total_iter
    msg = 'Validation accuracy: {:.3f}, validation time: {:.2f}, total time: {:.2f}s'.format(val_acc, val_time, end-start)
    print(msg)
    logging.info(msg)
    return val_acc
####################################################

####################################################
# Helper functions from 2019-5-31 notebook
####################################################

def get_low_res_heatmap(bagnet, images, targets, clip=None, **kwargs):
    """Generates low-resolution heatmap for Bagnet-33
    Input:
    - bagnet (pytorch): Bagnet without average pooling
    - images (pytorch tensor): images
    - targets (int list): for which class the heatmap is computed for
    Output: (numpy array) heatmap
    """
    with torch.no_grad():
        patch_logits = bagnet(images)
    patch_logits = patch_logits.permute([0, 3, 1, 2]).cpu().numpy()
    if clip:
        patch_logits = clip(patch_logits, **kwargs)
    N = images.shape[0]
    heatmaps = np.zeros((N, 224, 224))
    for z in range(N):
        patch_target_logits = patch_logits[z, targets[z], :, :]
        for p, i in enumerate(range(33, 224, 8)):
            for q, j in enumerate(range(33, 224, 8)):
                patch = np.full((33, 33), patch_target_logits[p, q])
                heatmaps[z, i-33:i, j-33:j] += patch
    return heatmaps

######################################################

######################################################
# Helper functions from 2019-6-15 notebook
######################################################
def generate_high_res_heatmap(model, patches, target, batchsize=1000, clip=None, **kwargs):
    with torch.no_grad():
        logits_list = []
        for batch_patches in torch.split(patches, batchsize):
            logits = model(batch_patches)
            logits = logits[:, target]
            if clip:
                logits = clip(logits, **kwargs)
            logits_list.append(logits.data.cpu().numpy().copy())
        logits = np.hstack(logits_list)
        return logits.reshape((224, 224))

def bagnet_patch_predict(bagnet, patches, batch_size=1000, k=1, return_class=True):
    with torch.no_grad():
        cum_logits = torch.zeros(1000).cuda() # ImageNet has 1000 classes
        N = patches.shape[0]
        for batch_patches in torch.split(patches, batch_size):
            logits = bagnet(batch_patches)
            sum_logits = torch.sum(logits, dim=0)
            cum_logits += sum_logits
        p = F.softmax(cum_logits/N, dim=0)
        values, indices = torch.topk(p, k, dim=1)
        if return_class:
            return indices.cpu().numpy()
        else:
            return values.cpu().numpy()

#####################################################

#####################################################
# Helper functions from 2019-6-21 notebook
#####################################################
def pad_tensor_image(image, patchsize):
    _, c, x, y = image.shape
    padded_image = torch.zeros((c, x + patchsize - 1, y + patchsize - 1))
    padded_image[:, (patchsize-1)//2 : (patchsize-1)//2 + x, (patchsize-1)//2 : (patchsize-1)//2 + y] = image[0]
    return padded_image[None].float()
#####################################################

#####################################################
# Helper function from 2019-6-22 notebook
#####################################################
def undo_imagenet_preprocess(image):
    """ Undo imagenet preprocessing
    Input:
    - image (pytorch tensor): image after imagenet preprocessing in CPU, shape = (3, 224, 224)
    Output:
    - undo_image (pytorch tensor): pixel values in [0, 1]
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view((3, 1, 1))
    std = torch.Tensor([0.229, 0.224, 0.225]).view((3, 1, 1))
    undo_image = image * std
    undo_image += mean
    return undo_image
#####################################################

######################################################
# Helper functions for CleverHans SPSA sticker attack
######################################################
class CleverhansDataLoader:
    def __init__(self, images, labels):
        """ An iterator yielding one image and its label at a time from a batch of images
        Input:
        - images (numpy array): shape (batch_size, 3, 224, 224)
        - labels (numpy array): shape (batch_size, )
        """
        self.images = images
        self.labels = labels
        self.length = images.shape[0]
        self.count = -1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.count += 1
        if self.count < self.length:
            return (self.images[self.count][None].copy(), np.array(self.labels[self.count]))
        else:
            raise StopIteration
        
    def __len__(self):
        return self.images.shape[0]

def load_image(img_path, mean=np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)), std=np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))):
    img = KImage.load_img(img_path, target_size=(224, 224))
    img = KImage.img_to_array(img).transpose([2, 0, 1])
    img /= 255
    img = (img - mean)/std
    return img
