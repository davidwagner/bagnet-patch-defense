import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, transform

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
    return image.transpose([1, 2, 0])

def imagenet_preprocess(image):
    # preprocess sample image before training
    image = image / 255.
    image -= np.array([0.485, 0.456, 0.406])[:, None, None]
    image /= np.array([0.229, 0.224, 0.225])[:, None, None]
    return image

def extract_patches(image, patchsize):
    patches = image.permute(0, 2, 3, 1)
    patches = patches.unfold(1, patchsize, 1).unfold(2, patchsize, 1)
    patches = patches.contiguous().view((-1, 3, patchsize, patchsize))
    return patches

def bagnet_predict(bagnet, patches, device, batch_size=1000, return_class=True):
    with torch.no_grad():
        cum_logits = torch.zeros(1000).to(device) # ImageNet has 1000 classes
        for batch_patches in torch.split(patches, batch_size):
            logits = bagnet(batch_patches)
            
            sum_logits = torch.sum(logits, dim=0)
            cum_logits += sum_logits
        p = F.softmax(cum_logits/(244*244), dim=0)
        if return_class:
            return torch.argmax(p).item()
        else:
            return p.cpu().numpy()

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