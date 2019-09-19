from utils import *
from clipping import *
import pytorchnet
import torch
import numpy as np
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    print(torch.cuda.get_device_name(0))
else:
	print("using cpu")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
imagenet_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224), 
                                         transforms.ToTensor(), 
                                         normalize])
imagenet_val = datasets.ImageNet("/mnt/data/imagenet", split='val', download=False,
                                     transform=imagenet_transform)

val_subset_indices = image_partition(42, 10)[0]
val_subset_loader = torch.utils.data.DataLoader(imagenet_val,
                                                batch_size=1,
                                                num_workers=4,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(val_subset_indices))

# load pretrained model
bagnet33 = pytorchnet.bagnet33(pretrained=True, avg_pool=False).to(device)
bagnet33.eval()
print()

run_sticker_spsa(val_subset_loader, bagnet33, 10)
