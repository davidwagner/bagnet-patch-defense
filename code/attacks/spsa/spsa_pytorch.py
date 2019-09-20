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
imagenet_transform = transforms.Compose([transforms.ToTensor(), 
                                         normalize])
folder = datasets.ImageFolder("/mnt/data/results/foolbox_results/robust/500-10-20x20-20-bagnet33-tanh_linear-AdamRandomPGD-False-1.0-40-0.1", transform=imagenet_transform)
id2id = {value:int(key) for key, value in folder.class_to_idx.items()}
N = 10
# There are 331 images in the robust dataset
val_subset_indices = image_partition(42, 330, N)[0]
val_subset_loader = torch.utils.data.DataLoader(folder,
                                                batch_size=1,
                                                num_workers=4,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(val_subset_indices))

# load pretrained model
model = pytorchnet.bagnet33(pretrained=True, avg_pool=False).to(device)
model.eval()
print()

with torch.no_grad():
    count = 0
    for image, label in val_subset_loader:
        label = id2id[label.item()]
        image = image.to(device)
        logits = model(image)
        logits = tanh_linear(logits, a=0.05, b=-1)
        logits = torch.mean(logits, dim=(1, 2))
        _, topk = torch.topk(logits, k=5, dim=1)
        topk = topk.cpu().numpy()[0]
        print(f"label: {label}, topk: {topk}")
        if label in topk:
            count += 1
    clean_acc = count / N
    print("Accuracy before attack: {}".format(clean_acc))


run_sticker_spsa(val_subset_loader, model, 1000, id2id, output_root='/mnt/data/results/spsa_results/1000iter')
