from utils import *
from clipping import *
import pytorchnet
import torch
import numpy as np
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Check GPU(s)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
gpu_count = torch.cuda.device_count()
print(f"number of gpu(s): {gpu_count}")
if use_cuda:
    print(torch.cuda.get_device_name(0))
else:
    print("using cpu")

# Prepare dataloader
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([[0.229, 0.224, 0.225]]).reshape((3, 1, 1))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
imagenet_transform = transforms.Compose([transforms.ToTensor(), 
                                         normalize])
folder = datasets.ImageFolder("/mnt/data/code/bagnet-patch-defense/code/attacks/spsa/spsa_validation_images/vulnerable", transform=imagenet_transform)
id2id = {value:int(key) for key, value in folder.class_to_idx.items()}
val_loader = torch.utils.data.DataLoader(folder,
                                         batch_size=1,
                                         num_workers=4)

output_root = '/mnt/data/results/spsa_results/vulnerable_images'

# load pretrained model
model = pytorchnet.bagnet33(pretrained=True, avg_pool=False).to(device)
model.eval()

# Get clean prediction and logits
N = 1
with torch.no_grad():
    count = 0
    for image, label in val_loader:
        true_label = id2id[label.item()]
        image = image.to(device)
        logits = model(image)
        logits = tanh_linear(logits, a=0.05, b=-1)
        logits = torch.mean(logits, dim=(1, 2))
        _, topk = torch.topk(logits, k=5, dim=1)
        topk = topk.cpu().numpy()[0]
        print(f"label: {true_label}, topk: {topk}")
        if true_label in topk:
            count += 1
    clean_acc = count / N
    print("Accuracy before attack: {}".format(clean_acc))

# SPSA configuration
wrapper=DynamicClippedPatchAttackWrapper
sticker_size = (20, 20)
step_size = 0.1
stride = 20
clip_fn=tanh_linear
a=0.05
b=-1
number_iter = 250
x = 20
y = 20
n = 13


# Attack
subimg = get_subimgs(image, (x, y), sticker_size)

wrapped_model = wrapper(model, image.clone(), sticker_size, (x, y), clip_fn, a, b)

if gpu_count > 1:
    wrapped_model = nn.DataParallel(wrapped_model)

spsa_attack = StickerSPSA(wrapped_model, subimg, true_label, sticker_size=sticker_size, step_size=step_size)

for i in range(number_iter):
    print(i)
    spsa_attack.run()


# evaluate the sticker
logits = wrapped_model(spsa_attack.adv_subimg)
values, topk = torch.topk(logits, 5, dim=1)
topk = topk[0].cpu().numpy()
if true_label not in topk: #TODO: add not
    print(f"Successfully attack at {(x, y)}")
    adv_img = image[0].cpu().numpy()
    adv_subimg = spsa_attack.adv_subimg[0].cpu().numpy()
    adv_img = apply_sticker(adv_subimg, adv_img, (x, y), sticker_size)
    adv_img = (adv_img*std) + mean
    adv_img = adv_img.transpose([1, 2, 0])
    adv_img = np.clip(adv_img, 0, 1)
    plt.imsave(os.path.join(output_root, f"{n}-{number_iter}.png"), adv_img)
else:
    print(f"Fail to attack at {(x, y)}")
    #TODO: no need to save failure picture
    adv_img = image[0].cpu().numpy()
    adv_subimg = spsa_attack.adv_subimg[0].cpu().numpy()
    adv_img = apply_sticker(adv_subimg, adv_img, (x, y), sticker_size)
    adv_img = (adv_img*std) + mean
    adv_img = adv_img.transpose([1, 2, 0])
    adv_img = np.clip(adv_img, 0, 1)
    plt.imsave(os.path.join(output_root, f"{n}-{x}-{y}-{number_iter}.png"), adv_img)
print(f"label: {true_label}, topk: {topk}")
print(f"top-5 logits: {values}")
