from advertorch.attacks import LinfPGDAttack
import bagnets
from bagnets.clipping import*
from bagnets.security import*
from get_robust_images import*
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import time
import os
import pickle
import logging
from foolbox.attacks import ProjectedGradientDescentAttack as PGD
from foolbox_alpha.attacks import AdamRandomPGD
from absl import app, flags
from time import gmtime, strftime
import json


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
imagenet_transform = transforms.Compose([transforms.ToTensor(), normalize])
folder = datasets.ImageFolder('/home/zhanyuan/data/results/foolbox_results/untargeted/densenet-untargeted', transform=imagenet_transform)
loader = torch.utils.data.DataLoader(folder, batch_size=1)
id2id = {value:int(key) for key, value in folder.class_to_idx.items()}

device = torch.device("cuda:0")
model_dic = {"resnet18":models.resnet18(pretrained=True),
                "alexnet":models.alexnet(pretrained=True),
                "vgg16":models.vgg16(pretrained=True),
                "resnet34":models.resnet34(pretrained=True),
                "bagnet9":bagnets.pytorch.bagnet9(pretrained=True, avg_pool=False),
                "bagnet17":bagnets.pytorch.bagnet17(pretrained=True, avg_pool=False),
                "bagnet33":bagnets.pytorch.bagnet33(pretrained=True, avg_pool=False),
                "resnet50":models.resnet50(pretrained=True),
                "resnet101":models.resnet101(pretrained=True),
                "resnet152":models.resnet152(pretrained=True),
                "densenet":models.densenet161(pretrained=True),
                "inception":models.inception_v3(pretrained=True)}

model = model_dic["densenet"].to(device)
model.eval()

print("Start attacking...")
total = 0
with torch.no_grad():
    count = 0
    for image, label in loader:
        total += 1
        image = image.to(device)
        logit = model(image)
        #if FLAGS.model in ["bagnet9", "bagnet17", "bagnet33"] and model.avg_pool == False: # if apply clipping function
        #        logit = clip_fn(logit, FLAGS.a, FLAGS.b)
        #        logit = torch.mean(logit, dim=(1, 2))
        _, topk = torch.topk(logit, k=1, dim=1)
        topk, label = topk.cpu().numpy()[0], id2id[label.numpy()[0]]
        if label in topk:
            count += 1
    clean_acc = count / total
    print(f"total: {total} correct: {count}")
