from utils import *
from ..bagnets.clipping import *
import torch
import numpy as np
import time

def spsa(data_loader, model, wrapper=DynamicClippedPatchAttackWrapper, clip_fn=tanh_linear, a=0.05, b=-1,
		 attack_size=(20, 20), stride=20,
		 delta=0.01, num_samples=128, num_iter=1000, step_size=0.01)
	mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
	std = np.array([[0.229, 0.224, 0.225]]).reshape((3, 1, 1))
	for image, label in (data_loader):
	    image = image.cuda()
		logits = model(image)
		logits = torch.mean(logits, dim=(1, 2))
		print(f'clean topk {label, topk}')
		earlyreturn = False
	    tic = time.time()
	    for x in range(0, h - attack_size[0] + 1, stride):
				if earlyreturn: break
			for y in range(0, w - attack_size[1] + 1, stride):
				if earlyreturn: break
				wrapped_model = wrapper(model, image.clone(), attack_size, (x, y), clip_fn, a, b)
				subimg = get_subimgs(image, (x, y), attack_size)
				adv_subimg = subimg.clone()
				adam_optimizer = AdamOptimizer(shape=(3, 20, 20), learning_rate=step_size)
				for i in range(num_iter):
				    # Sample perturbation from Bernoulli +/- 1 distribution
				    _samples = torch.sign(torch.empty((num_samples//2, 3, 20, 20), dtype=adv_subimg.dtype).uniform_(-1, 1))
				    _samples = _samples.to(device)
				    delta_x = delta * _samples
				    delta_x = torch.cat([delta_x, -delta_x], dim=0) # so there are 2*num_samples
				    _sampled_perturb = adv_subimg + delta_x

				    with torch.no_grad():
				        logits = wrapped_model(_sampled_perturb)

				    # calculate the margin logit loss
				    label_logit = logits[:, label].reshape((-1, ))
				    value, indices = torch.topk(logits, k=5, dim=1)
				    logits[:, label] = float('-inf')
				    best_other_logit, _ = torch.max(logits, dim=1)
				    ml_loss = label_logit - best_other_logit

				    # estimate the gradient
				    all_grad = ml_loss.reshape((-1, 1, 1, 1)) / delta_x
				    est_grad = torch.mean(all_grad, dim=0)

				    # update the sticker
				    adv_subimg += adam_optimizer(est_grad[None])

				    # clip the perturbation so that it is in a valid range
				    adv_pertub = adv_subimg - subimg 
				    adv_pertub = torch.clamp(adv_pertub, -1.8044, 2.2489)
				    adv_subimg = subimg + adv_pertub

		   		# apply sticker
				logits = wrapped_model(adv_subimg)
				values, topk = torch.topk(logits, 5, dim=1)
				topk = topk[0].cpu().numpy()
				if label.item() not in topk:
					earlyreturn = True
				print(f"top-5: {topk}")
	    tac = time.time()
	    print(f'Time duration for one position: {(tac - tic)/60} min.')
