import torch
from AdamOptimizer import *

class StickerSPSA:
    def __init__(self, model, subimg, label, sticker_size=(20, 20), 
                 delta = 0.01, num_samples=128, step_size=0.01, random_init=True, epsilon=1e-10):
        self.model = model
        self.clean_subimg = subimg.clone()
        #self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).cuda()
        #self.std = torch.tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1)).cuda()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).to(subimg.get_device())
        self.std = torch.tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1)).to(subimg.get_device())

        self.clean_undo_subimg = self.undo_imagenet_preprocess_pytorch(subimg)
        if random_init:
            self.adv_subimg = (torch.rand((1, 3)+sticker_size).to(torch.float).cuda() - self.mean)/self.std
        else:
            self.adv_subimg = subimg.clone()
        self.label = label
        self.sticker_size = sticker_size
        self.adv_pertub = None
        self.num_samples = num_samples
        self.delta = delta
        self.epsilon = epsilon
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

        # calculate the margin logit loss
        label_logit = logits[:, self.label].reshape((-1, )).clone()
        #print(f'{(label_logit.min().item(), label_logit.max().item())}')
        value, indices = torch.topk(logits, k=5, dim=1)
        logits[:, self.label] = float('-inf')
        #print(f'{(label_logit.min().item(), label_logit.max().item())}')
        best_other_logit, _ = torch.max(logits, dim=1)
        ml_loss = label_logit - best_other_logit

        # estimate the gradient
        all_grad = ml_loss.reshape((-1, 1, 1, 1)) / (delta_x + self.epsilon)
        est_grad = torch.mean(all_grad, dim=0)
        
        adam_grad = self.adam_optimizer(est_grad[None])
        
        self.adv_subimg += adam_grad

        # clip the pixel to the valid range
        adv_undo_subimg = self.undo_imagenet_preprocess_pytorch(self.adv_subimg)
        
        adv_undo_subimg = torch.clamp(adv_undo_subimg, 0, 1)

        self.adv_subimg = (adv_undo_subimg - self.mean) / self.std

