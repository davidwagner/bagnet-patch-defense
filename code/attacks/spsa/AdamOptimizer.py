import torch
from utils import *

# Reference: https://github.com/FlashTek/foolbox/blob/adam-pgd/foolbox/optimizers.py#L31
class AdamOptimizer:
    """Basic Adam optimizer implementation that can minimize w.r.t.
    a single variable.
    Parameters
    ----------
    shape : tuple
        shape of the variable w.r.t. which the loss should be minimized
    """

    def __init__(self, shape, learning_rate,
                 beta1=0.9, beta2=0.999, epsilon=10e-8):
        """Updates internal parameters of the optimizer and returns
        the change that should be applied to the variable.
        Parameters
        ----------
        shape : tuple
            the shape of the image
        learning_rate: float
            the learning rate in the current iteration
        beta1: float
            decay rate for calculating the exponentially
            decaying average of past gradients
        beta2: float
            decay rate for calculating the exponentially
            decaying average of past squared gradients
        epsilon: float
            small value to avoid division by zero
        """

        self.m = torch.zeros(shape).cuda()
        self.v = torch.zeros(shape).cuda()
        self.t = 0

        self._beta1 = beta1
        self._beta2 = beta2
        self._learning_rate = learning_rate
        self._epsilon = epsilon

    def __call__(self, gradient):
        """Updates internal parameters of the optimizer and returns
        the change that should be applied to the variable.
        Parameters
        ----------
        gradient : `np.ndarray`
            the gradient of the loss w.r.t. to the variable
        """

        self.t += 1

        self.m = self._beta1 * self.m + (1 - self._beta1) * gradient
        self.v = self._beta2 * self.v + (1 - self._beta2) * gradient**2
        #TODO: remove print
        print(f"self.m: min: {torch.min(self.m)}, max: {torch.max(self.m)}")
        print(f"self.v: min: {torch.min(self.v)}, max: {torch.max(self.v)}")
        
        bias_correction_1 = 1 - self._beta1**self.t
        bias_correction_2 = 1 - self._beta2**self.t

        #TODO:
        print(f"bias_correction_1: {bias_correction_1}")
        print(f"bias_correction_2: {bias_correction_2}")

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2
        #TODO: remove print
        print(f"m_hat: min: {torch.min(m_hat)}, max: {torch.max(m_hat)}")
        print(f"v_hat: min: {torch.min(v_hat)}, max: {torch.max(v_hat)}")

        return -self._learning_rate * m_hat / (torch.sqrt(v_hat) + self._epsilon)


