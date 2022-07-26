import torch
from torch.optim.optimizer import Optimizer

class SuperKekW(Optimizer):
    ''' Better version of Adamw - SuperKekW. Watch Spy X Family ''' 
    def __init__(self, parameters, alpha = 0.002, beta1 = 0.9, beta2 = 0.998, eps = 1e-8, lr = 5e-3, l1 = 0.01): 
  
      # Initializes meta-parameters of SupeKekW
      self.alpha = alpha 
      self.beta1 = beta1 
      self.beta2 = beta2 
      self.eps = eps 
      self.lr = lr
      self.l1 = l1 
      self.currB1 = self.beta1
      self.currB2 = self.beta2
      
      # Converts generator into list
      self.parameters = list(parameters)

      # Initializes dictionary to store last values of parameters: { parameter : last_value }
      self.last_gradient = {} 
      self.last_momentum = {} 
      self.last_nu = {} 

      # Initializes last values of parameters as tensor filled with zeros
      for param in self.parameters: 
          self.last_gradient[param] = torch.zeros_like(param) 
          self.last_momentum[param] = torch.zeros_like(param) 
          self.last_nu[param] = torch.zeros_like(param) 
      
      # Calls the __init__ of torch.optim.Optimizer
      defaults = dict(lr=lr, betas=(beta1, beta2), eps=eps, 
                        weight_decay=l1, amsgrad=False) 
      super().__init__(self.parameters, defaults) 
 

    def step(self): 
        ''' Perfoms a single optimization step '''

        for param in self.parameters: 

            # If there is no gradient
            if param.grad is None: 
                continue 

            # Gets the gradient from model 
            gradient = param.grad + param

            # Calculates additional information
            m = self.beta1 * self.last_momentum[param] + (1 - self.beta1) * gradient  
            nu = self.beta2 * self.last_nu[param] + (1 - self.beta2) * gradient ** 2 
            mt = (m) / (1 - self.currB1) 
            nut = (nu) / (1 - self.currB2)
            self.currB1 *= self.beta1
            self.currB2 *= self.beta2

            # Calculates deltas and updates weights of model 
            current_w = param.data 
            delta_w = self.lr * ((self.alpha * mt) / ((nut) ** 0.5 + self.eps) + self.l1 * current_w)
            param.data.add_(-delta_w) 
            
            # Updates the last values of parameter
            self.last_gradient[param] = gradient 
            self.last_momentum[param] = m
            self.last_nu[param] = nu

