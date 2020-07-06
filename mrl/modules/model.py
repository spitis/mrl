import mrl
import torch
from typing import Callable
import os
import pickle
import dill

class PytorchModel(mrl.Module):
  """
  Generic wrapper for a pytorch nn.Module (e.g., the actorcritic network).
  These live outside of the learning algorithm modules so that they can easily be 
  shared by different modules (e.g., critic can be used by intrinsic curiosity module). 
  They are also saved independently of the agent module (which is stateless). 
  """

  def __init__(self, name : str, model_fn : Callable):
    super().__init__(name, required_agent_modules=[], locals=locals())
    self.model_fn = model_fn
    self.model = self.model_fn()

  def _setup(self):
    if self.config.get('device'):
      self.model = self.model.to(self.config.device)

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    torch.save(self.model.state_dict(), path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    self.model.load_state_dict(torch.load(path), strict=False)

  def copy(self, new_name):
    """Makes a copy of the Model; e.g., for target networks"""
    new_model = dill.loads(dill.dumps(self.model))
    model_fn = lambda: new_model
    return self.__class__(new_name, model_fn)

  def __call__(self, *args, **kwargs):
    if self.training:
      self.model.train()
    else:
      self.model.eval()
    return self.model(*args, **kwargs)