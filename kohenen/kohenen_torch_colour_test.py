import torch
import numpy as np
import timeit
from models.kohenen_torch import SOM
import os

def run_colour_test(view=False, save_path=None):
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  m = 10
  n = 10
  # Number of training examples
  n_samples = 20 #3000
  n_inputs = 3
  n_iter = 500

  # Initialize the training data
  rand = np.random.RandomState(0)
  input_data = rand.random((n_samples,n_inputs))
  train_data = torch.tensor(input_data).float().to(device) 
  torch.manual_seed(0)
  # train_data = torch.rand(n_samples, 3, dtype=torch.float).to(device)

  som = SOM(n_inputs=n_inputs, width=m, height=n, alpha_0=1.0, max_iter=n_iter).to(device)
  som.fit(train_data, n_iter=n_iter, fit_rand=False)
  
  if view:
    som.view_weights()
  
  if save_path:
    torch.save(som.state_dict(), save_path)

if __name__ == '__main__':
  PATH = os.environ['MODEL_PATH']+'kohenen_colour_test.pt'
  print(timeit.timeit(stmt=run_colour_test, number=10)/10)
  run_colour_test(save_path=PATH)