import numpy as np
import timeit
from models.kohenen_numpy import SOM

def run_colour_test(vis=False):
  m = 10
  n = 10
  # Number of training examples
  n_samples = 20 #3000
  n_inputs = 3
  n_iter = 500
  rand = np.random.RandomState(0)

  # Initialize the training data
  # train_data = rand.uniform(0.0, 1.0, (n_samples, n_inputs))
  train_data = rand.random((n_samples,n_inputs))

  som = SOM(n_inputs, width=m, height=n, alpha_0=1.0, max_iter=n_iter, rand=rand)
  som.fit(train_data, n_iter, fit_rand=False)
  
  if vis:
    som.view_weights()

if __name__ == '__main__':
  print(timeit.timeit(stmt=run_colour_test, number=10)/10)