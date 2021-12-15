import torch  
import math
import matplotlib.pyplot as plt
import numpy as np

import torch  
import math
import matplotlib.pyplot as plt
import numpy as np

class Torch_SOM(torch.nn.Module):
  '''
  A Torch implementation of a Self Oranizing Map (SOM).
  Method: forward(input):               Forward pass of SOM. returns bmus for input batch
  Method: update(input):                Applies single update to SOM weights using input batch
  Method: fit(input, n_iter, fit_rand): trains SOM weights on input 
  Attribute: width:     width of SOM
  Attribute: height:    height of SOM
  Attribute: n_inputs:  size of input SOM input vector, V
  Attribute: alpha_0:   initial learning rate
  Attribute: alpha:     current learning rate for training loop
  Attribute: sigma_0:   initial radius
  Attribute: sigma:     current radius for training loop
  Attribute: max_iter:  maximum number of iterations/epochs in training loop
  Attribute: lambda_0:  decay time constant
  Attribute: weights:   torch tensor of shape [height * width, n_inputs] containing SOM weights
  Attribute: locations: torch tensor of shape [height, width, 2] containing SOM node indexes
  '''

  def __init__(self, n_inputs=3, width=10, height=10, alpha_0=0.1, max_iter=500):
    '''
    Constructor for SOM class
    Param: n_inputs: size of input SOM input vector, V, default = 3
    Param: width:    width of SOM, default = 10
    Param: height:   height of SOM, default = 10
    Param: alpha_0:  initial learning rate, default = 0.1
    Param: max_iter: maximum number of iterations/epochs in training loop, default = 500
    :return:
    '''

    super(Torch_SOM, self).__init__()

    self.width = width
    self.height = height
    self.n_inputs = n_inputs
    self.alpha_0 = alpha_0
    # self.sigma_0 = max(width, height) / 2
    self.sigma_0 = width * height / 2.0
    # self.sigma_0 = width**2 + height**2
    self.max_iter = max_iter
    self.lambda_0 = self.max_iter / math.log(self.sigma_0)
    self.weights = torch.nn.Parameter(torch.rand(self.height * self.width, self.n_inputs), requires_grad=False)
    locations = [np.array([i, j]) for i in range(self.height) for j in range(self.width)]
    self.locations = torch.nn.Parameter(torch.tensor(np.array(locations), dtype=float), requires_grad=False)
    self.pdist_fn = torch.nn.PairwiseDistance(p=2)

  def forward(self, input):
    '''
    Forward Pass (torch style) of the Self Oranizing Map(SOM).
    Makes use of broadcasting for batch operations.
    Param: input: tensor of shape [batch size, input vector (V) size]
    :return:  bmu_location: tensor of shape [batch size, 2] containing x, y
                            co-ords for the bmu for each input in the batch
              losses:       tensor of shape [batch size] containing euclidean distances
                            for the bmu of each input in batch
    '''

    b_size = input.size()[0]
    if len(input.size()) == 1:
      input = input.unsqueeze(0)
    else:
      input = input.unsqueeze(1)
    distances = self.pdist_fn(input, self.weights)
    losses, bmu_idxs = torch.min(distances, -1)
    bmu_locations = self.locations[bmu_idxs]
    return bmu_locations, losses

  def update(self, input, t):
    '''
    batch weight update of the Self Oranizing Map(SOM).
    Makes use of broadcasting for batch operations.
    Param: input: tensor of shape [batch size, input vector (V) size]
    Param: t: epoch or step in training loop
    :return:  loss: value of average loss for the input batch bmu's
    '''

    b_size = input.size()[0]
    bmu_locations, losses = self.forward(input)
    if len(input.size()) == 1:
      input = input.unsqueeze(0)
    else:
      input = input.unsqueeze(1)
    loss = losses.sum().div_(b_size).item() 
    bmu_distance_squares = (self.locations.float() - bmu_locations.unsqueeze(1).float()).pow_(2).sum(dim=2)
    sigma_t = self.sigma_0 * math.exp(-t / self.lambda_0)
    alpha_t = self.alpha_0 * math.exp(-t / self.lambda_0)
    theta_t = bmu_distance_squares.neg_().div_(2 * sigma_t**2 + 1e-5).exp_()
    delta = theta_t.mul_(alpha_t).unsqueeze(2) * (input - self.weights)
    self.weights.add_(delta.mean(dim=0))
    return loss

  def fit(self, input, n_iter=None, fit_rand=False, device='cpu'):
    '''
    sklearn style fit method: trains SOM on batch of input data.
    Param: input: tensor of shape [batch size, input vector (V) size]
    Param: n_iter: number of epochs or steps in training loop
    Param: fit_rand: boolean, if true, will set training batch to tensor with same
                     shape as input but filled with new random values between
                     0.0 and 1.0 every epoch
    Param: device: device torch tensor is stored on (cpu or gpu)
    :return:  loss: value of average loss for the final training data batch bmu's
    '''

    n_samples = input.shape[0]
    if self.alpha_0 is None:
      self.alpha_0 = float((self.width * self.height) / n_samples)
    if not n_iter: 
      n_iter = self.max_iter
    for t in range(n_iter):
      if fit_rand:
        input = torch.rand(n_samples, self.n_inputs, dtype=torch.float).to(device)
      loss = self.update(input, t)

    return loss

  def view_weights(self):
    '''
    Visualizes the weights of the Self Oranizing Map(SOM)
    :return:
    '''
    image = self.weights.reshape(self.height, self.width, 3).cpu().numpy()
    plt.imshow(image)
    plt.show()