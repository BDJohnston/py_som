import math
import numpy as np
import matplotlib.pyplot as plt

class SOM():
  '''
  A numpy implementation of a Self Oranizing Map (SOM).
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
  Attribute: rand:      numpy random number generator
  Attribute: weights:   numpy ndarray of shape [height, width, n_inputs] containing SOM weights
  Attribute: locations: numpy ndarray of shape [height, width, 2] containing SOM node indexes
  '''

  def __init__(self, n_inputs=3, width=10, height=10, alpha_0=0.1, max_iter=500, rand=None):
    '''
    Constructor for SOM class
    Param: n_inputs: size of input SOM input vector, V, default = 3
    Param: width:    width of SOM, default = 10
    Param: height:   height of SOM, default = 10
    Param: alpha_0:  initial learning rate, default = 0.1
    Param: max_iter: maximum number of iterations/epochs in training loop, default = 500
    Param: rand:     numpy random number generator
    :return:
    '''

    self.width = width
    self.height = height
    self.n_inputs = n_inputs
    self.alpha_0 = alpha_0
    self.alpha = self.alpha_0
    # self.sigma_0 = max(SOM.shape[0], SOM.shape[1])/2
    self.sigma_0 = width * height / 2.0
    # self.sigma_0 = width**2 + height**2
    self.sigma = self.sigma_0
    self.max_iter = max_iter
    self.lambda_0 = self.max_iter / math.log(self.sigma_0)
    if not rand:
      self.rand = np.random.RandomState(0)
    else:
      self.rand = rand
    self.weights = self.rand.random((self.height, self.width, self.n_inputs))
    self.locations = np.array([np.array([np.array([i, j], dtype=np.float32) for j in range(self.weights.shape[1])]) for i in range(self.weights.shape[0])])
  
  def forward(self, input):
    '''
    Forward Pass (torch style) of the Self Oranizing Map (SOM).
    Makes use of broadcasting for batch operations.
    Param: input: numpy array of shape [batch size, input vector (V) size]
    :return: numpy array of shape [batch size, 2] containing x, y
             co-ords for the bmu for each input in the batch
    '''
    
    distSq = (np.square(self.weights - input[:,None,None,:])).sum(axis=3).reshape((input.shape[0], -1))
    return np.column_stack(np.unravel_index(np.argmin(distSq, axis=1), self.weights.shape[:2])).astype(np.float32)

  def update(self, input):
    '''
    batch weight update of the Self Oranizing Map (SOM).
    Makes use of broadcasting for batch operations.
    Param: input: numpy array of shape [batch size, input vector (V) size]
    :return:
    '''

    xy_z = self.forward(input)
    dist_sq = np.sum(np.square(self.locations-xy_z[:,None,None,:]), axis=3)
    dist_fn = self.alpha * np.exp(-dist_sq / (2 * self.sigma**2))
    self.weights += (dist_fn[:, :, :, None] * (input[:,None, None,:] - self.weights)).mean(axis=0)
  
  def fit(self, input, n_iter, fit_rand=False):
    '''
    sklearn style fit method: trains SOM on batch of input data.
    Param: input: numpy array of shape [batch size, input vector (V) size]
    Param: n_iter: number of epochs or steps in training loop
    Param: fit_rand: boolean, if true, will set training batch to tensor with same
                     shape as input but filled with new random values between
                     0.0 and 1.0 every epoch
    :return:  loss: value of average loss for the final training data batch bmu's
    '''
    
    for t in np.arange(0, n_iter):
        if fit_rand:
          input = self.rand.random((20,3))
        # self.rand.shuffle(input)      
        self.update(input)
        # Update learning rate and radius
        self.alpha = self.alpha_0 * np.exp(-t / self.lambda_0)
        self.sigma = self.sigma_0 * np.exp(-t / self.lambda_0)   
  
  def view_weights(self):
    '''
    Visualizes the weights of the Self Oranizing Map (SOM)
    :return:
    '''

    image = self.weights.reshape(self.height, self.width, 3)
    plt.imshow(image)
    plt.show()