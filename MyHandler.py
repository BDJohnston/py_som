import logging
import torch
import torch.nn.functional as F
import io
import numpy as np
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

class MyHandler(BaseHandler):
    """
    Custom handler for pytorch serve. This handler supports batch requests.
    For a deep description of all method check out the doc:
    https://pytorch.org/serve/custom_service.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor())
        ])

    def preprocess_one_batch(self, req):
        """
        Process one single image.
        """
        # get batch from the request
        batch = req.get("data")
        if batch is None:
            batch = req.get("body")       
         # create a stream from the encoded image
        batch = np.array(batch)
        batch = self.transform(batch)
        # add batch dim
        # batch = batch.unsqueeze(0)
        return batch

    def preprocess(self, requests):
        """
        Process all the images from the requests and batch them in a Tensor.
        """
        batch = [self.preprocess_one_batch(req) for req in requests]
        batch = torch.cat(batch)    
        return batch

    def inference(self, input):
        """
        Given the data from .preprocess, perform inference using the model.
        We return the predicted label for each image.
        """
        bmus = self.model.forward(input)
        return bmus

    def postprocess(self, bmus):
        """
        Given the data from .inference, postprocess the output.
        In our case, we get the human readable label from the mapping 
        file and return a json. Keep in mind that the reply must always
        be an array since we are returning a batch of responses.
        """
        res = []
        # preds has size [BATCH_SIZE, 1]
        # convert it to list
        bmus = bmus.cpu().tolist()
        for bmu in bmus:
            res.append({'bmu' : bmu})
        return res