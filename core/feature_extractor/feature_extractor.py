import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

import cv2

from utils.helpers import print_info

print_info('Loading ResNet Model...')
# Load the pretrained model
model = models.resnet18(pretrained=True)
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')
# Set model to evaluation mode
model.eval()
print_info('Model Loaded!')


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_cosine_similarity(input_image, to_match_image):
    '''
    Returns the cosine similarity score between the two images.
    '''
    
    input_vector = _get_vector(input_image)
    to_match_vector = _get_vector(to_match_image)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_score = cos(input_vector.unsqueeze(0), to_match_vector.unsqueeze(0))

    return cosine_score.numpy()[0]


def _get_vector(image):
    '''
    Returns the vector embeddings for an image.
    '''

    image = cv2.resize(image, (224, 224))
    tensor_img = Variable(normalize(to_tensor(image)).unsqueeze(0))
    embedding = torch.zeros(512)
    def copy_data(m, i, o):
        embedding.copy_(o.data.squeeze())
    h = layer.register_forward_hook(copy_data)
    model(tensor_img)
    h.remove()
    return embedding
