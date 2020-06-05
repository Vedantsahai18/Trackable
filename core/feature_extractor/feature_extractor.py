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
    
    input_vector = _get_vector(input_image)

    to_match_vector = _get_vector(to_match_image)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_score = cos(input_vector.unsqueeze(0), to_match_vector.unsqueeze(0))

    return cosine_score.numpy()[0]


def _get_vector(image):

    image = cv2.resize(image, (224, 224))
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(image)).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.squeeze())
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding
