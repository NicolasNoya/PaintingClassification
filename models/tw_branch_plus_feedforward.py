#%%
import torch
import torchvision
from torchvision.models import resnet50
from models.two_branch_rnn import TwoBranchRNN

class TwoBranchRNNTwoLayersFF(torch.nn.Module):
    """
    This class implements a two-branch RNN model, the first branch
    will have as input the whole image and the second branch
    will have as input the image patches.
    The model is based on a ResNet50 backbone for feature extraction and 
    will have a two layer Fully Connected at the end to output the final prediction.
    
    Args:
        freeze_layers (int): Proportion of layers to freeze in the ResNet50 model.
    """
    def __init__(self, freeze_layers:float=0.8):
        super(TwoBranchRNNTwoLayersFF, self).__init__()
        self.model = TwoBranchRNN(freeze_layers)
        self.model.fc = torch.nn.Linear(2048*2, int(2048*3/2))
        self.fc1 = torch.nn.Linear(int(2048*3/2), 2)
        self.relu = torch.nn.ReLU()

        

    def forward(self, full_image, patch):
        embeddings = self.model(full_image, patch)
        # embeddings = self.embedding(combined_features)
        output = self.fc1(self.relu(embeddings))
        return embeddings, output
    
    def get_embeddings(self, full_image, patch):
        return self.model(full_image, patch)
    
    



if __name__ == "__main__":
    # Example usage
    model = TwoBranchRNNTwoLayersFF(freeze_layers=0.8)
    
    # Create dummy inputs
    full_image = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    patch = torch.randn(1, 3, 224, 224)       # Same shape for the patch


    # Forward pass
    output = model(full_image, patch)
    print(output.shape)  # Should be [1, 2] for binary classification
    print(output)
    embedding = model.get_embeddings(full_image, patch)
    print(embedding.shape)
    