#%%
import torch
import torchvision
from torchvision.models import resnet50

class TwoBranchRNN(torch.nn.Module):
    """
    This class implements a two-branch RNN model, the first branch
    will have as input the whole image and the second branch
    will have as input the image patches.
    The model is based on a ResNet50 backbone for feature extraction and 
    will have a Fully Connected layer at the end to output the final prediction.
    
    Args:
        freeze_layers (int): Proportion of layers to freeze in the ResNet50 model.
    """
    def __init__(self, freeze_layers:float=0.8):
        super(TwoBranchRNN, self).__init__()
        if not (0 < freeze_layers < 1):
            raise ValueError("freeze_layers must be a float between 0 and 1.")
        
        # Load the ResNet50 model
        self.full_image_resnet = resnet50(pretrained=True)
        self.patch_resnet = resnet50(pretrained=True)
        
        # Freeze the specified number of layers
        self.len_layers = len(list(self.full_image_resnet.parameters()))
        self.freeze_layers = int(self.len_layers * freeze_layers)
        
        # Replace the final fully connected layer with an identity layer
        self.full_image_resnet.fc = torch.nn.Identity()
        self.patch_resnet.fc = torch.nn.Identity()

        for param in list(self.full_image_resnet.parameters())[:self.freeze_layers]:
            param.requires_grad = False


        for param in list(self.patch_resnet.parameters())[:self.freeze_layers]:
            param.requires_grad = False

        for param in list(self.patch_resnet.fc.parameters())[self.freeze_layers:]:
            param.requires_grad = True
 
        for param in list(self.full_image_resnet.fc.parameters())[self.freeze_layers:]:
            param.requires_grad = True

        # Define a Linear layer for the final output
        self.fc = torch.nn.Linear(2048 * 2, 2) 

    def forward(self, full_image, patch):
        full_image_features = self.full_image_resnet(full_image)
        patch_features = self.patch_resnet(patch)
        combined_features = torch.cat((full_image_features, patch_features), dim=1)
        output = self.fc(combined_features)
        return output
    
    



if __name__ == "__main__":
    # Example usage
    model = TwoBranchRNN(freeze_layers=0.8)
    
    # Create dummy inputs
    full_image = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    patch = torch.randn(1, 3, 224, 224)       # Same shape for the patch


    # Forward pass
    output = model(full_image, patch)
    print(output.shape)  # Should be [1, 2] for binary classification
    print(output)