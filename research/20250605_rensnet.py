#%%
import torch
import torchvision
from torchvision.models import resnet50


full_image_resnet = resnet50(pretrained=True)
# print how many layers are in the model
num_layers = sum(1 for _ in full_image_resnet.named_children())
print(f"Number of layers in the ResNet50 model: {num_layers}")

# Print the layers of the ResNet50 model
count = 0
for name, layer in full_image_resnet.named_children():
    print(f"{name}: {layer}")
    count += 1
    print(count)
#%%
full_image_resnet.fc = torch.nn.Identity()
#%%
count = 0
for param in full_image_resnet.parameters():
    param.requires_grad = False
    print(f"Parameter {count}: {param}")
    count+=1
    print(count)
#%%
import torch

a = torch.tensor([1, 2, 3, 4])  # shape (2, 2)
b = torch.tensor([5, 6, 7, 8])  # shape (2, 2)

# Concatenate along rows (dim=0)
cat0 = torch.cat((a, b), dim=0)  # shape (4, 2)

# Concatenate along columns (dim=1)
# cat1 = torch.cat((a, b), dim=1)  # shape (2, 4)

#%%
print(cat0)
# print(cat1)
