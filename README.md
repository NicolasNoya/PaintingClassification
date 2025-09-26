# Painting Classification

The main objective of this project is to determine whether a given painting is abstract or not. Multiple approaches are explored for this classification task. The first approach uses a standard supervised classification pipeline.


## Approaches

- **ResNet Baseline**:  
  We used a pretrained ResNet model as a strong baseline. Fine-tuning on our dataset showed that the model was able to distinguish between abstract and figurative paintings.

- **Dual-Architecture (Global + Patch)**:  
  A modified version of ResNet was developed to incorporate two complementary views of the same painting:  
  - **Global View**: The whole image is processed to capture overall composition, style, and texture.  
  - **Patch View**: A cropped region of the painting is processed separately to focus on localized patterns and finer details.  
  The features from both branches are then combined for classification, allowing the model to balance holistic understanding with localized evidence.


## Findings

- The **ResNet baseline** achieved solid results largely because figurative paintings often contain recognizable objects (faces, animals, everyday items). ResNet, pretrained on ImageNet, leveraged its object-detection ability to separate figurative from abstract works.   
- A key observation is that models pretrained on object-centric datasets like ImageNet may perform well on figurative art due to transfer learning but risk overfitting to unintended cues rather than true abstractness.


## Dependency Management

This project uses **Poetry** for dependency management. All required libraries are included, except for **PyTorch**, since different versions may be needed depending on the target device. Please install the appropriate version of PyTorch for your environment from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).


## Project Structure

To ensure modularity and atomicity, the project follows this folder and class structure:

- **Metrics Management**: A class is responsible for computing and storing evaluation metrics.
- **Interfaces**: A class handles training, validation, and testing interfaces for every model created.
- **Data Management**: A class is responsible for data handling, including data loading, augmentation, and transformations.
- **Models**: A folder containing all model definitions.
- **Profiling and Analysis**: A class used for profiling during training and for data analysis.
- **Research**: A folder dedicated to experimental or preliminary code. This is not intended for production use and is generally not recommended for reference.


## Future Work

- Extend experimentation with transformer-based vision models (e.g., ViT, Swin Transformer) for style-aware classification.  
- Incorporate self-supervised or contrastive learning to better capture style similarity without relying solely on object detection.  
- Explore interpretability methods (Grad-CAM, feature visualization) to understand which regions the models rely on when classifying abstract vs. figurative paintings.  
- Evaluate on additional datasets of paintings to test generalization beyond MNIST-like binary abstraction/figurative splits.  



