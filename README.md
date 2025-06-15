# Painting Classification

The main objective of this project is to determine whether a given painting is abstract or not. Multiple approaches will be explored for this classification task. The first approach will use a standard supervised classification pipeline.

### Dependency Management

This project uses Poetry for dependency management. All required libraries will be included, except for PyTorch, since different versions might be required depending on the target device. Please install the appropriate version of PyTorch for your environment from https://pytorch.org/get-started/locally/.

### Project Structure

To ensure modularity and atomicity, the project follows this folder and class structure:

    Metrics Management: A class is responsible for computing and storing evaluation metrics.

    Interfaces: A class handles training, validation, and testing interfaces for every model created.

    Data Management: A class is responsible for data handling, including data loading, augmentation, and transformations.

    Models: A folder containing all model definitions.

    Profiling and Analysis: A class used for profiling during training and for data analysis.

    Research: A folder dedicated to experimental or preliminary code. This is not intended for production use and is generally not recommended for reference.