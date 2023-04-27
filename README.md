# DL-Activity
Image Segmentation using CNN on Breast Cancer Dataset

Breast cancer image segmentation using CNN is a popular application of deep learning. Image segmentation is the process of dividing an image into different parts or segments, each of which represents a meaningful object or region in the image. In the case of breast cancer, image segmentation can be used to identify and locate malignant regions in mammogram images.
Here are the steps you can follow to perform image segmentation on a breast cancer dataset using CNN:
Data Preprocessing: The first step is to preprocess the dataset by normalizing the pixel values, resizing the images to a uniform size, and splitting the data into training and validation sets.
Build the CNN Model: Next, you need to build a convolutional neural network (CNN) model that is capable of learning from the breast cancer dataset. A typical CNN model for image segmentation consists of several convolutional layers, followed by pooling layers, and finally a fully connected layer.
Train the CNN Model: Once the CNN model is built, you need to train it on the breast cancer dataset. During training, the CNN model learns to identify the malignant regions in the mammogram images by adjusting the weights of the network.
Evaluate the Model: After training, you need to evaluate the performance of the CNN model on the validation set. This will help you determine if the model is overfitting or underfitting and adjust the hyperparameters accordingly.
Test the Model: Finally, you need to test the CNN model on a test set of breast cancer images to see how well it can identify malignant regions in unseen images.
Here are some additional tips for building a CNN model for breast cancer image segmentation:
Use a deep CNN architecture such as U-Net or SegNet, which are specifically designed for image segmentation tasks.
Use data augmentation techniques such as image flipping, rotation, and scaling to increase the size of the dataset and improve the robustness of the model.
Use an appropriate loss function such as binary cross-entropy or Dice loss, which is specifically designed for image segmentation tasks.
Use transfer learning techniques to leverage the pre-trained CNN models such as VGG16 or ResNet50 that are trained on a large dataset such as ImageNet.
Overall, breast cancer image segmentation using CNN is a promising application of deep learning that has the potential to improve the accuracy and efficiency of breast cancer diagnosis and treatment.

What is CNN ?
CNN is a powerful algorithm for image processing. These algorithms are currently the best algorithms we have for the automated processing of images. Many companies use these algorithms to do things like identifying the objects in an image.

Three Layers of CNN:
Convolutional Neural Networks specialized for applications in image & video recognition. CNN is mainly used in image analysis tasks like Image recognition, Object detection & Segmentation.
There are three types of layers in Convolutional Neural Networks:

1) Convolutional Layer:
In a typical neural network each input neuron is connected to the next hidden layer. In CNN, only a small region of the input layer neurons connect to the neuron hidden layer.

2) Pooling Layer:
The pooling layer is used to reduce the dimensionality of the feature map. There will be multiple activation & pooling layers inside the hidden layer of the CNN.

3) Fully-Connected layer:
Fully Connected Layers form the last few layers in the network. The input to the fully connected layer is the output from the final Pooling or Convolutional Layer, which is flattened and then fed into the fully connected layer.
