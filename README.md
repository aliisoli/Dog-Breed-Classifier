# dog-breed-classifier
Using transfer learning with CNNs to develop a model that identifies the breed of dogs


### Summary
In this project, we aim at developing a convolutional neural network in keras which is able to classify the breed of a dog into one of 133 categories.
When given a picture of a human, the model acknowledges the human and returns the most resembling dog breed to the person.
We use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images. OpenCV provides many pre-trained face detectors, stored as XML files on github.
we use a pre-trained ResNet-50 model to detect dogs in images. we download the ResNet-50 model, along with weights that have been trained on ImageNet, a very large, very popular dataset used for image classification and other vision tasks. 
ImageNet contains over 10 million URLs, each linking to an image containing an object from one of 1000 categories.
Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.
In the first run, we try to fit a CNN on the data set from scratch. Even on a cluster of GPUs, each epoch takes roughly a minute to finish.
After 5 epochs, the CNN has an accuracy of only 8%. Since we are not planning ro re-invent the wheel, we use transfer learning.
For transfer learning, I used pretrained features on Xception network. I created CNN using Xception bottleneck features and changed the architecture until ~85% accuracy was achieved.
In the end, a few pictures of humans and dogs were given to the model and the performance was tested. 

### Installation
This project requires Python 3.x and the following Python libraries installed: 
- keras
- Tensorflow
- sklearn
- Numpy
- Matplotlib

You will also need to have software installed to run and execute an iPython Notebook

### Files
dog_app.ipynb: jupyter notebook where all the steps of the project are explained
model.h5, model.json : saved keras CNN for future use

### License
Feel free to download and use the code or the model. Let me know if you were able to achieve more than 90% accuracy!
