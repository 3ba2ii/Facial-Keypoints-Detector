# Facial Keypoints Detection

This project will be all about defining and training a convolutional neural network to perform facial keypoint detection, and using computer vision techniques to transform images of faces.

<br />


![68 Facial Keypoints](https://miro.medium.com/max/1200/1*a6kXOpZQ4abIk0EfIkKOpw.jpeg)


<br />

## Description 

This model takes an input image and make transformations  to it like converting it to grayscale image , reclaing it to be (224x224) and Normalizing its values to be in range [0,1].

Then the image is fed to the network to predict the output facial keypoints of the input image with ````error ~ 2%````.


The output would be in shape ```(136,1)``` then we reshape it to be ```(68,2)``` which means a pair of values for each keypoints (x,y) that identifies the facial keypoints.

And finally we plot the output points ```(x,y)``` on the original input image to show the facial keypoints.

<br />


## What To Improve 

We should be able to get less ```error < 2%``` 

### Methods to decrease the error :
  
    1. Try splitting the given test set into Validation set and Test set in order to
        get better results.

    2. Try training the model for more epochs > 10 epochs

    3. Try adding more Convolutional layers and make your model more complex.

    4. Try getting a larger data set.


With these methods we can get error that is close to ```error ~ 0.5%```

<br />

## Installation


This project uses opncv library [opencv](https://pypi.org/project/opencv-python/) and [PyTorch](https://pytorch.org/docs/stable/index.html) to install these libraries.

#### Install OpenCv :
```bash
pip install opencv-python
```
#### Install PyTorch :
```bash
pip3 install torch torchvision

```
<br />

## Network Architecture 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

def ___init___(self):
# 5 Convolutional Layers 
    self.conv1 = nn.Conv2d(1, 32, 5)
    self.conv2 = nn.Conv2d(32, 64, 3)
    self.conv3 = nn.Conv2d(64, 128, 3)
    self.conv4 = nn.Conv2d(128, 256, 3)
    self.conv5 = nn.Conv2d(256, 512, 1)

# MaxPooling Layer to decrease the width and height to half the input layer

    self.pool = nn.MaxPool2d(2, 2)

# Two Fully-Connected Layers 

    self.fc1 = nn.Linear(512 * 6 * 6  , 1024)
    self.fc2 = nn.Linear(1024,136)


# A Dropout Layer to eliminate overfitting

    self.drop1 = nn.Dropout(p = 0.1)
    self.drop2 = nn.Dropout(p = 0.2)
    self.drop3 = nn.Dropout(p = 0.25)
    self.drop4 = nn.Dropout(p = 0.3)
    self.drop5 = nn.Dropout(p = 0.4)
        
```
<br />

## Forward Pass Technique 

```python

def forward(self, x):
  
        ## x is the input image (grayscale image with
        ##dimensions (224x224)

        ## 1. Pass the input image (224x224) to the a 
        ##    Convolutional Layer + ReLU + MaxPooling Layer
        ## 2. Add dropout layer after each step 

      
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop3(x)
        
        x = self.pool(F.relu(self.conv5(x)))
        x = self.drop4(x)

        ## 3. Flatten the output of the last layers 
        ## to be fed into the Fully-Connected Layer

        x = x.view(x.size(0), -1)

        ## 4. Fed the output to the fully-connected layer and add 
        ## drouout.
        
        x = F.relu(self.fc1(x))
        
        x = self.drop5(x)
        
        ## 5. output of the netword would be (136,1)
        ## Which is pairs of (x,y) that defines the 
        ## desired  Facial points 
   
        x = self.fc2(x)


        # a modified x, having gone through all 
        #the layers of your model, should be returned


        return x
        
```
<br />

## Optimizer and Loss Function Used 

```python 
import torch.optim as optim

criterion = nn.SmoothL1Loss()

optimizer = optim.Adam(net.parameters(), lr = 0.001)
```

<br />


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


