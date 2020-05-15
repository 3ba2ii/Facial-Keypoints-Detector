# Facial Keypoints Detection

This project is all about defining and training a convolutional neural network to perform facial keypoint detection, and using computer vision techniques to transform images of faces.

<br />


![68 Facial Keypoints](https://miro.medium.com/max/1200/1*a6kXOpZQ4abIk0EfIkKOpw.jpeg)


<br />

## Description 

This model takes an input image and makes transformations  to it like converting it to grayscale image , reclaing it to be (224x224) and Normalizing its values to be in range [0,1].

Then the image is fed to the network to predict the output facial keypoints of the input image with ~~````error ~ 2%````~~      ```error ~ 0.4% ```.


The output would be in shape ```(136,1)``` then we reshape it to be ```(68,2)``` which means a pair of values for each keypoints (x,y) that identifies the facial keypoints.

And finally we plot the output points ```(x,y)``` on the original input image to show the facial keypoints.

<br />


## What To Improve 

We should be able to get less ```error < 2%``` 


##### Methods to decrease the error :

  
 - [x] ~~Try splitting the given test set into Validation set and Test set in order to get better results.~~
 - [x] ~~Try training the model for more epochs > 10 epochs~~ 
 - [x] ~~Try EarlyStopping to protect the model from overfitting the data~~
 - [x] ~~Try different kinds of pretrained networks like AlexNet , ResNet , etc..~~
 - [ ] ~~Try adding more Convolutional layers and make your model more complex.~~ ```Not Needed```
 - [ ] ~~Try getting a larger data set.~~    ```Not Needed```
 

With these methods we can get error that is close to ```error ~ 0.5%``` ✅

![68 Decrease Error](https://i.ibb.co/DfRTcdM/Screen-Shot-2020-05-13-at-4-11-33-PM.png)

Error now has decreased to ``` ~ 0.004  ``` on both **Training** and **Validation** Sets


<br />

### Prerequisites

This project [opencv](https://pypi.org/project/opencv-python/) and [PyTorch](https://pytorch.org/docs/stable/index.html) to install these libraries.

##### Install OpenCv :
```bash
pip install opencv-python
```
##### Install PyTorch :
```bash
pip3 install torch torchvision
```
<br />

## Network Architecture 

Used [```ResNet```](https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624) instead of the previous architecture 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        # change from supporting color to gray scale images
        self.resnet18.conv1 = 
                            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=False)
        n_inputs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(n_inputs, 136)
                        
    def forward(self, x):
        x = self.resnet18(x)
        return x
```
<br />

## Optimizer and Loss Function Used 

```python 

import torch.optim as optim

criterion = nn.SmoothL1Loss().cuda if device == 'cuda' else nn.SmoothL1Loss()

#To Turn on the gradients after being disabled in the Network Architecture 

optimizer = optim.Adam(filter(lambda p: p.requires_grad,net.parameters()), lr = 0.001)


```


## Authors

- **Ahmed Abd-Elbakey Ghonem** - [**Github**](https://github.com/3ba2ii)


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## Acknowledgments

* Hat tip to [@stefanonardo](https://github.com/stefanonardo) whose EarlyStopping class code was used 


