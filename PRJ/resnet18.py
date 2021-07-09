import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class Net(nn.Module):
    def __init__(self,param):
        super().__init__()
        # Paremeters for different input sizes based on the dataset
        width = param[0]
        height = param[1]
        channel_num = param[2]

        self.model = models.resnet18(True)     
        self.model.conv1 = nn.Conv2d(channel_num, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Test variable to get the ouput size of the hidden layers
        x = torch.randn(width*3,height*3).view(-1,channel_num,width,height) 

        self._to_linear = None
        self.convs(x) 
        self.fc1 = nn.Linear(self._to_linear,10)
    def convs(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    def forward(self, x):
        x = self.convs(x)   
        # The output from the hidden layers is flattened here 
        x = x.view(-1, self._to_linear)
        x = self.fc1(x)
        return x