import torch.nn as nn
import torch
from torch import sigmoid, cat, relu



class UNET(nn.Module):
    def __init__(self, hp, output_channels=1, input_channels=4080, **kwargs):
        super(UNET, self).__init__()
        self.params = {'kernel_size': int(hp['k_size']), 'stride':1}
        self.params['padding'] = int((self.params['kernel_size'] -1)/2)
        self.output_channels = output_channels
        self.nn_layers = nn.ModuleList()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hp['c1'], **self.params)
        self.conv1b = nn.Conv1d(in_channels=hp['c1'], out_channels=hp['c1'], **self.params)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(in_channels=hp['c1'], out_channels=hp['c2'], **self.params)
        self.conv2b = nn.Conv1d(in_channels=hp['c2'], out_channels=hp['c2'], **self.params)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.conv3 = nn.Conv1d(in_channels=hp['c2'], out_channels=hp['c3'], **self.params)
        self.conv3b = nn.Conv1d(in_channels=hp['c3'], out_channels=hp['c3'], **self.params)
        self.pool3 = nn.MaxPool1d(kernel_size=5)

        self.conv4 = nn.Conv1d(in_channels=hp['c3'], out_channels=hp['c4'], **self.params)
        self.conv4b = nn.Conv1d(in_channels=hp['c4'], out_channels=hp['c4'], **self.params)
        self.pool4 = nn.MaxPool1d(kernel_size=6)

        self.conv5 = nn.Conv1d(in_channels=hp['c4'], out_channels=hp['c5'], **self.params)
        self.conv5b = nn.Conv1d(in_channels=hp['c5'], out_channels=hp['c5'], **self.params)
        self.conv6 = nn.Conv1d(in_channels=hp['c5'] + hp['c4'], out_channels=hp['c4'], **self.params)
        self.conv6b = nn.Conv1d(in_channels=hp['c4'], out_channels=hp['c4'], **self.params)
        self.conv7 = nn.Conv1d(in_channels=hp['c4']+hp['c3'], out_channels=hp['c3'], **self.params)
        self.conv7b = nn.Conv1d(in_channels=hp['c3'], out_channels=hp['c3'], **self.params)
        self.conv8 = nn.Conv1d(in_channels=hp['c3']+hp['c2'], out_channels=hp['c2'], **self.params)
        self.conv8b = nn.Conv1d(in_channels=hp['c2'], out_channels=hp['c2'], **self.params)
        self.conv9 = nn.Conv1d(in_channels=hp['c2']+hp['c1'], out_channels=hp['c1'], **self.params)
        self.conv9b = nn.Conv1d(in_channels=hp['c1'], out_channels=hp['c1'], **self.params)
        self.outputs = nn.Conv1d(in_channels=hp['c1'], out_channels=self.output_channels, kernel_size=1)

    def forward(self,x):
        conv1 = relu(self.conv1(x))
        conv1 = relu(self.conv1b(conv1))
        pool1 = self.pool1(conv1) #MaxPooling3D(pool_size=2)(conv1)

        conv2 = relu(self.conv2(pool1))
        conv2 = relu(self.conv2b(conv2))
        pool2 = self.pool2(conv2)

        conv3 = relu(self.conv3(pool2))
        conv3 = relu(self.conv3b(conv3))
        pool3 = self.pool3(conv3)

        conv4 = relu(self.conv4(pool3))
        conv4 = relu(self.conv4b(conv4))
        pool4 = self.pool4(conv4)

        conv5 = relu(self.conv5(pool4))
        conv5 = relu(self.conv5b(conv5))

        up6 = cat([nn.Upsample(size=6)(conv5) ,conv4], dim=1)
        conv6 = relu(self.conv6(up6))
        conv6 = relu(self.conv6b(conv6))

        up7 = cat([nn.Upsample(size=30)(conv6) ,conv3], dim=1)
        conv7 = relu(self.conv7(up7))
        conv7 = relu(self.conv7b(conv7))

        up8 = cat([nn.Upsample(size=120)(conv7), conv2], dim=1)
        conv8 = relu(self.conv8(up8))
        conv8 = relu(self.conv8b(conv8))

        up9 = cat([nn.Upsample(size=480)(conv8), conv1], dim=1)
        conv9 = relu(self.conv9(up9))
        conv9 = relu(self.conv9b(conv9))

        outputs = sigmoid(self.outputs(conv9))
        return outputs.squeeze()
