import torch.nn as nn
import torch

FILTERS=128

class model(nn.Module):
    # TODO:define model
    def __init__(self):
        super(model, self).__init__()
        self.conv41 = nn.Conv2d(in_channels=16, out_channels=FILTERS, kernel_size=(4, 1), padding=0, bias=True)
        self.conv14 = nn.Conv2d(in_channels=16, out_channels=FILTERS, kernel_size=(1, 4), padding=0, bias=True)
        self.conv22 = nn.Conv2d(in_channels=16, out_channels=FILTERS, kernel_size=(2, 2), padding=0, bias=True)
        self.conv33 = nn.Conv2d(in_channels=16, out_channels=FILTERS, kernel_size=(3, 3), padding=0, bias=True)
        self.conv44 = nn.Conv2d(in_channels=16, out_channels=FILTERS, kernel_size=(4, 4), padding=0, bias=True)


        #self.batchnorm = nn.BatchNorm1d(FILTERS*16*8)

        self.fc1 = nn.Linear(FILTERS*22, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)



        self.relu = nn.functional.relu

    def forward(self, x):

        x = torch.cat([self.conv14(x).view(x.size()[0], -1),
                       self.conv41(x).view(x.size()[0], -1),
                       self.conv22(x).view(x.size()[0], -1),
                       self.conv33(x).view(x.size()[0], -1),
                       self.conv44(x).view(x.size()[0], -1),], dim = 1)

        #x = self.batchnorm(x)

        x = self.fc1(self.relu(x))
        x = self.fc2(self.relu(x))
        x = self.fc3(self.relu(x))

        return x