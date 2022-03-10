import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):

    def __init__(self):
        super(VGG16,self).__init__()
        #3*224*224
        self.conv1_1=nn.Conv2d(3,64,kernel_size=3, padding=1)
        #64*222*222
        self.conv1_2=nn.Conv2d(64,64,kernel_size=3, padding=1)
        #64*222*222
        self.maxpool1=nn.MaxPool2d(kernel_size=2, stride=2)
        #64*112*112

        self.conv2_1=nn.Conv2d(64,128,kernel_size=3, padding=1)
        #128*110*110
        self.conv2_2=nn.Conv2d(128,128,kernel_size=3, padding=1)
        #128*110*110
        self.maxpool2=nn.MaxPool2d(kernel_size=2, stride=2)
        #128*56*56

        self.conv3_1=nn.Conv2d(128,256,kernel_size=3, padding=1)
        #256*54*54
        self.conv3_2=nn.Conv2d(256,256,kernel_size=3, padding=1)
        #256*54*54
        self.conv3_3=nn.Conv2d(256,256,kernel_size=3, padding=1)
        #256*54*54
        self.maxpool3=nn.MaxPool2d(kernel_size=2, stride=2)
        #256*28*28

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # 512*26*26
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # 512*26*26
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # 512*26*26
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 512*14*14

        self.conv5_1=nn.Conv2d(512,512,kernel_size=3, padding=1)
        #512*12*12
        self.conv5_2=nn.Conv2d(512,512,kernel_size=3, padding=1)
        #512*12*12
        self.conv5_3=nn.Conv2d(512,512,kernel_size=3, padding=1)
        #512*12*12
        self.maxpool5=nn.MaxPool2d(kernel_size=2, stride=2)
        #512*7*7

        self.avgpool=nn.AdaptiveAvgPool2d((7,7))
        """
        self.feature=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        """
        '''
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        '''

    def forward(self,x):
        out=self.conv1_1(x)
        out=F.relu(out)
        out=self.conv1_2(out)
        out=F.relu(out)
        out=self.maxpool1(out)
        print('the first layer out:')
        print(out)

        out = self.conv2_1(out)
        out = F.relu(out)
        out = self.conv2_2(out)
        out = F.relu(out)
        out = self.maxpool2(out)
        print('the second layer out:')
        print(out)

        out = self.conv3_1(out)
        out = F.relu(out)
        out = self.conv3_2(out)
        out = F.relu(out)
        out = self.conv3_3(out)
        out = F.relu(out)
        out = self.maxpool3(out)
        print('the third layer out:')
        print(out)

        out = self.conv4_1(out)
        out = F.relu(out)
        out = self.conv4_2(out)
        out = F.relu(out)
        out = self.conv4_3(out)
        out = F.relu(out)
        out = self.maxpool4(out)
        print('the fourth layer out:')
        print(out)

        out = self.conv5_1(out)
        out = F.relu(out)
        out = self.conv5_2(out)
        out = F.relu(out)
        #out = self.conv5_3(out)
        #out = F.relu(out)
        #out = self.maxpool5(out)

        #out=self.avgpool(out)
        #out=torch.flatten(x,1)
        #out=self.classifier(out)

        return out
        
    def get_feature(self,imgs):
        out=self.forward(imgs)
        return out

