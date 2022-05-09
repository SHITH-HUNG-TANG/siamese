import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class siamese(nn.Module):
    def __init__(self,in_ch = 3):
        super(siamese,self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=16,kernel_size=7),
            # nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            # nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=16,out_channels=64,kernel_size=7),
            # nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            # nn.Dropout(p=0.1)
        )

        self.together = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=256,kernel_size=7),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256*21*21,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,input1,input2):
        output1 = self.cnn(input1)
        output2 = self.cnn(input2)
        difference = torch.abs(output1 - output2)
        result = self.together(difference)
        # result = F.pairwise_distance(output1, output2, p=2)
        return result

if __name__ == '__main__':
    model = siamese(in_ch=1).cuda()
    model.eval()
    input1 = torch.rand(1,1,128,128).cuda()
    input2 = torch.rand(1,1,128,128).cuda()
    output = model(input1,input2)



#
# #keras model
# digit_input = Input(inputshape)
# m = Conv2D(filters=16, kernel_size=7, strides=(1, 1))(digit_input)
# # m = BatchNormalization()(m)
# m = Activation('relu')(m)
# m = MaxPooling2D((2, 2), strides=(2, 2))(m)
# # m = Dropout(0.1)(m)
# m = Conv2D(filters=64, kernel_size=7, strides=(1, 1))(m)
# # m = BatchNormalization()(m)
# m = Activation('relu')(m)
# m = MaxPooling2D((2, 2), strides=(2, 2))(m)
# # out = Dropout(0.1)(m)
#
#
# pair_model = Model(digit_input, m)
# input_a = Input(shape=inputshape)
# input_b = Input(shape=inputshape)
# out_a = pair_model(input_a)
# out_b = pair_model(input_b)
# subtracted = Lambda(sub)([out_a, out_b])
#
# M = Conv2D(filters=256, kernel_size=7, strides=(1, 1))(subtracted)
# M = Dropout(0.1)(M)
# M = Activation('relu')(M)
# M = Flatten()(M)
# M = Dense(256)(M)
# M = Activation('relu')(M)
# M = Dense(1)(M)
# out = Activation('sigmoid')(M)
# c3_model = Model(sub_input,out)
# out = c3_model(subtracted)
#
# model = Model([input_a, input_b], out)

