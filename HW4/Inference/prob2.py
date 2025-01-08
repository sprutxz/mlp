import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import math
class Prob2(nn.Module):
    def __init__(self):
        super(Prob2, self).__init__()
        self.relu = nn.ReLU()
        #c1 layer
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        Fi, _ = nn.init._calculate_fan_in_and_fan_out(self.c1.weight)
        nn.init.uniform_(self.c1.weight, a=-2.4 / Fi, b=2.4 / Fi)
        self.c1.bias.data.fill_(2.4/Fi)

        #s2 layer
        self.s2_weight = nn.Parameter(torch.Tensor(1,6,1,1))
        self.s2_bias = nn.Parameter(torch.Tensor(1,6,1,1))
        Fi, _ = nn.init._calculate_fan_in_and_fan_out(self.s2_weight)
        nn.init.uniform_(self.s2_weight, a=-2.4 / Fi, b=2.4 / Fi)
        self.s2_bias.data.fill_(2.4/Fi)

        #c3 layer
        self.c3_weight = nn.Parameter(torch.Tensor(10, 6, 5, 5))
        self.c3_bias = nn.Parameter(torch.Tensor(1, 16, 1, 1))
        Fi, _ = nn.init._calculate_fan_in_and_fan_out(self.c3_weight)
        nn.init.uniform_(self.c3_weight, a=-2.4 / Fi, b=2.4 / Fi)
        self.c3_bias.data.fill_(2.4/Fi)

        #s4 layer
        self.s4_weight = nn.Parameter(torch.Tensor(1,16,1,1))
        self.s4_bias = nn.Parameter(torch.Tensor(1,16,1,1))
        Fi, _ = nn.init._calculate_fan_in_and_fan_out(self.s4_weight)
        nn.init.uniform_(self.s4_weight, a=-2.4 / Fi, b=2.4 / Fi)
        self.s4_bias.data.fill_(2.4/Fi)

        #c5 layer
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        Fi, _ = nn.init._calculate_fan_in_and_fan_out(self.c5.weight)
        nn.init.uniform_(self.c5.weight, a=-2.4 / Fi, b=2.4 / Fi)
        self.c5.bias.data.fill_(2.4/Fi)
        #f6 layer
        self.f6 = nn.Linear(in_features=120, out_features=84)
        Fi, _ = nn.init._calculate_fan_in_and_fan_out(self.f6.weight)
        nn.init.uniform_(self.f6.weight, a=-2.4 / Fi, b=2.4 / Fi)
        self.f6.bias.data.fill_(2.4/Fi)
        self.rbf_params = self.rbf_tensor()
        #Various prob2 layers
        self.batch1 = nn.BatchNorm2d(6)
        self.batch2 = nn.BatchNorm2d(16)
        self.batch3 = nn.BatchNorm2d(120)
        self.dropout1 = nn.Dropout(0.2)

    def custom_connection(self):
        return [
            [0, 4, 5, 6, 9, 10, 11, 12, 14, 15],
            [0, 1, 5, 6, 7, 10, 11, 12, 13, 15],
            [0, 1, 2, 6, 7, 8, 11, 13, 14, 15],
            [1, 2, 3, 6, 7, 8, 9, 12, 14, 15],
            [2, 3, 4, 7, 8, 9, 10, 12, 13, 15],
            [3, 4, 5, 8, 9, 10, 11, 13, 14, 15]
        ]
    def rbf(self, x):
        x_expanded = x.unsqueeze(1).expand((x.size(0), self.rbf_params.size(0), self.rbf_params.size(1)))  
        params_expanded = self.rbf_params.unsqueeze(0).expand((x.size(0), self.rbf_params.size(0), self.rbf_params.size(1)))         
        output = (x_expanded - params_expanded).pow(2).sum(-1)
        return output
        return output
    def rbf_tensor(self):
        kernel_list = []
        root_dir = r"C:\Users\patel\Downloads\digits"
        bitmap_size = (7, 12)
        for root, dirs, files in os.walk(root_dir):
            for dir_name in dirs:
                if dir_name.isdigit():
                    folder_path = os.path.join(root, dir_name)
                    digit_images = []
                    for file_name in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file_name)
                        image = cv2.imread(file_path, 0)
                        if image is not None:
                            image = cv2.resize(image, bitmap_size)
                            digit_images.append(image)
                    if digit_images:
                        mean_image = np.mean(digit_images, axis=0)
                        mean_image = cv2.threshold(mean_image, 127, 1, cv2.THRESH_BINARY)[1].astype(np.int16) * -1 + 1
                        kernel_list.append(mean_image.flatten())
        kernel_array = np.array(kernel_list)
        return torch.tensor(kernel_array, dtype=torch.float32)
    def move_to_cpu(self):
        self.to('cpu')  
        self.rbf_params = self.rbf_params.to('cpu')  
        print("Model moved to CPU")
    def forward(self, x):
        x = self.c1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)*self.s2_weight + self.s2_bias
        x = self.relu(x)
        output = torch.zeros(x.size(0), 16, x.size(3) - 5 + 1, x.size(3) - 5 + 1)
        custom = self.custom_connection()
        for i in range(len(custom)):
            output[:,custom[i],:,:] += (F.conv2d(x[:,i,:,:].unsqueeze(1),self.c3_weight[:,i,:,:].unsqueeze(1)) + self.c3_bias[:,custom[i],:,:])
        x = output
        x = self.batch2(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)*self.s4_weight + self.s4_bias
        x = self.relu(x)
        x = self.c5(x)
        x = self.batch3(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.f6(x)
        x = self.relu(x)
        x = self.rbf(x)
        return x
    