
import torch
import torch.nn as nn
import torch.nn.functional as F

import ai8x



# class AI85Net_FPR(nn.Module):
#     """
#     5-Layer CNN that uses max parameters in AI84
#     """
#     def __init__(self, num_classes=100, num_channels=1,dimensions=(64, 64),  bias=False, **kwargs):
#         super().__init__()
#
#         # self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.conv1 = ai8x.FusedMaxPoolConv2dReLU(num_channels, 16, 3, pool_size=2, pool_stride=2,
#                                                  stride=1, padding=1, bias=bias, **kwargs)
#         self.conv2 = ai8x.FusedConv2dReLU(16, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.conv3 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.conv4 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.resid1 = ai8x.Add()
#         self.conv5 = ai8x.FusedMaxPoolConv2dReLU(20, 20, 3, pool_size=2, pool_stride=2,
#                                                  stride=1, padding=1, bias=bias, **kwargs)
#         self.conv6 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.resid2 = ai8x.Add()
#         self.conv7 = ai8x.FusedConv2dReLU(20, 44, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.conv8 = ai8x.FusedMaxPoolConv2dReLU(44, 48, 3, pool_size=2, pool_stride=2,
#                                                  stride=1, padding=1, bias=bias, **kwargs)
#         self.conv9 = ai8x.FusedConv2dReLU(48, 48, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.resid3 = ai8x.Add()
#         self.conv10 = ai8x.FusedMaxPoolConv2dReLU(48, 32, 3, pool_size=2, pool_stride=2,
#                                                   stride=1, padding=0, bias=bias, **kwargs)
#         # self.conv11 = ai8x.FusedAvgPoolConv2dReLU(96, 32, 1, pool_size=2, pool_stride=1,
#         #                                           padding=0, bias=bias, **kwargs)
#         self.fc = ai8x.Linear(32*2*2, num_classes, bias=True, wide=True, **kwargs)
#         for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#     def forward(self, x):  # pylint: disable=arguments-differ
#         """Forward prop"""
#
#         x = self.conv1(x)  # 16x32x32
#         x_res = self.conv2(x)  # 20x32x32
#         x = self.conv3(x_res)  # 20x32x32
#         x = self.resid1(x, x_res)  # 20x32x32
#         x = self.conv4(x)  # 20x32x32
#         x_res = self.conv5(x)  # 20x16x16
#         x = self.conv6(x_res)  # 20x16x16
#         x = self.resid2(x, x_res)  # 20x16x16
#         x = self.conv7(x)  # 44x16x16
#         x_res = self.conv8(x)  # 48x8x8
#         x = self.conv9(x_res)  # 48x8x8
#         x = self.resid3(x, x_res)  # 48x8x8
#         x = self.conv10(x)  # 96x4x4
#         # x = self.conv11(x)  # 512x2x2
#         # print(x.size())
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x


class AI85Net_FPR(nn.Module):
    """
    5-Layer CNN that uses max parameters in AI84
    """
    def __init__(self, num_classes=10, num_channels=1, dimensions=(64, 64),
                 planes=32, pool=2, fc_inputs=256, bias=False, **kwargs):
        super().__init__()

        # Limits
        assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        assert planes + fc_inputs <= ai8x.dev.WEIGHT_DEPTH-1

        # Keep track of image dimensions so one constructor works for all image sizes
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, stride=1,padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 20, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(20,20, kernel_size=3, stride=1, padding=1, bias=False)
        # self.resid1 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv5 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1, bias=False)
        # self.resid2 = nn.Conv2d(20, 20, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv7 = nn.Conv2d(20, 44, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8 = nn.Conv2d(44, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool8 = nn.MaxPool2d(2, 2)
        self.conv9 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        # self.resid3 = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv10 = nn.Conv2d(48, 32, kernel_size=3, stride=1, padding=0, bias=False)
        self.pool10 = nn.MaxPool2d(2, 2)
        # self.conv11 = nn.Conv2d(96, 32, kernel_size=1, stride=1, padding=0, bias=False)
        # self.pool11 = nn.AvgPool2d(2, 2)

        self.fc = nn.Linear(32*2*2, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.pool1(F.relu(self.conv1(x)))
        x_res = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x_res))
        x= x + x_res

        x = F.relu(self.conv4(x))
        x_res = self.pool5(F.relu(self.conv5(x)))
        x = F.relu(self.conv6(x_res))
        x= x + x_res

        x = F.relu(self.conv7(x))
        x_res = self.pool8(F.relu(self.conv8(x)))
        x = F.relu(self.conv9(x_res))
        x = x + x_res
        x=self.pool10(x)
        x = F.relu(self.conv10(x))
        # x = self.pool11(F.relu(self.conv11(x)))

        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ai85net_fpr(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return AI85Net_FPR(**kwargs)

models = [
    {
        'name': 'ai85net_fpr',
        'min_input': 1,
        'dim': 2,
    },

]