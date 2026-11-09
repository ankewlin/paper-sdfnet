"""作者 MaMIMO CNN 的 PyTorch 逐层适配版。

来源：sibrendebast/MaMIMO-CSI-positioning-using-CNNs，positioning.py/build_nn。
上游仓库许可证：GPL-3.0；本派生实现按相同许可证使用。
输入适配经用户确认：统计天线行与全部物理路径行补零后沿行拼接。
原来的复数 CSI 实部/虚部、幅相转换和 IFFT 不适用于这些指纹，故由输入适配替代。
原始六种信号表示的深度轴变成一个指纹值轴；对应卷积核深度从 6 改成 1。
其余卷积核、步长、8个卷积输出通道、拼接位置、激活和丢弃概率均按原代码保留。
"""

import math

import torch
from torch import nn
from torch.nn import functional as functional


class SameConv3d(nn.Module):
    """复现 Keras padding='same'；偶数卷积核也按前少后多补零。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1)):
        super().__init__()
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.conv = nn.Conv3d(in_channels, out_channels, self.kernel_size,
                              stride=self.stride, padding=0, bias=True)

    def forward(self, values):
        padding = []
        for length, kernel, stride in reversed(list(zip(
                values.shape[2:], self.kernel_size, self.stride))):
            total = max((math.ceil(length / stride) - 1) * stride + kernel - length, 0)
            padding.extend((total // 2, total - total // 2))
        return self.conv(functional.pad(values, padding))


class OpenCNN(nn.Module):
    """单流指纹 CNN；输入 [B,1,行数,特征列数,1]，输出三维米制坐标。"""

    def __init__(self, input_rows=31, input_columns=7, output_dim=3):
        super().__init__()
        if input_rows < 1 or input_columns < 1 or output_dim != 3:
            raise ValueError("需要正的输入尺寸和三维定位输出")
        self.input_rows = input_rows
        self.input_columns = input_columns

        # 下列名称与作者代码 layD1、layD2...、layV1... 一一对应。
        # 指纹没有原始六种复数信号表示，表示深度为1；仅相关核深度随之变成1。
        self.d1 = SameConv3d(1, 8, (1, 23, 1), (1, 5, 1))
        self.d2 = SameConv3d(8, 8, (1, 23, 1))
        self.d2_mix = SameConv3d(16, 8, (1, 1, 1))
        self.d2_down = SameConv3d(8, 8, (1, 23, 1), (1, 5, 1))
        self.d3 = SameConv3d(8, 8, (1, 23, 1))
        self.d3_mix = SameConv3d(16, 8, (1, 1, 1))
        self.d3_down = SameConv3d(8, 8, (1, 23, 1), (1, 5, 1))
        self.d4 = SameConv3d(8, 8, (1, 23, 1))
        self.d4_mix = SameConv3d(16, 8, (1, 1, 1))

        # 作者的V段每次拼接保留前一层的全部通道，因此依次为16、24、32、40。
        self.v1 = SameConv3d(8, 8, (8, 1, 1))
        self.v2 = SameConv3d(16, 8, (8, 1, 1))
        self.v3 = SameConv3d(24, 8, (8, 1, 1))
        self.v4 = SameConv3d(32, 8, (8, 1, 1))
        self.v5 = SameConv3d(40, 8, (8, 1, 1))

        self.activation = nn.LeakyReLU(negative_slope=0.3)
        # 使用逐元素Dropout，不能替换成整通道丢弃的Dropout3d。
        self.dropout = nn.Dropout(p=0.25)
        reduced_columns = math.ceil(math.ceil(math.ceil(input_columns / 5) / 5) / 5)
        self.fc1 = nn.Linear(input_rows * reduced_columns * 8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, output_dim)

        # 对齐Keras默认的GlorotUniform和零偏置。
        for layer in self.modules():
            if isinstance(layer, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, values):
        if tuple(values.shape[1:]) != (1, self.input_rows, self.input_columns, 1):
            raise ValueError("输入应为 [B,1,{}, {},1]".format(
                self.input_rows, self.input_columns))

        d1 = self.dropout(self.activation(self.d1(values)))
        d2 = self.activation(self.d2(d1))
        d2 = torch.cat((d1, d2), dim=1)
        d2 = self.activation(self.d2_mix(d2))
        d2 = self.dropout(self.activation(self.d2_down(d2)))

        d3 = self.activation(self.d3(d2))
        d3 = torch.cat((d2, d3), dim=1)
        d3 = self.activation(self.d3_mix(d3))
        d3 = self.dropout(self.activation(self.d3_down(d3)))

        d4 = self.activation(self.d4(d3))
        d4 = torch.cat((d4, d3), dim=1)
        d4 = self.activation(self.d4_mix(d4))

        v1 = self.dropout(self.activation(self.v1(d4)))
        v1 = torch.cat((v1, d4), dim=1)
        v2 = self.dropout(self.activation(self.v2(v1)))
        v2 = torch.cat((v2, v1), dim=1)
        v3 = self.dropout(self.activation(self.v3(v2)))
        v3 = torch.cat((v3, v2), dim=1)
        v4 = self.dropout(self.activation(self.v4(v3)))
        v4 = torch.cat((v4, v3), dim=1)
        v5 = self.dropout(self.activation(self.v5(v4)))

        # Keras默认channels_last，Flatten前按其轴顺序排列，以保留全连接连接含义。
        flat = v5.permute(0, 2, 3, 4, 1).contiguous().flatten(1)
        hidden = functional.relu(self.fc1(flat))
        hidden = functional.relu(self.fc2(hidden))
        return self.output(hidden)

    def kernel_regularization(self):
        # 仅在三处卷积核上指定regularizers.l2(0.01)，不使用全局weight_decay替代。
        return 0.01 * sum(layer.conv.weight.square().sum()
                          for layer in (self.d2_down, self.d3_down, self.v2))
