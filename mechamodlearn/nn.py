# File: nn.py
#
import torch
import torch.nn as nn

class Identity(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

ACTIVATIONS = {
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU,
    'elu': torch.nn.ELU,
    'identity': Identity
}


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation='relu', ln=False):
        super().__init__()
        self.activation = ACTIVATIONS[activation]()
        self.ln = ln
        
        # 主路径
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        
        # 跳跃连接（维度调整）
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim)
            )
        
        # 层归一化
        if ln:
            self.norm1 = nn.LayerNorm(out_dim)
            self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.fc1(x)
        if self.ln:
            out = self.norm1(out)
        out = self.activation(out)
        
        out = self.fc2(out)
        if self.ln:
            out = self.norm2(out)
            
        out += residual  # 残差连接
        return self.activation(out)

class ResMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', ln=True):
        super().__init__()
        layers = []
        current_dim = input_size
        
        # 构建残差块
        for h_dim in hidden_sizes:
            layers.append(
                ResidualBlock(current_dim, h_dim, activation, ln)
            )
            current_dim = h_dim
        
        # 输出层
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(current_dim, output_size)
        
        # 初始化参数
        # self._init_weights()
        self.reset_params(gain=0.1)

    def reset_params(self, gain=1.0):
        self.apply(lambda x: weights_init_mlp(x, gain=gain))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    


# class LNMLP(torch.nn.Module):

#     def __init__(self, input_size, hidden_sizes, output_size, activation='tanh', gain=1.0,
#                  ln=False):
#         self._hidden_sizes = hidden_sizes
#         self._gain = gain
#         self._ln = ln
#         super().__init__()
#         activation = ACTIVATIONS[activation]
#         layers = [torch.nn.Linear(input_size, hidden_sizes[0])]
#         layers.append(activation())
#         if ln:
#             layers.append(torch.nn.LayerNorm(hidden_sizes[0]))

#         for i in range(len(hidden_sizes) - 1):
#             layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
#             layers.append(activation())
#             if ln:
#                 layers.append(torch.nn.LayerNorm(hidden_sizes[i + 1]))

#         layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))

#         self._layers = layers
#         self.mlp = torch.nn.Sequential(*layers)
#         self.reset_params(gain=gain)

#     def forward(self, inp):
#         return self.mlp(inp)

#     def reset_params(self, gain=1.0):
#         self.apply(lambda x: weights_init_mlp(x, gain=gain))


def weights_init_mlp(m, gain=1.0):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init_normc_(m.weight.data, gain)
        if m.bias is not None:
            m.bias.data.fill_(0)


def init_normc_(weight, gain=1.0):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
