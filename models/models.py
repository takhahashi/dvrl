import torch
import torch.nn as nn

class DataValueEstimater(nn.Module):
    def __init__(self, x_dim, y_dim, y_hat_dim):
        super(DataValueEstimater, self).__init__()
        self.layer1 = nn.Linear(x_dim+y_dim, 100)  # 入力層から1つ目の隠れ層（64ユニット）
        self.layer2 = nn.Linear(100, 100)    # 1つ目の隠れ層から2つ目の隠れ層（128ユニット）
        self.layer3 = nn.Linear(100, 100)   # 2つ目の隠れ層から3つ目の隠れ層（256ユニット）
        self.layer4 = nn.Linear(100, 100)   # 3つ目の隠れ層から4つ目の隠れ層（128ユニット）
        self.layer5 = nn.Linear(100, 10)    # 4つ目の隠れ層から5つ目の隠れ層（64ユニット）
        self.comb_layer = nn.Linear(y_hat_dim+10, 10)
        self.output_layer = nn.Linear(10, 1)
        self.relu = nn.ReLU()  # ReLU活性化関数
        self.sigmoid = nn.Sigmoid()  # Sigmoid活性化関数

    def forward(self, hidden_vec, y, y_hat_diff):
        y = y.unsqueeze(1)
        y_hat_diff = y_hat_diff.unsqueeze(1)
        assert len(hidden_vec) == len(y) and len(hidden_vec) == y_hat_diff, "Shape Error at len(hidden_vec) == len(y) and len(hidden_vec)"
        x = torch.cat([hidden_vec, y], dim=1)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        comb_input = torch.cat([x, y_hat_diff], dim=1)
        comb_out = self.comb_layer(comb_input)
        output = self.sigmoid(self.output_layer(comb_out))
        return output
    
class Predictor(nn.Module):
    def __init__(self, x_dim, output_type, output_dim=None):
        super(DataValueEstimater, self).__init__()
        self.output_type = output_type
        self.layer1 = nn.Linear(x_dim, 100)  # 入力層から1つ目の隠れ層（64ユニット）
        self.layer2 = nn.Linear(100, 100)    # 1つ目の隠れ層から2つ目の隠れ層（128ユニット）
        if output_type == 'reg':
            self.output_layer = nn.Linear(100, 1)
        elif output_type == 'class':
            self.output_layer = nn.Linear(100, output_dim)
        self.relu = nn.ReLU()  # ReLU活性化関数
        self.sigmoid = nn.Sigmoid()  # Sigmoid活性化関数

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        if self.output_type == 'reg':
            output = self.sigmoid(self.output_layer(x))
        elif self.output_type == 'class':
            output = self.output_layer(x)
        return output