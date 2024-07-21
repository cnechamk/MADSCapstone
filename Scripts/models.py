import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification


class LSTM(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        # nn.init.xavier_normal_(self.lstm.all_weights)
        self.fn_0 = nn.Linear(13 * 27973, input_size * hidden_size)

        self.fn_1 = nn.Linear(input_size * hidden_size, 512)
        self.bn_1 = nn.BatchNorm1d(512)

        self.fn_2 = nn.Linear(512, 256)
        self.bn_2 = nn.BatchNorm1d(256)

        self.fn_3 = nn.Linear(256, 128)
        self.bn_3 = nn.BatchNorm1d(128)

        self.fn_4 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        # x, _ = self.lstm(x)
        x = x.flatten(1)
        x = self.fn_0(x)

        x = self.fn_1(x)
        x = self.bn_1(x)
        x = F.tanh(x)

        x = self.fn_2(x)
        x = self.bn_2(x)
        x = F.tanh(x)

        x = self.fn_3(x)
        x = self.bn_3(x)
        x = F.tanh(x)

        x = self.fn_4(x)
        x = F.tanh(x) / 10

        return x

class BertRegressor(nn.Module):
    def __init__(self, start_channels=32):
        super().__init__()

        self.bert = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.bert.requires_grad_(False)

        self.district_block = nn.Sequential(  # focuses on district
            nn.Conv2d(1, start_channels, kernel_size=(13, 1)),
            nn.Tanh()
        )
        self.score_block = nn.Sequential(  # focuses on sentiment score
            nn.Conv2d(1, start_channels, kernel_size=(1, 3)),
            nn.Tanh()
        )

        self.expand_block = nn.Sequential(
            nn.Conv2d(1, start_channels, kernel_size=1),
            nn.Tanh()
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(start_channels * 2 * 13 * 3, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.final_conv_2d = nn.Conv2d(start_channels * 2, start_channels, (13, 3))
        self.final_conv_1d = nn.Conv1d(1, 1, start_channels)

    def forward(self, x):
        x = torch.concat([self.bert(**inputs)[0].mean(dim=0, keepdim=True) for inputs in x])
        x = x[None, None, :, :]
        xh = self.district_block(x)
        xw = self.score_block(x)

        xhw = xw @ xh
        x = self.expand_block(x)
        x = torch.concat((x, xhw), dim=1)

        linear_x = self.linear_layers(x.flatten(1))

        x = self.final_conv_2d(x)
        x = F.tanh(x)
        x = self.final_conv_1d(x[:, :, 0, 0])
        x = linear_x + x
        x = F.tanh(x) / 10

        return x


if __name__ == "__main__":
    """ Example usage: """
    from datasets import FOMCImpactDataset

    p_bb = "../Data/beige_books.csv"
    p_fomc = "../Data/fomc_impact.csv"

    dset = FOMCImpactDataset(p_bb, p_fomc, device='mps')
    x, y = dset[2]

    model = BertRegressor().to('mps')
    print(sum(p.numel() for p in model.parameters()))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    y_pred = model(x)
    print(y_pred)
