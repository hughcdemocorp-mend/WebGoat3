import torch
import torch.nn as nn

class LargeDummyModel(nn.Module):
    def __init__(self):
        super(LargeDummyModel, self).__init__()
        # Add more layers or parameters to increase the size
        self.linear1 = nn.Linear(1000, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, 1000)
        self.linear4 = nn.Linear(1000, 1000)
        self.linear5 = nn.Linear(1000, 1000)
        self.linear6 = nn.Linear(1000, 1000)
        self.linear7 = nn.Linear(1000, 1000)
        self.linear8 = nn.Linear(1000, 1000)
        self.linear9 = nn.Linear(1000, 1000)
        self.linear10 = nn.Linear(1000, 1000)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.linear6(x)
        x = self.linear7(x)
        x = self.linear8(x)
        x = self.linear9(x)
        x = self.linear10(x)
        return x

# Save the model
model = LargeDummyModel()
torch.save(model.state_dict(), "large_dummy_model.pth")
