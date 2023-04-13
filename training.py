# %%
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# %%

WINDOW_SIZE = 14

HIDDEN_SIZE = 128

NUM_LSTMS = 4

BATCH_SIZE = 32

EPOCHS = 500

# %%

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%

# print(DEVICE)
# print(torch.cuda.get_device_name())

# %%

stock_df = pd.read_csv('AAPL.csv')
stock_df.head()

# %%

news_data = pd.read_csv('News_sentiment.csv')
news_data.head()

# %%

data = pd.merge(news_data, stock_df, on='Date', how='left')
data = data.dropna()
data

# %%

cols = data.columns.tolist()
# cols
cols = [col for col in cols if col != 'Close'] + ['Close']
# cols
data = data.reindex(columns=cols)
data

# %%

data['Volume'] = data['Volume'].astype(float)

# %%

# train_series = data['Close'].to_list()[::-1]

# # train_series
# test_series = data['Close'].to_list()[::-1]

# test_time = test_df['timestamp'].to_list()[::-1]
# test_time = [date.fromisoformat(time) for time in test_time]


# %%

X = data.loc[:, 'Label':'Volume']

# %%

y = data.loc[:, 'Close']


#%%
# print(X)

# print(y)
# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = X_train.drop(['Adj Close'], axis=1)

X_test = X_test.drop(['Adj Close'], axis=1)



print((y_test.shape))

# y_test

# %%
class SeriesDataset(Dataset):
    def __init__(self, window_size, X, y):
        # self.data = data.values
        self.data = X.values
        self.labels = y.values

        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        sequence = self.data[idx: idx + self.window_size]
        label = self.labels[idx: idx + self.window_size]

        # print("Sequence of class: ", sequence)
        # print("Label of class: ", label)

        return torch.tensor(sequence).float(), torch.tensor(label).unsqueeze(1)

# %%

train_dataset = SeriesDataset(WINDOW_SIZE, X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# %%

test_dataset = SeriesDataset(WINDOW_SIZE, X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# %%

class LSTM_Model(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(LSTM_Model, self).__init__()

        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(input_size=8, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        # x = x[:, -1]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# %%
model = LSTM_Model(HIDDEN_SIZE, NUM_LSTMS)
model.to(DEVICE)

optimizer = optim.Adam(model.parameters())
loss_function = nn.L1Loss()

# %%

for epoch in range(EPOCHS):
    loop = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{EPOCHS}', colour='green')
    for sequence, label in loop:
        sequence = sequence.to(DEVICE)
        label = label.to(DEVICE)
        print(sequence.shape, label.shape)
        output = model(sequence)
        print(output.shape)
        loss = loss_function(label, output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix({'L1 Loss': loss.item()})

# %%
predictions = []
model.eval()

loop = tqdm(test_dataloader, colour='green')

for sequence, label in loop:
    sequence = sequence.to(DEVICE)
    label = label.to(DEVICE)

    output = model(sequence)

    for val in output:
        temp = val.detach().cpu()

        # print(temp.numpy())
        
        predictions.append(temp.numpy())
    
# %%

torch.save(model.state_dict(), "stock_model_2.pt")


predictions = []
model.eval()

loop = tqdm(test_dataloader, colour='green')
for sequence, label in loop:
  sequence = sequence.to(DEVICE)
  label = label.to(DEVICE)
  
  output = model(sequence)

  for val in output:
    # predictions.append(val.item())
    predictions.append(val)

# print(predictions)

print(len(predictions))
# %%

# plt.figure(figsize=(10, 4))
# plt.plot(test_time[WINDOW_SIZE:], test_series[WINDOW_SIZE:], label='Original')
# plt.plot(test_time[WINDOW_SIZE:], predictions, label='Prediction')
# plt.legend()
# plt.xlabel('Days', fontsize=12)
# plt.ylabel('Price in USD', fontsize=12)
# plt.title(f'{symbol} Stock', fontsize=18)
# plt.show()
