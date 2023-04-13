import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
     

# Sequence length of input
WINDOW_SIZE = 14
# Number of hidden layer neurons in LSTM
HIDDEN_SIZE = 128
# Number of LSTMs
NUM_LSTMS = 4
     

stock_df = pd.read_csv('AAPL.csv')
stock_df.head()

news_data = pd.read_csv('News_sentiment.csv')
news_data.head()

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

X = data.loc[:, 'Label':'Volume']

# %%

y = data.loc[:, 'Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = X_train.drop(['Adj Close'], axis=1)

X_test = X_test.drop(['Adj Close'], axis=1)

## X_test and y_test for prediction

# %%

news_data = pd.read_csv('News_sentiment.csv')
news_data.head()

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

model = LSTM_Model(HIDDEN_SIZE, NUM_LSTMS)

model.load_state_dict(torch.load("stock_model.pt", map_location=torch.device('cpu')))
model.eval()
     

prediction = model(X_test)