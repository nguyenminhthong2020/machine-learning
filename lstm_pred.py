import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./data/NSE-TATA.csv')
df.head()

# Format date-time as year-month-date and use as index
df['Date'] = pd.to_datetime(df.Date, format = '%Y-%m-%d')
df.index = df['Date']

plt.figure(figsize = (16, 8))
plt.plot(df['Close'], label = 'Close Price history')

data = df.sort_index()
new_dataset = pd.DataFrame(index = range(0, len(df)), columns = ['Date','Close'])

for i in range(0, len(data)):
    new_dataset['Date'][i] = data['Date'][i]
    new_dataset['Close'][i] = data['Close'][i]

new_dataset.index = new_dataset.Date
new_dataset.drop('Date', axis = 1, inplace = True)

final_dataset = new_dataset.to_numpy()

train_data = final_dataset[0:987,:]
valid_data = final_dataset[987:,:]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(final_dataset)

x_train_data, y_train_data = [], []

for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i - 60 : i, 0])
    y_train_data.append(scaled_data[i, 0])
    
x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

lstm_model.compile(loss = 'mean_squared_error', optimizer = 'adam')
lstm_model.fit(x_train_data, y_train_data, epochs = 1, batch_size = 1, verbose = 2)

lstm_model.save('./model/saved_model.h5')

inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
inputs_data = inputs_data.reshape(-1,1)
inputs_data = scaler.transform(inputs_data)

X_test = []
for i in range(60, inputs_data.shape[0]):
    X_test.append(inputs_data[i - 60 : i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_closing_price = lstm_model.predict(X_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

previous_closing_price = new_dataset[:987]
test_closing_price = new_dataset[987:]
test_closing_price['Predictions'] = predicted_closing_price

plt.figure(figsize = (16, 8))
# data that used for training (black)
plt.plot(previous_closing_price['Close'], 'k')
# actual data (green)
plt.plot(test_closing_price['Close'], 'g')
# predicted data after trained (red)
plt.plot(test_closing_price['Predictions'], 'r')