
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model

def prediction(model):

  #Read data and analyze closing price from dataframe
  df_nse = pd.read_csv("./assets/NSE-TATA.csv")

  df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
  df_nse.index=df_nse['Date']

  # Sort the dataset on date time and filter “Date” and “Close” columns
  data=df_nse.sort_index(ascending=True,axis=0)
  new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])

  for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Close"][i]=data["Close"][i]

  #Normalize the new filtered dataset
  new_data.index=new_data.Date
  new_data.drop("Date",axis=1,inplace=True)

  dataset=new_data.values

  index_start = int(len(dataset)*0.8)

  train=dataset[0:index_start,:]
  valid=dataset[index_start:,:]

  scaler=MinMaxScaler(feature_range=(0,1))
  scaled_data=scaler.fit_transform(dataset)

  x_train,y_train=[],[]

  for i in range(50,len(train)):
    x_train.append(scaled_data[i-50:i,0])
    y_train.append(scaled_data[i,0])
    
  x_train,y_train=np.array(x_train),np.array(y_train)

  x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


  #Load yourmodel
  model=load_model(model)



  #Take a sample of a dataset to make stock price predictions using the "YOUR" model
  inputs=new_data[len(new_data)-len(valid)-50:].values
  inputs=inputs.reshape(-1,1)
  inputs=scaler.transform(inputs)

  X_test=[]
  for i in range(50,inputs.shape[0]):
    X_test.append(inputs[i-50:i,0])
  X_test=np.array(X_test)

  X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
  closing_price=model.predict(X_test)
  closing_price=scaler.inverse_transform(closing_price)

  train=new_data[:index_start]
  valid=new_data[index_start:]
  valid['Predictions']=closing_price

  return train, valid