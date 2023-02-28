import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

print("Reading the dataset...")
uberdata = pd.read_csv('uber-raw-data-janjune-15.csv')
print("Preparing data and performing LSTM model...")
uberdata['Month_Day'] = uberdata['Pickup_date'].apply(lambda pickup: datetime.strptime(pickup, '%Y-%m-%d %H:%M:%S').strftime('%m-%d').split('-'))
uberdata['Month'] = [month_day [0] for month_day in uberdata['Month_Day']]
uberdata['Day'] = [month_day [1] for month_day in uberdata['Month_Day']]

uberdata_grouped = uberdata.groupby(by = ['Month', 'Day']).size().unstack()

all_uberdata = [uberdata_grouped.iloc[r,:] for r in range(uberdata_grouped.shape[0])]
all_uberdata = [trips for month in all_uberdata for trips in month]

cleaned_data = list(np.argwhere(np.isnan(all_uberdata) == True).reshape((1,5))[0])
all_uberdata_mod = [all_uberdata[i] for i,j in enumerate(all_uberdata) if i not in cleaned_data]
uberdata_final = pd.DataFrame({'Days': range(1,len(all_uberdata_mod)+1), 'Trips': all_uberdata_mod})
uberdata_final.head()

train_uberdata = uberdata_final.iloc[0:167,1:2].values
test_uberdata = uberdata_final.iloc[167:,1:2].values


min_max_data = MinMaxScaler(feature_range = (0,1))
train_uberdata_scaled = min_max_data.fit_transform(train_uberdata)

trainX = []
trainY = []

for taxi_rides in range(14, 167):
    trainX.append(train_uberdata_scaled[taxi_rides-14:taxi_rides,0])
    trainY.append(train_uberdata_scaled[taxi_rides,0])
    
trainX, trainY = np.array(trainX), np.array(trainY)
trainX = np.reshape(trainX, newshape = (trainX.shape[0], trainX.shape[1], 1))

np.random.seed(11)



def model(n_units, inpX, inpY, inp_drop, epoch_num, n_batch, model_optm, model_loss):
    
    model_reg = Sequential()
    model_reg.add(LSTM(units = n_units, return_sequences = True, input_shape = (inpX.shape[1],1)))
    model_reg.add(Dropout(inp_drop))
    model_reg.add(LSTM(units = n_units, return_sequences = True))
    model_reg.add(Dropout(inp_drop))
    model_reg.add(LSTM(units = n_units, return_sequences = True))
    model_reg.add(Dropout(inp_drop))
    model_reg.add(LSTM(units = n_units, return_sequences = True))
    model_reg.add(Dropout(inp_drop))
    model_reg.add(LSTM(units = n_units, return_sequences = False))
    model_reg.add(Dropout(inp_drop))
    model_reg.add(Dense(units = 1))
    model_reg.compile(optimizer = model_optm, loss = model_loss)
    model_reg.fit(x = inpX, y = inpY, epochs = epoch_num, batch_size = n_batch)

    return model_reg
    
model_reg = model(n_units = 40, inpX = trainX, inpY = trainY, inp_drop = 0.2, epoch_num = 100, n_batch = 16, model_optm = 'adam', model_loss = 'mean_squared_error')

inp_adj = uberdata_final[len(uberdata_final) - len(test_uberdata) - 14:]['Trips'].values
inp_adj = inp_adj.reshape(-1,1)
inp_adj = min_max_data.transform(inp_adj)
inp_adj[0:10]

testX = []
for taxi_rides in range(14,29):
    testX.append(inp_adj[taxi_rides-14:taxi_rides,0])
    
testX = np.array(testX)
testX = np.reshape(testX, newshape = (testX.shape[0], testX.shape[1], 1))
testX.shape

model_pred = model_reg.predict(testX)
model_pred = min_max_data.inverse_transform(model_pred)

model_res = model_pred[0:-1] - test_uberdata
model_root_mean_sq_err = np.sqrt(np.mean(model_res**2))
model_root_mean_sq_err

out_fig ,out_axis = plt.subplots(figsize = (12,6))
out_axis.plot(model_res)
out_axis.set_xlabel('Days last two weeks of June-2015 (Test Data)', fontsize = 15)
out_axis.set_ylabel('Predictions Error', fontsize = 15)
out_axis.set_title('Model Performance: {}'.format(round(model_root_mean_sq_err, 3)), fontsize = 20)
plt.show()