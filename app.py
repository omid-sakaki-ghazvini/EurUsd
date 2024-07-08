import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import keras.models
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf

####### Select type ########
start = "2024-02-01"
end = datetime.datetime.now()
with st.sidebar:
    select_type = st.selectbox('select type?', ['EUR USD'])
    st.write("-------------------------------------")
    search_date = st.radio('select time?',["D","H","30M"],horizontal=True)
    if search_date == "D":
        df = yf.download(tickers = 'EURUSD=X' ,start=start ,end=end)
        df = yf.download(tickers='EURUSD=X', start=start, end=end)
        model = tf.keras.models.load_model("/content/gdrive/MyDrive/kaggle/data_treade3/data_treader3/final2_model_EUR_USD.h5" , custom_objects=None, compile=True, safe_mode=True)
    elif search_date == "H":
        df = yf.download(tickers = 'EURUSD=X' ,start=start ,end=end , interval="2h")
        model = tf.keras.models.load_model("/content/gdrive/MyDrive/kaggle/data_treade3/data_treader3/final2_model_EUR_USD.h5" , custom_objects=None, compile=True, safe_mode=True)
    elif search_date == "30M":
        df = yf.download(tickers = 'EURUSD=X' ,start=start,end=end , interval="1h")
        model = tf.keras.models.load_model("/content/gdrive/MyDrive/kaggle/data_treade3/data_treader3/final2_model_EUR_USD.h5" , custom_objects=None, compile=True, safe_mode=True)
    st.write("-------------------------------------")    
    
####### load data ########
df = yf.download(tickers = 'EURUSD=X' ,start="2000-10-01",end=datetime.datetime.now())
df.drop(columns=['Volume','Adj Close'], inplace=True)

####### Data preparation ########
dataset_train_actual = df.copy()
dataset_train_actual = dataset_train_actual.reset_index()

dataset_train_timeindex = dataset_train_actual.set_index('Date')
dataset_train = dataset_train_actual.copy()

cols = list(dataset_train)[1:5]

datelist_train = list(dataset_train['Date'])
datelist_train = [date for date in datelist_train]
dataset_train = dataset_train[cols].astype(str)
for i in cols:
    for j in range(0, len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(',', '')

dataset_train = dataset_train.astype(float)
training_set = dataset_train.values

sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)

sc_predict = StandardScaler()
sc_predict.fit_transform(training_set[:, 0:1])

X_train = []
y_train = []

n_future = 10   # Number of days we want top predict into the future
n_past = 15     # Number of past days we want to use to predict the future

for i in range(n_past, len(training_set_scaled) - n_future +1):
    X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1]])
    y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

####### Output preparation ########
datelist_future = pd.date_range(datelist_train[-1],
                                periods=n_future, freq='1d').tolist()


# Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
datelist_future_ = []
for this_timestamp in datelist_future:
    datelist_future_.append(this_timestamp.date())


predictions_future = model.predict(X_train[-n_future:])
predictions_train = model.predict(X_train[n_past:])

def datetime_to_timestamp(x):
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)

PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Close']).set_index(pd.Series(datelist_future))
PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Close']).set_index(pd.Series(datelist_train[2 * n_past + n_future -1:]))

fu = PREDICTIONS_FUTURE.index[-1]
tomorrow=PREDICTIONS_FUTURE.loc[fu]['Close']

pa = dataset_train_timeindex.index[-1]
today=dataset_train_timeindex.loc[pa]['Close']

delta = tomorrow - today

####### Output streamlit ########
st.title("EUR USD **Forecasting**")
col1, col2 = st.columns(2)
col1.metric(label="Close Volume **today**",
          value=today)
col2.metric(label="Close Volume **tomorrow**",
          value=tomorrow, delta = delta)


fig = plt.figure(figsize = (14, 5))

# Plot parameters
START_DATE_FOR_PLOTTING = start

plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Close'], color='r', label='Predicted EUR USD')
#plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Close'], color='orange',
 #        label='Training predictions')
plt.plot(dataset_train_timeindex.loc[START_DATE_FOR_PLOTTING:].index, dataset_train_timeindex.loc[START_DATE_FOR_PLOTTING:]['Close'], color='b',
         label='Actual EUR USD')

plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

plt.grid(which='major', color='#cccccc', alpha=0.5)

plt.legend(shadow=True)
plt.title('Predcitions and Acutal EUR USD values', fontsize=12)
plt.xlabel('Timeline', fontsize=10)
plt.ylabel('Stock Price Value', fontsize=10)
st.pyplot(fig)
