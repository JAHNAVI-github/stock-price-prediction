from flask import Flask, render_template, request
from flask_pymongo import PyMongo
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

# from matplotlib import Scalar
from flask import Flask, render_template, request, flash, redirect, url_for
from alpha_vantage.timeseries import TimeSeries
from flask_pymongo import PyMongo
from alpha_vantage.timeseries import TimeSeries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor

# from matplotlib import Scalar
import pandas as pd
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Flatten, Conv1D,  Dropout,  GRU, Bidirectional
# from sklearn.linear_model import LinearRegression
#Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#***************** FLASK *****************************
app = Flask(__name__)

#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response



# Configuring the MongoDB connection
app.config['MONGO_URI'] = 'mongodb://localhost:27017/signupdb'
mongo = PyMongo(app)

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route("/signup", methods=['POST','GET'])
def signup():
    name = request.form.get('name','')
    number = request.form.get('number','')
    email = request.form.get('email','')
    password = request.form.get('password','')

    users = mongo.db.detail
    users.insert_one({'name': name, 'number': number, 'email': email, 'password': password})

    return render_template("signin.html")
    

@app.route("/signin", methods=['POST','GET'])
def signin():
    try:
        if request.method == 'POST':  # Corrected indentation and condition
            mail1 = request.form.get('email', '')
            password1 = request.form.get('password', '')
            users = mongo.db.detail
            user_data = users.find_one({'email': mail1, 'password': password1})
            
            if user_data is None:
                return render_template("index.html")

            if mail1 == 'admin' and password1 == 'admin':
                return render_template("index.html")

            if user_data and mail1 == user_data['email'] and password1 == user_data['password']:
                return render_template("index.html")
            else:
                return render_template("signin.html")
    except Exception as e:
        # Handle exceptions if needed
        print(f"An error occurred: {e}")
        return render_template("signin.html")

@app.route('/index')
def index():
   return render_template('index.html')

# @app.route('/about')
# def about():
# 	return render_template('about.html')

# @app.route('/notebook')
# def notebook():
# 	return render_template('NOTEBOOK.html')

# Define the path to the static folder
import os
static_path = os.path.join(app.root_path, 'static')

@app.route('/predict',methods = ['POST'])
def predict():
    nm = request.form['nm']

    #**************** FUNCTIONS TO FETCH DATA ***************************
    def get_historical(quote):
        end = datetime.now()
        start = datetime(end.year-2,end.month,end.day)
        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)
        df.to_csv(''+quote+'.csv')
        if(df.empty):
            ts = TimeSeries(key='YWXJYZG1ARJV8KNJ',output_format='pandas') #N6A6QT6IBFJOPJ70 Coins Ticker which are not avaiable in this API can't use for analysis
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
            #Format df
            #Last 2 yrs rows => 502, in ascending order => ::-1
            data=data.head(503).iloc[::-1]
            data=data.reset_index()
            #Keep Required cols only
            df=pd.DataFrame()
            df['Date']=pd.to_datetime(data['Date'])
            df['Open']=data['1. Open']
            df['High']=data['2. High']
            df['Low']=data['3. Low']
            df['Close']=data['4. Close']
            df['Adj Close']=data['5. Adjusted Close']
            df['Volume']=data['6. Volume']
            df.to_csv(os.path.join(""+ quote + ".csv"), index=False)
        return


    import matplotlib.pyplot as plt

#******************** LSTM SECTION ********************
    def LSTM_ALGO(df):
        dataset_train = df.iloc[0:int(0.8 * len(df)), :]
        dataset_test = df.iloc[int(0.8 * len(df)):, :]

        sc = MinMaxScaler()
        training_set = df.iloc[:, 4:5].values
        training_set_scaled = sc.fit_transform(training_set)

        X_train = []
        y_train = []
        for i in range(7, len(training_set_scaled)):
            X_train.append(training_set_scaled[i - 7:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        #preprocessing testing data
        test = dataset_test.iloc[:, 4:5].values 
        scaler = MinMaxScaler() 
        test_scaled = scaler.fit_transform(test)
        timesteps = 7
        X_test = []
        y_test = []
        for i in range(timesteps, test.shape[0]):
            X_test.append(test_scaled[i-timesteps:i, 0]) 
            y_test.append(test_scaled[i, 0]) 
        X_test, y_test = np.array(X_test), np.array(y_test)

        #Model building
        regressor = Sequential()
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='relu'))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50, return_sequences=True, activation='relu'))
        regressor.add(Dropout(0.25))
        regressor.add(LSTM(units=50, return_sequences=True, activation='relu'))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50, activation='relu'))
        regressor.add(Dropout(0.25))
        regressor.add(Dense(units=1))


        optimizer = Adam(learning_rate=0.001)

        regressor.compile(optimizer=optimizer, loss='mean_squared_error')
        regressor.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

        real_stock_price = dataset_test.iloc[:, 4:5].values
        combine = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
        test_inputs = combine[len(combine) - len(dataset_test) - 7:].values
        test_inputs = test_inputs.reshape(-1, 1)
        test_inputs = sc.transform(test_inputs)
        X_test = []
        for i in range(7, dataset_test.shape[0] + 7):
            X_test.append(test_inputs[i - 7:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        lstm_pred=predicted_stock_price[len(predicted_stock_price)-1]
        error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        mae_lstm = mean_absolute_error(real_stock_price, predicted_stock_price)
        mse_lstm = mean_squared_error(real_stock_price, predicted_stock_price)

       

        # Plotting
        plt.figure(figsize=(16, 8))
        plt.plot(df.index[-600:], df['Open'].tail(600), color='green', label='Train Stock Price')
        plt.plot(dataset_test.index, real_stock_price, color='red', label='Real Stock Price')
        plt.plot(dataset_test.index, predicted_stock_price, color='blue', label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        plt.savefig('static/LSTM.png')
        plt.close()

        print()
        print("##############################################################################")
        print("Tomorrow's ",quote," Closing Price Prediction by LSTM: ", lstm_pred)
        print("LSTM RMSE:", error_lstm)
        print("LSTM MAE:", mae_lstm)
        print("LSTM MSE:", mse_lstm)
        print("##############################################################################")
        
        return lstm_pred, error_lstm, mae_lstm, mse_lstm

 


    #**************GET DATA ***************************************
    quote=nm
    #Try-except to check if valid stock symbol
    try:
        get_historical(quote)
    except:
        return render_template('index.html',not_found=True)
    else:
    
        #************** PREPROCESSUNG ***********************
        df = pd.read_csv(''+quote+'.csv')
        print("##############################################################################")
        print("Today's",quote,"Stock Data: ")
        today_stock=df.iloc[-1:]
        print(today_stock)
        print("##############################################################################")
        df = df.dropna()
        code_list=[]
        for i in range(0,len(df)):
            code_list.append(quote)
        df2=pd.DataFrame(code_list,columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df=df2
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('Dates')
        plt.ylabel('Open Prices')
        plt.plot(df['Date'], df['Open'], label='Open Prices')
        plt.title(f'Stock Price Trend Over Time - {quote}')
        plt.legend()
        plt.tight_layout()

        # Save the plot as an image
        trend_plot_path = 'static/Trends.png'
        plt.savefig(trend_plot_path)
        plt.close()
       
        lstm_pred, error_lstm,mae_lstm, mse_lstm = LSTM_ALGO(df)
        # print(bilstm_mae, bilstm_mse, bilstm_rmse, bilstm_r2)
        print("Forecasted Prices for Next 7 days:")
        
        today_stock=today_stock.round(2)
        return render_template('result.html',quote=quote,
                               open_s=today_stock['Open'].to_string(index=False),
                               close_s=today_stock['Close'].to_string(index=False),
                               adj_close=today_stock['Adj Close'].to_string(index=False),
                               high_s=today_stock['High'].to_string(index=False),
                               low_s=today_stock['Low'].to_string(index=False),
                               vol=today_stock['Volume'].to_string(index=False),
                               lstm_pred=np.round(lstm_pred, 2), 
                               error_lstm=np.round(error_lstm, ),
                               mae_lstm=np.round(mae_lstm,2),
                               mse_lstm=np.round(mse_lstm,2)

                            )
if __name__ == '__main__':
   app.run(host='localhost',debug=True)