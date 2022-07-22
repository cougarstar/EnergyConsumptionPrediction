from tkinter.tix import Select
from torch import lstm
from energyfeature import *

from enum import Enum
import os

import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from prophet import Prophet

from sklearn import linear_model

lstfolders = ["images", "metric", "output", "input"]


            
lstmodelcol = [
                ['day', 'denoised_avg_energy', 'temperatureMax', \
                'dewPoint', 'visibility', 'humidity','windSpeed', 'uvIndex', 'weekday', 'holiday_ind', \
                'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'sin_day', 'cos_day'],
                ['day', 'denoised_avg_energy', 'weather_cluster', 'weekday', 'holiday_ind', \
                'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'sin_day', 'cos_day'],
                ['day', 'denoised_avg_energy', 'weather_cluster0', 'weather_cluster1', 'weekday', 'holiday_ind', \
                'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'sin_day', 'cos_day'],
                ['day', 'denoised_avg_energy', 'weather_encoded3_0', 'weather_encoded3_1', 'weather_encoded3_2', 'weekday', 'holiday_ind', \
                'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'sin_day', 'cos_day'],
                ['day', 'denoised_avg_energy', 'weather_encoded4_0', 'weather_encoded4_1', 'weather_encoded4_2', 'weather_encoded4_3', 'weekday', \
                'holiday_ind', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'sin_day', 'cos_day'],
                ['day', 'denoised_avg_energy', 'weather_encoded5_0', 'weather_encoded5_1', 'weather_encoded5_2', 'weather_encoded5_3', 'weather_encoded5_4', 'weekday', \
                'holiday_ind', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'sin_day', 'cos_day'],
            ]
lstmodelcol = [
                ['day', 'denoised_avg_energy','weekday', 'holiday_ind', \
                'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'sin_day', 'cos_day'],
                
            ]
           
lstmodelcol = [
                ['day', 'denoised_avg_energy', 'temperatureMax', \
                'dewPoint', 'visibility', 'humidity','windSpeed', 'uvIndex'],
                ['day', 'denoised_avg_energy', 'weather_cluster'],
                ['day', 'denoised_avg_energy', 'weather_cluster0', 'weather_cluster1'],
                ['day', 'denoised_avg_energy', 'weather_encoded3_0', 'weather_encoded3_1', 'weather_encoded3_2'],
                ['day', 'denoised_avg_energy', 'weather_encoded4_0', 'weather_encoded4_1', 'weather_encoded4_2', 'weather_encoded4_3'],
                ['day', 'denoised_avg_energy', 'weather_encoded5_0', 'weather_encoded5_1', 'weather_encoded5_2', 'weather_encoded5_3', 'weather_encoded5_4'],
            ]

lstnpast = [7, 10, 14]

def calculate_metric(train_pred, y_train, \
                        test_pred, y_test):
    """
        This function calculates Rmse and Mape 
            for train and test datasets.
    """
    
    train_rmse = np.sqrt(mean_squared_error(train_pred, y_train)).round(2)
    test_rmse = np.sqrt(mean_squared_error(test_pred, y_test)).round(2)

    train_mae = mean_absolute_error(train_pred, y_train).round(2)
    test_mae = mean_absolute_error(test_pred, y_test).round(2)

    train_mape = np.round(np.mean(np.abs(train_pred - y_train)/y_train) * 100, 2)
    test_mape = np.round(np.mean(np.abs(test_pred - y_test)/y_test) * 100, 2)
    return train_rmse, test_rmse, train_mae, test_mae, train_mape, test_mape

def plot_predict(train_pred, test_pred, 
                train_origin, test_origin, 
                date_train, date_test,                
                bRetrain, strmodel,
                modelID, stage, n_past = 0):
    """
        This function plots prediction of test data 
            with and without incremental training.

    """

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.xaxis_date()
    ax.plot(date_train[-100:], train_origin[-100:])
    ax.plot(date_train[-100:], train_pred[-100:])

    # ax.plot(self.get_date_train(), train_origin)
    # ax.plot(self.get_date_train(), train_pred)

    ax.plot(date_test, test_origin)
    ax.plot(date_test, test_pred)

    ax.set_xlabel('Date')
    ax.set_ylabel('Energy Consumption')

    plt.xticks( \
            range(0,len(date_test) + 100, 10), \
            rotation=45)

    if (bRetrain):
        plt.title('Prediction of Energy Consumption With Incremental Training: ' + \
            strmodel + "_" + str(modelID))
    else:
        plt.title('Prediction of Energy Consumption: ' + \
            strmodel + "_" + str(modelID))

    plt.legend(['Train','Train_Pred', 'Test', 'Test_Pred'], loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    if n_past: 
        plt.savefig(os.path.join("images", strmodel + '_stage' + str(stage) + "_" + 'past' + str(n_past) + '_' + str(modelID) + '.png'), bbox_inches='tight')
    else:
        plt.savefig(os.path.join("images", strmodel + '_stage' + str(stage) + '_' + str(modelID) + '.png'), bbox_inches='tight')
    #plt.show()

def checkfolder(mydir):
    """
        This function checks whether mydir folder exists,
            creates it if not.
    """

    if not os.path.isdir(mydir):
        os.makedirs(mydir)

class SelectedModel(Enum):
    """
        This class defines a list of Enums 
            of various time series models.
    """

    LSTM = 1
    GRU = 2
    CNN = 3
    UNI_PROPHET  = 4
    MULTI_PROPHET = 5
    SARIMA = 6
    SARIMAX = 7
    LINEAR_REGRESSION = 8

class EnergyModel:
    """
        This class loads preprocessed data, 
            sets up time series model,
            fits the model on the data,
            predicts test dataset
    """

    def __init__(self):

        # check whether folders in lstfolders exist
        # create them if not.
        for folder in lstfolders:
            checkfolder(folder)

        if (os.path.exists(os.path.join("input", "energycombined.csv"))):
            self.df_weather_energy = pd.read_csv(os.path.join("input", "energycombined.csv"))
        else:
            print("Execute energyfeature.py first \
                and create energycombined.csv")
            quit()

        self.pred_results = []

        self.stage1_train = None
        self.stage1_test = None

        self.get_df_weather_energy()['day'] = pd.to_datetime(self.get_df_weather_energy()['day'])
        self.get_df_weather_energy().set_index('day', inplace=True)
        
        self.plot_temperature()
       
        self.plot_weather(self.get_df_weather_energy().humidity, 
                        self.get_df_weather_energy().avg_energy, 
                        'Humidity', 'Average Energy/Household', 
                        'Energy Consumption Vs Humidity')
        
        self.plot_weather(self.get_df_weather_energy().cloudCover, 
                        self.get_df_weather_energy().avg_energy, 
                        'Cloud Cover', 'Average Energy/Household', 
                        'Energy Consumption Vs Cloud Cover')

        self.plot_weather(self.get_df_weather_energy().visibility, 
                        self.get_df_weather_energy().avg_energy, 
                        'Visibility', 'Average Energy/Household', 
                        'Energy Consumption Vs Visibility')

                    
        self.plot_weather(self.get_df_weather_energy().windSpeed, 
                        self.get_df_weather_energy().avg_energy, 
                        'Wind Speed', 'Average Energy/Household', 
                        'Energy Consumption Vs Wind Speed')

        self.plot_weather(self.get_df_weather_energy().uvIndex, 
                        self.get_df_weather_energy().avg_energy, 
                        'UV Index', 'Average Energy/Household', 
                        'Energy Consumption Vs UV Index')

        self.plot_weather(self.get_df_weather_energy().dewPoint, 
                        self.get_df_weather_energy().avg_energy, 
                        'Dew Point', 'Average Energy/Household', 
                        'Energy Consumption Vs Dew Point')

        self.get_df_weather_energy().reset_index(inplace = True)

        self.plot_businessday()
        self.df_weather_energy = pd.read_csv(os.path.join("input", "energycombined.csv"))
        
    def get_n_past(self):
        return self.n_past 

    def set_n_past(self, n_past):
        self.n_past = n_past
    
    def get_date_train(self):
        return self.date_train 

    def set_date_train(self, date_train):
        self.date_train = date_train

    def get_date_test(self):
        return self.date_test 

    def set_date_test(self, date_test):
        self.date_test = date_test

    def get_scaler(self):
        return self.scaler 

    def set_scaler(self, scaler):
        self.scaler = scaler   

    def set_pred_results(self, pred_results):
        self.pred_results = pred_results

    def get_pred_results(self):
        return self.pred_results

    def set_str_model(self, str_model):
        self.str_model = str_model

    def get_str_model(self):
        return self.str_model

    def get_retrain(self):
        return self.bRetrain

    def set_retrain(self, bRetrain):
        self.bRetrain = bRetrain

    def get_train_model(self):
        return self.train_model

    def set_train_model(self, train_model):
        self.train_model = train_model

    def get_df_weather_energy(self):
        return self.df_weather_energy

    def set_df_weather_energy(self, df_weather_energy):
        self.df_weather_energy = df_weather_energy

    def get_df_weather_energy1(self):
        return self.df_weather_energy1

    def set_df_weather_energy1(self, df_weather_energy1):
        self.df_weather_energy1 = df_weather_energy1

    def set_X_train(self, X_train):
        self.X_train = X_train

    def get_X_train(self):
        return self.X_train

    def set_X_test(self, X_test):
        self.X_test = X_test

    def get_X_test(self):
        return self.X_test

    def set_y_train(self, y_train):
        self.y_train = y_train

    def get_y_train(self):
        return self.y_train

    def set_y_test(self, y_test):
        self.y_test = y_test

    def get_y_test(self):
        return self.y_test

    def set_trainX(self, trainX):
        self.trainX = trainX

    def get_trainX(self):
        return self.trainX

    def set_trainY(self, trainY):
        self.trainY = trainY

    def get_trainY(self):
        return self.trainY

    def get_prophet_train(self):
        return self.prophet_train

    def get_prophet_test(self):
        return self.prophet_test

    def get_sarima_train(self):
        return self.sarima_train

    def get_sarima_test(self):
        return self.sarima_test

    def get_scaling_dimension(self):
        return self.scaling_dimension

    def set_scaling_dimension(self, scaling_dimension):
        self.scaling_dimension = scaling_dimension

    def get_sarima_model_fit(self):
        return self.sarima_model_fit

    def set_ModelID(self, ModelID):
        self.ModelID = ModelID

    def get_ModelID(self):
        return self.ModelID

    def set_stage1_train(self, stage1_train):
        self.stage1_train = stage1_train

    def get_stage1_train(self):
        return self.stage1_train

    def set_stage1_test(self, stage1_test):
        self.stage1_test = stage1_test

    def get_stage1_test(self):
        return self.stage1_test

    # function for differencing
    def difference(self, dataset, interval):

        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset.iloc[i] - dataset.iloc[i - interval]
            diff.append(value)
        return diff

    def adfuller(self, dataset):

        t = sm.tsa.adfuller(dataset, autolag='AIC')
        print(pd.Series(t[0:4], \
                index=['Test Statistic','p-value', \
                '#Lags Used','Number of Observations Used']))

    def setup_model_data(self, lstmodelcol, selectedmodel):
        """
            This method makes a copy of dataframe 
                based on predefined columns from weather_energy,
                sets a minmax scaler, 
                transforms on the new dataframe,
                formats data for each day,
                split the data into train and test set
        """

        self.set_df_weather_energy1( \
            self.get_df_weather_energy()[lstmodelcol].copy())

        if (selectedmodel == SelectedModel.LSTM \
            or selectedmodel == SelectedModel.GRU \
                or selectedmodel == SelectedModel.CNN):

            date_df = self.get_df_weather_energy1()['day']
            self.set_df_weather_energy1( \
                self.get_df_weather_energy1().set_index('day'))
            
            self.set_scaler(MinMaxScaler()) #StandardScaler()
            self.set_scaler(self.get_scaler().fit(self.get_df_weather_energy1()))
            self.set_scaling_dimension(self.get_df_weather_energy1().shape[1])
            weather_energy1_scaled = self.get_scaler().transform(self.get_df_weather_energy1())

            dates, trainX, trainY = self.format_data(weather_energy1_scaled, date_df)

            self.set_trainX(trainX)
            self.set_trainY(trainY)

            q_90 = int(len(trainX) * .9)

            self.set_date_train(dates[:q_90])
            self.set_date_test(dates[q_90:])

            self.set_X_train(trainX[:q_90])
            self.set_y_train(trainY[:q_90])

            self.set_X_test(trainX[q_90:])
            self.set_y_test(trainY[q_90:])

        else:
            self.set_scaler(MinMaxScaler()) #StandardScaler()
            # print("Scaling: \n", self.get_df_weather_energy1()[lstmodelcol[1:]])
            if (selectedmodel == SelectedModel.LINEAR_REGRESSION):
                self.set_scaler(self.get_scaler().fit(self.get_df_weather_energy1()[lstmodelcol[2:]]))
            else:    
                self.set_scaler(self.get_scaler().fit(self.get_df_weather_energy1()[lstmodelcol[1:]]))

            self.set_scaling_dimension(self.get_df_weather_energy1().shape[1] - 1)

            date_df = self.get_df_weather_energy1()['day']
            #self.set_df_weather_energy1( \
            #    self.get_df_weather_energy1().set_index('day'))

            q_90 = int((len(self.get_df_weather_energy1()) - self.get_n_past()) * .9)
            dates = np.array(date_df[self.get_n_past():])
            self.set_date_train(dates[:q_90])
            self.set_date_test(dates[q_90:])
            
            if (selectedmodel == SelectedModel.LINEAR_REGRESSION):
                self.get_df_weather_energy1()[lstmodelcol[2:]] = \
                            self.get_scaler().transform(self.get_df_weather_energy1()[lstmodelcol[2:]])
            else:
                self.get_df_weather_energy1()[lstmodelcol[1:]] = \
                            self.get_scaler().transform(self.get_df_weather_energy1()[lstmodelcol[1:]])
            
            if (selectedmodel == SelectedModel.UNI_PROPHET \
                or selectedmodel == SelectedModel.MULTI_PROPHET):

                self.set_df_weather_energy1(self.get_df_weather_energy1().rename( \
                                            columns={"day": "ds", "denoised_avg_energy": "y"}))

                self.prophet_train = self.get_df_weather_energy1().iloc[self.get_n_past():q_90 + self.get_n_past()]
                self.prophet_test = self.get_df_weather_energy1().iloc[q_90 + self.get_n_past():]
    
            elif (selectedmodel == SelectedModel.LINEAR_REGRESSION):
                self.X_train = self.get_df_weather_energy1()[lstmodelcol[2:]].iloc[self.get_n_past():q_90 + self.get_n_past()]
                self.y_train = self.get_df_weather_energy1()[lstmodelcol[1]].iloc[self.get_n_past():q_90 + self.get_n_past()]
                self.X_test = self.get_df_weather_energy1()[lstmodelcol[2:]].iloc[q_90 + self.get_n_past():]
                self.y_test = self.get_df_weather_energy1()[lstmodelcol[1]].iloc[q_90 + self.get_n_past():]

            else:
                self.get_df_weather_energy1().set_index(['day'],inplace=True)
                self.sarima_train = self.get_df_weather_energy1().iloc[self.get_n_past():q_90 + self.get_n_past()]
                self.sarima_test = self.get_df_weather_energy1().iloc[q_90 + self.get_n_past():]

                plot_acf(self.sarima_train[lstmodelcol[1]],lags=100)
                plt.show()

                plot_pacf(self.sarima_train[lstmodelcol[1]],lags=50)
                plt.show()

                self.adfuller(self.sarima_train[lstmodelcol[1]])
                self.adfuller(self.difference(self.sarima_train[lstmodelcol[1]], 1))

                s = sm.tsa.seasonal_decompose(self.sarima_train[lstmodelcol[1]], freq=12)
                
                fig = s.plot()
                
                fig.set_size_inches((12, 9))
                # Tight layout to realign things
                fig.tight_layout()
                plt.savefig("images/decompose.png", bbox_inches='tight')                

    def setup_training_model(self, lstmodelcol, selectedmodel):
        """
            This method sets up training model, 
                fits the model onto the preprocessed data
        """

        if selectedmodel == SelectedModel.LSTM:
            self.set_str_model("LSTM")            
            self.set_train_model(self.setup_LSTM())
        elif selectedmodel == SelectedModel.GRU:
            self.set_str_model("GRU")
            self.set_train_model(self.setup_GRU())
        elif selectedmodel == SelectedModel.CNN:
            self.set_str_model("CNN")
            self.set_train_model(self.setup_CNN())
        elif selectedmodel == SelectedModel.UNI_PROPHET:
            self.set_str_model("ProphetUnivariate")
            self.set_train_model(self.setup_univ_prophet())
        elif selectedmodel == SelectedModel.MULTI_PROPHET:
            self.set_str_model("ProphetMultivariate")
            self.set_train_model(self.setup_multi_prophet(lstmodelcol))
        elif selectedmodel == SelectedModel.SARIMA:
            self.set_str_model("SARIMA")
            self.set_train_model(self.setup_sarima(lstmodelcol))
        elif selectedmodel == SelectedModel.SARIMAX:
            self.set_str_model("SARIMAX")
            self.set_train_model(self.setup_sarimax(lstmodelcol))
        elif selectedmodel == SelectedModel.LINEAR_REGRESSION:
            self.set_str_model("LinearRegression")
            self.set_train_model(self.setup_linReg(lstmodelcol))
        
        if (selectedmodel == SelectedModel.LSTM \
            or selectedmodel == SelectedModel.GRU \
                or selectedmodel == SelectedModel.CNN):
            # fit the model
            history = self.get_train_model().fit(self.get_X_train(), \
                                                self.get_y_train(), epochs=500, verbose=0)

            #plt.plot(history.history['loss'], label='Training loss')

            #plt.legend()

        elif (selectedmodel == SelectedModel.SARIMA \
            or selectedmodel == SelectedModel.SARIMAX):
            self.sarima_model_fit = self.get_train_model().fit()
            self.get_sarima_model_fit().summary()

        elif (selectedmodel == SelectedModel.LINEAR_REGRESSION):
            self.get_train_model().fit(self.get_X_train(), self.get_y_train())

        else:
            self.get_train_model().fit(self.get_prophet_train())

    def setup_linReg(self, lstmodelcol):

        self.set_retrain(False)
        model = linear_model.LinearRegression()
        return model

    def setup_sarima(self, lstmodelcol):

        self.set_retrain(False)
        endog = self.get_sarima_train()[lstmodelcol[1]]
        model = sm.tsa.statespace.SARIMAX(endog=endog, \
                order=(7,1,1),seasonal_order=(1,1, 0, 12),trend='c')
        return model

    def setup_sarimax(self, lstmodelcol):

        self.set_retrain(False)
        endog = self.get_sarima_train()[lstmodelcol[1]]
        exog = self.get_sarima_train()[lstmodelcol[2:]]
        model = SARIMAX(endog=endog, exog=exog, \
                order=(7,1,1),seasonal_order=(1,1, 0, 12),trend='c')
        return model


    def setup_univ_prophet(self):

        self.set_retrain(False)
        model = Prophet()
        return model

    def setup_multi_prophet(self, lstmodelcol):

        self.set_retrain(False)
        model = Prophet()
        for modelcol in lstmodelcol[2:]:
            model.add_regressor(modelcol)
        return model

    def setup_LSTM(self):

        self.set_retrain(True)
        model = Sequential()
        model.add(LSTM(128, activation='relu', \
                input_shape=(self.get_trainX().shape[1], \
                self.get_trainX().shape[2]), return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(8, 'relu'))
        model.add(Dense(self.get_trainY().shape[1], activation='linear'))

        model.compile(optimizer=Adam(learning_rate=0.002), loss='mse')
        model.summary()
        return model

    def setup_GRU(self):

        self.set_retrain(True)
        model = Sequential()
        model.add(GRU(128, activation='relu', \
                input_shape=(self.get_trainX().shape[1], \
                self.get_trainX().shape[2]), return_sequences=True))
        model.add(GRU(64, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(8, 'relu'))
        model.add(Dense(self.get_trainY().shape[1], activation='linear'))

        model.compile(optimizer=Adam(learning_rate=0.002), loss='mse')
        model.summary()
        return model

    def setup_CNN(self):

        self.set_retrain(True)
        model = Sequential()

        model.add(Conv1D(filters=64, \
                kernel_size=2, activation='relu', \
                input_shape=(self.get_trainX().shape[1], \
                self.get_trainX().shape[2])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())

        model.add(Dropout(0.2))
        model.add(Dense(8, 'relu'))
        model.add(Dense(self.get_trainY().shape[1], \
                    activation='linear'))

        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model


    def format_data(self, weather_energy1_scaled, date_df):
        """
            This method configures data to appropriate format 
                for recurrent neural network
        """
        #Empty lists to be populated using formatted training data
        trainX = []
        trainY = []

        n_future = 1   # Number of days we want to look into the future based on the past days.
        n_past = self.get_n_past()  # Number of past days we want to use to predict the future.

        #Reformat input data into a shape: (n_samples x timesteps x n_features)
        #In my example, my df_for_training_scaled has a shape (12823, 5)
        #12823 refers to the number of data points and 5 refers to the columns (multi-variables).
        for i in range(n_past, len(weather_energy1_scaled) - n_future +1):
            trainX.append(weather_energy1_scaled[i - n_past:i, 0:weather_energy1_scaled.shape[1]])
            trainY.append(weather_energy1_scaled[i + n_future - 1:i + n_future, 0])

        return np.array(date_df[n_past:]), np.array(trainX), np.array(trainY)


    def model_predict(self, 
                        lstmodelcol, 
                        selectedmodel,
                        featureset):
        """
            This method makes prediction on train and test datasets,
                updates prediction metric,
                plots prediction graphs,
                saves prediction output for stage 2 use.
        """

        if (selectedmodel == SelectedModel.LSTM \
            or selectedmodel == SelectedModel.GRU \
                or selectedmodel == SelectedModel.CNN):
        
            train_pred = self.get_train_model().predict(self.get_X_train())
            test_pred = self.get_train_model().predict(self.get_X_test())

            if self.get_retrain():
                test_retrain_pred = self.predict_test_retrain(self.get_X_test(), self.get_y_test())

            train_pred = self.inverse_transform_data( \
                train_pred, self.get_scaling_dimension())
            test_pred = self.inverse_transform_data( \
                test_pred, self.get_scaling_dimension())
            train_origin = self.inverse_transform_data( \
                self.get_y_train(), self.get_scaling_dimension())
            test_origin = self.inverse_transform_data( \
                self.get_y_test(), self.get_scaling_dimension())

            self.update_metric(train_pred, 
                            train_origin, test_pred, 
                            test_origin, False, featureset)

            self.plot_prediction(train_pred, test_pred, train_origin, test_origin, False)

            if self.get_retrain():
                test_retrain_pred = self.inverse_transform_data( \
                    test_retrain_pred, self.get_scaling_dimension())

                self.update_metric(train_pred, train_origin, 
                                test_retrain_pred, test_origin, 
                                True, featureset)


                self.plot_prediction(train_pred, test_retrain_pred, \
                                train_origin, test_origin, True)

            if self.get_retrain():
                if self.get_stage1_train() is None:                    
                    self.initialize_stage1_output(train_origin, test_origin)
                
                self.get_stage1_train()['Model' + str(self.get_ModelID())] = train_pred
                self.get_stage1_test()['Model' + str(self.get_ModelID())] = test_retrain_pred


        elif selectedmodel == SelectedModel.MULTI_PROPHET \
            or selectedmodel == SelectedModel.UNI_PROPHET:

            future_data = self.get_train_model().make_future_dataframe( \
                                            periods=len(self.get_prophet_test()), freq = 'D')
            
            if selectedmodel == SelectedModel.MULTI_PROPHET:
                df = self.get_prophet_train()
                df = df.append(self.get_prophet_test())
                df = df.rename( \
                    columns={'denoised_avg_energy': 'y', 'day':'ds'})

                future_data = df[['ds'] + lstmodelcol[2:]]


            forecast_data = self.get_train_model().predict(future_data)

            pred = forecast_data['yhat'].values.reshape((forecast_data.shape[0], 1))

            y_train = self.get_prophet_train()['y'].values.reshape( \
                                            (self.get_prophet_train()['y'].shape[0], 1))
            y_test = self.get_prophet_test()['y'].values.reshape( \
                                            (self.get_prophet_test()['y'].shape[0], 1))

            train_pred = pred[:self.get_prophet_train()['y'].shape[0]]
            test_pred = pred[self.get_prophet_train()['y'].shape[0]:]

            self.get_train_model().plot(forecast_data)

            self.get_train_model().plot_components(forecast_data)

            train_pred = self.inverse_transform_data( \
                train_pred, self.get_scaling_dimension())
            test_pred = self.inverse_transform_data( \
                test_pred, self.get_scaling_dimension())
            train_origin = self.inverse_transform_data( \
                y_train, self.get_scaling_dimension())
            test_origin = self.inverse_transform_data( \
                y_test, self.get_scaling_dimension())

            self.update_metric(train_pred[self.get_n_past():], train_origin[self.get_n_past():], \
                                test_pred, test_origin, False, featureset)

            if self.get_stage1_train() is None:
                self.initialize_stage1_output(train_origin, test_origin)

            self.get_stage1_train()['Model' + str(self.get_ModelID())] = train_pred
            self.get_stage1_test()['Model' + str(self.get_ModelID())] = test_pred

            self.plot_prediction(train_pred, test_pred, \
                train_origin, test_origin, False)

        elif selectedmodel == SelectedModel.LINEAR_REGRESSION:
            
            train_pred = self.get_train_model().predict(self.get_X_train())
            test_pred = self.get_train_model().predict(self.get_X_test())

            self.update_metric(train_pred, 
                            self.get_y_train(), test_pred, 
                            self.get_y_test(), False, featureset)

            if self.get_stage1_train() is None:
                self.initialize_stage1_output(self.get_y_train(), self.get_y_test())

            self.get_stage1_train()['Model' + str(self.get_ModelID())] = train_pred
            self.get_stage1_test()['Model' + str(self.get_ModelID())] = test_pred

            self.plot_prediction(train_pred, test_pred, self.get_y_train(), self.get_y_test(), False)

        else:
            if selectedmodel == SelectedModel.SARIMA:
                train_pred = self.get_sarima_model_fit().predict( \
                    start = 0, 
                    end = len(self.get_sarima_train()) - 1).values

                test_pred = self.get_sarima_model_fit().predict( \
                    start = len(self.get_sarima_train()), 
                    end = len(self.get_sarima_train()) + len(self.get_sarima_test()) - 1).values
                
            else:
                train_pred = self.get_sarima_model_fit().predict( \
                    start = 0, 
                    end = len(self.get_sarima_train()) - 1, \
                    exog = self.get_sarima_train()[lstmodelcol[2:]]).values

                test_pred = self.get_sarima_model_fit().predict( \
                    start = len(self.get_sarima_train()), 
                    end = len(self.get_sarima_train()) + len(self.get_sarima_test()) - 1, \
                    exog = self.get_sarima_test()[lstmodelcol[2:]]).values
                
            train_pred = train_pred.reshape((train_pred.shape[0], 1))
                
            test_pred = test_pred.reshape((test_pred.shape[0], 1))

            y_train = self.get_sarima_train()[lstmodelcol[1]].values.reshape( \
                                (self.get_sarima_train()[lstmodelcol[1]].shape[0], 1))
            y_test = self.get_sarima_test()[lstmodelcol[1]].values.reshape( \
                                (self.get_sarima_test()[lstmodelcol[1]].shape[0], 1))


            print(self.get_sarima_train().columns)

            train_pred = self.inverse_transform_data( \
                train_pred, self.get_scaling_dimension())
            test_pred = self.inverse_transform_data( \
                test_pred, self.get_scaling_dimension())
            train_origin = self.inverse_transform_data( \
                y_train, self.get_scaling_dimension())
            test_origin = self.inverse_transform_data( \
                y_test, self.get_scaling_dimension())

            self.update_metric(train_pred[self.get_n_past():], \
                train_origin[self.get_n_past():], test_pred, test_origin, False, featureset)

            if self.get_stage1_train() is None:
                self.initialize_stage1_output(train_origin, test_origin)

            self.get_stage1_train()['Model' + str(self.get_ModelID())] = train_pred
            self.get_stage1_test()['Model' + str(self.get_ModelID())] = test_pred

            self.plot_prediction(train_pred, test_pred, \
                train_origin, test_origin, False)

    def initialize_stage1_output(self, train_origin, test_origin):
        """
            This method sets up stage1 output dataframe
        """

        self.set_stage1_train(pd.DataFrame(train_origin, \
                                            index=self.get_date_train(), columns=['y']))
        self.get_stage1_train()['date'] = self.get_stage1_train().index                                    
        self.set_stage1_test(pd.DataFrame(test_origin, \
                                            index=self.get_date_test(), columns=['y']))
        self.get_stage1_test()['date'] = self.get_stage1_test().index

    def predict_test_retrain(self, X_test, y_test):
        """
            This method predicts test dataset 
                in an incremental manner
        """

        self.get_train_model().compile(optimizer=Adam(learning_rate=0.001), \
                                    loss='mse')
        test_pred = np.array([])
        arr_prev_feature = np.array([])
        arr_curr_output = np.array([])
        fit_size = 1
        count = 0
        for a, b in zip(X_test, y_test):

            prev_feature = a.reshape(1, a.shape[0], a.shape[1])
            curr_output = b.reshape(1, b.shape[0])
            #if count % fit_size != 0 and count != 0:
            arr_prev_feature = np.append(arr_prev_feature, prev_feature)
            arr_curr_output = np.append(arr_curr_output, curr_output)

            curr_pred = self.get_train_model().predict(prev_feature)
            test_pred = np.append(test_pred, curr_pred[0][0])

            if fit_size == 1:
                #print(count)

                self.get_train_model().fit(arr_prev_feature.reshape(fit_size, a.shape[0], a.shape[1]), \
                                        arr_curr_output.reshape(fit_size, b.shape[0]), 
                                        verbose = 0, 
                                        epochs=10)

                arr_prev_feature = np.array([])
                arr_curr_output = np.array([])
            else:
                if (count % fit_size == (fit_size - 1) and (count != 0)):
                    print(count)

                    self.get_train_model().fit(arr_prev_feature.reshape(fit_size, a.shape[0], a.shape[1]), \
                                        arr_curr_output.reshape(fit_size, b.shape[0]), 
                                        verbose = 0, 
                                        epochs=10)
                                        
                    arr_prev_feature = np.array([])
                    arr_curr_output = np.array([])

            count += 1

        test_pred = test_pred.reshape(test_pred.shape[0], 1)
        return test_pred

    def update_metric(self, train_pred, y_train, \
                        test_pred, y_test, bCurrRetrain, featureset):
        """
            This method calculates RMSE, MAE, and MAPE for prediction of 
                train and test datasets, and then updates 
                the result metric dataframe.
        """
        
        # y_test = y_test.reshape((y_test.shape[0],))
        train_rmse, test_rmse, train_mae, test_mae, train_mape, test_mape = \
            calculate_metric(train_pred, y_train, \
                        test_pred, y_test)
        
        ModelID = len(self.get_pred_results()) + 1
        self.set_ModelID(ModelID)

        tmpresults = pd.DataFrame({'ModelID': [ModelID], \
                            'Model':[self.get_str_model()], \
                            'Feature': [featureset], \
                            'Past': [self.get_n_past()], \
                            'Retrain': [bCurrRetrain], \
                            'Train MAPE': [train_mape], \
                            'Train RMSE': [train_rmse], \
                            'Train MAE': [train_mae], \
                            'Test MAPE': [test_mape], \
                            'Test RMSE': [test_rmse], \
                            'Test MAE': [test_mae]})
            
        if ModelID == 1:
            self.set_pred_results(tmpresults[['ModelID', 'Model', 'Feature', 'Past', 'Retrain', \
                        'Train MAPE', 'Train RMSE', 'Train MAE', 'Test MAPE', 'Test RMSE', 'Test MAE']])
        else:
            self.set_pred_results(pd.concat([self.get_pred_results(), tmpresults]))
            self.set_pred_results(self.get_pred_results()[['ModelID', 'Model', 'Feature', 'Past', \
                        'Retrain', 'Train MAPE', 'Train RMSE', 'Train MAE', \
                        'Test MAPE', 'Test RMSE', 'Test MAE']])


    def inverse_transform_data(self, \
                                y_pred, \
                                scaling_dimension):
        """
            This method inverse transform data 
                after the data is scaled in the preprocessing stage
        """

        prediction_copies = np.repeat( \
                        y_pred, 
                        scaling_dimension,
                        axis=-1)

        return self.get_scaler().inverse_transform(prediction_copies)[:,0]

        
    def plot_prediction(self, 
                        train_pred, 
                        test_pred, 
                        train_origin, 
                        test_origin, 
                        bRetrain):
        """
            This method plots prediction of test data 
                with and without incremental training.

        """
        
        plot_predict(train_pred, test_pred, 
                    train_origin, test_origin, 
                    self.get_date_train(), 
                    self.get_date_test(), 
                    bRetrain, 
                    self.get_str_model(),
                    self.get_ModelID(), 
                    1,
                    self.get_n_past())
        

    def plot_businessday(self):
        """
            This method plots average energy consumption 
                vs. business days including 
                weekday and holiday

        """

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            self.get_df_weather_energy().avg_energy)
        ax.plot(
            self.get_df_weather_energy().weekday)
        ax.plot(
            self.get_df_weather_energy().holiday_ind)
    
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Energy')

        plt.xticks( \
                range(0,len(self.get_df_weather_energy()), 50), \
                rotation=45)
        
        plt.title('Average Energy Consumption Vs Business Days')

        plt.legend(['Average Energy Consumption','Weekday', 'Holiday'])
        #ax.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102))
        fig.tight_layout()

        imgfile = os.path.join("images", "energy_holiday.png") 
        if (not os.path.exists(imgfile)):
            plt.savefig(imgfile, bbox_inches='tight')

    def plot_temperature(self):
        """
            This method plots average energy consumption 
                vs. maximum temperature and 
                minimum temperature
        """

        fig, ax1 = plt.subplots(figsize = (12,6))
        ax1.plot(self.get_df_weather_energy()["temperatureMax"], \
                color = 'tab:orange')
        ax1.plot(self.get_df_weather_energy().temperatureMin, \
                color = 'tab:pink')    

        ax1.set_ylabel('Temperature')
        ax1.legend()

        plt.xticks(rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(self.get_df_weather_energy().avg_energy, \
            color = 'tab:blue')
        ax2.set_ylabel('Average Energy/Household',color = 'tab:blue')
        ax2.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102))
        
        plt.title('Energy Consumption Vs Temperature')
        fig.tight_layout()

        imgfile = os.path.join("images", "energy_temperature.png")
        if (not os.path.exists(imgfile)):
            plt.savefig(imgfile, bbox_inches='tight')      

    def plot_weather(self, ax1_y, ax2_y, ax1_ylabel, ax2_ylabel, title):
        """
            This method plots average energy consumption 
                vs. weather information such as
                humidity, cloudcover, uvindex, wind speed,
                pressure, etc.
        """

        fig, ax1 = plt.subplots(figsize = (12,6))
        ax1.plot(ax1_y, color = 'tab:orange')
        ax1.set_ylabel(ax1_ylabel, color = 'tab:orange')

        plt.xticks(rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(ax2_y, color = 'tab:blue')
        ax2.set_ylabel(ax2_ylabel, color = 'tab:blue')        

        plt.title(title)
        fig.tight_layout()

        strtemp = ax1_ylabel.replace(" ", "_")
        imgfile = os.path.join("images", "energy_" + strtemp + ".png")
        if (not os.path.exists(imgfile)):
            plt.savefig(imgfile, bbox_inches='tight')

    def model_process(self, 
                    lstmodelcol, 
                    selectedmodel,
                    featureset,
                    n_past): 
        """
            This method set up model data, 
                set up training model,
                predict testing dataset.

            This method is the main entry point 
                after the object is instantiated
        """

        self.set_n_past(n_past)
        self.setup_model_data(lstmodelcol, selectedmodel)
        self.setup_training_model(lstmodelcol,selectedmodel)
        self.model_predict(lstmodelcol, selectedmodel, featureset)

        self.get_pred_results().set_index('ModelID')

        print(self.get_pred_results())

        # print(self.get_stage1_train())

        # print(self.get_stage1_test())

def runmodels(myenergy, n_past):
    """
        This function executes every model in SelectedModel            
            for every featureset in lstmodelcol

            and saves the output and prediction metric into files.

    """

    # clear stage1 output and prediction metric dataframes
    myenergy.set_stage1_train(None)
    myenergy.set_stage1_test(None)
    myenergy.set_pred_results([])

    for featureset in range(len(lstmodelcol)):
        for sModel in SelectedModel:
            myenergy.model_process( \
                lstmodelcol[featureset], 
                sModel, 
                featureset, 
                n_past)

    # save prediction metric    
    myenergy.get_pred_results().to_csv(os.path.join("metric", "Stage1_metric" + str(n_past) + ".csv"), index=False)

    # save stage1 output after merging 
    df_stage1_output = pd.concat([myenergy.get_stage1_train(), myenergy.get_stage1_test()])    
    df_stage1_output.to_csv(os.path.join("output", "Stage1_output" + str(n_past) + ".csv"), index = False)


if __name__ == '__main__':
    myenergy = EnergyModel()
    
    
    myenergy.model_process( 
                    lstmodelcol[0], 
                    SelectedModel.LINEAR_REGRESSION,
                    0,
                    7)
    
    """
    
    runmodels(myenergy, 7)

    runmodels(myenergy, 10)

    runmodels(myenergy, 14)
    """
    
    