import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from energymodel import *

class EnsembleModel(Enum):
    """
        This class defines a list of Enums 
            of various time series models.
    """

    LinearRegression = 1
    ArtificialNeuralNetwork = 2
    RandomForest = 3
    
class EnergyEnsemble:
    """
        This class defines stage 2 models 
            for energy consumption prediction
            including linear regression, 
            deep learning, and random forest
    """
    
    def __init__(self, stage1_output_file = os.path.join("output", "Stage1_output.csv")):

        self.df_stage1_output = pd.read_csv(stage1_output_file)

        self.set_df_date(self.get_df_stage1_output()['date'])

        # self.get_df_stage1_output()['date'] = \
        # pd.to_datetime(self.get_df_stage1_output()['date']).dt.date
        
        self.get_df_stage1_output().set_index('date', inplace=True)

        self.pred_results = []

        lststage1cols = self.get_df_stage1_output().columns

        self.set_scaler(MinMaxScaler())
        df_stage1_scaled = self.get_scaler().fit_transform(self.get_df_stage1_output()[lststage1cols[1:]])
        self.set_df_stage1_scaled(DataFrame(df_stage1_scaled, columns=lststage1cols[1:]))

        # print(self.get_df_stage1_scaled())

    def get_df_date(self):
        return self.df_date 

    def set_df_date(self, df_date):
        self.df_date = df_date

    def get_df_stage1_output(self):
        return self.df_stage1_output

    def set_df_stage1_output(self, df_stage1_output):
        self.df_stage1_output = df_stage1_output

    def set_scaler(self, scaler):
        self.scaler = scaler

    def get_scaler(self):
        return self.scaler

    def set_df_stage1_scaled(self, df_stage1_scaled):
        self.df_stage1_scaled = df_stage1_scaled

    def get_df_stage1_scaled(self):
        return self.df_stage1_scaled

    def set_selected_feature(self, lstfeature):
        self.selected_features = lstfeature

    def get_selected_features(self):
        return self.selected_features

    def set_X_train(self, X_train):
        self.X_train = X_train

    def get_X_train(self):
        return self.X_train    

    def get_X_test(self):
        return self.X_test

    def set_X_test(self, X_test):
        self.X_test = X_test

    def set_y_train(self, y_train):
        self.y_train = y_train

    def get_y_train(self):
        return self.y_train

    def set_y_test(self, y_test):
        self.y_test = y_test

    def get_y_test(self):
        return self.y_test

    def set_training_model(self, training_model):
        self.training_model = training_model

    def get_training_model(self):
        return self.training_model

    def set_str_model(self, str_model):
        self.str_model = str_model

    def get_str_model(self):
        return self.str_model

    def set_ModelID(self, ModelID):
        self.ModelID = ModelID

    def get_ModelID(self):
        return self.ModelID

    def set_pred_results(self, pred_results):
        self.pred_results = pred_results

    def get_pred_results(self):
        return self.pred_results

    def get_date_train(self):
        return self.date_train 

    def set_date_train(self, date_train):
        self.date_train = date_train

    def get_date_test(self):
        return self.date_test 

    def set_date_test(self, date_test):
        self.date_test = date_test

    def setup_train_test(self, lstfeature):
        """
            This method splits data into training and test datasets.
        """

        X = self.get_df_stage1_scaled()[lstfeature]
        y = self.get_df_stage1_output()['y']
        q_90 = int(0.9*X.shape[0])

        self.set_X_train(X[:q_90])
        self.set_y_train(y[:q_90])
        self.set_X_test(X[q_90:])
        self.set_y_test(y[q_90:])

        self.set_date_train(np.array(self.get_df_date()[:q_90]))
        self.set_date_test(np.array(self.get_df_date()[q_90:]))
        # print("date train: ", self.get_date_train())

    def setup_training_model(self, lstfeature, selectedmodel):
        """
            This method sets up training model,
                fit the training dataset to the model
        """

        if selectedmodel == EnsembleModel.LinearRegression:
            self.set_training_model(linear_model.LinearRegression())
            self.set_str_model("Linear Regression")
            self.get_training_model().fit(self.get_X_train(), self.get_y_train())
        elif selectedmodel == EnsembleModel.ArtificialNeuralNetwork:
            model = Sequential()
            model.add(Dense(128, input_dim=len(lstfeature), activation='relu'))
            model.add(Dense(64, activation='relu'))
            #Output layer
            model.add(Dense(1, activation='linear'))

            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
            print(model.summary())

            history = model.fit(self.get_X_train(), self.get_y_train(), epochs =100)
            self.set_training_model(model)
            self.set_str_model("Deep Learning")
        else:
            self.set_training_model(RandomForestRegressor(n_estimators = 30, random_state=30))
            self.set_str_model("Random Forest")
            self.get_training_model().fit(self.get_X_train(), self.get_y_train())
        

    def model_predict(self, selectedmodel):
        """
            This method predicts on test dataset,
                update metrics, and plot the prediction against ground truth
        """

        train_pred = self.get_training_model().predict(self.get_X_train())
        test_pred = self.get_training_model().predict(self.get_X_test())

        if selectedmodel == EnsembleModel.ArtificialNeuralNetwork:
            train_pred = train_pred.reshape((train_pred.shape[0],))
            test_pred = test_pred.reshape((test_pred.shape[0],))

        self.update_metric(train_pred, test_pred)

        plot_predict(train_pred, 
                    test_pred, 
                    self.get_y_train(), 
                    self.get_y_test(), 
                    self.get_date_train(), 
                    self.get_date_test(), 
                    False, 
                    self.get_str_model(),
                    self.get_ModelID(), 
                    2)

    def update_metric(self, train_pred, test_pred):
        """
            This method calculates RMSE and MAPE for prediction of 
                train and test datasets, and then updates 
                the result metric dataframe.
        """

        ModelID = len(self.get_pred_results()) + 1
        self.set_ModelID(ModelID)

        train_rmse, test_rmse, train_mae, test_mae, train_mape, test_mape = \
            calculate_metric(train_pred, self.get_y_train(), \
                        test_pred, self.get_y_test())

        tmpresults = pd.DataFrame({'ModelID': [ModelID], \
                            'Model':[self.get_str_model()], \
                            'Train MAPE': [train_mape], \
                            'Train RMSE': [train_rmse], \
                            'Train MAE': [train_mae], \
                            'Test MAPE': [test_mape], \
                            'Test RMSE': [test_rmse], \
                            'Test MAE': [test_mae]})


        if ModelID == 1:
            self.set_pred_results(tmpresults[['ModelID', 'Model',  \
                        'Train MAPE', 'Train RMSE', 'Train MAE', 'Test MAPE', 'Test RMSE', 'Test MAE']])
        else:
            self.set_pred_results(pd.concat([self.get_pred_results(), tmpresults]))
            self.set_pred_results(self.get_pred_results()[['ModelID', 'Model',  \
                        'Train MAPE', 'Train RMSE', 'Train MAE', 'Test MAPE', 'Test RMSE', 'Test MAE']])

    def model_process(self, lstfeature, selectedmodel):
        """
            This method sets up model data, 
                sets up training model,
                and predicts testing dataset.

            This method is the main entry point 
                after the object is instantiated
        """

        self.setup_train_test(lstfeature)
        self.setup_training_model(lstfeature, selectedmodel)
        self.model_predict(selectedmodel)

        print(self.get_pred_results())

def runensemblemodels(myenergy, lstfeature):

    for sModel in EnsembleModel:
        myenergy.model_process(lstfeature = lstfeature, \
            selectedmodel = sModel)

    myenergy.get_pred_results().to_csv(os.path.join("metric", "Stage2_metric.csv"), index = False)

if __name__ == '__main__':
    
    myenergy = EnergyEnsemble()
    lstfeature = ['Model2', 'Model4', 'Model6']
    runensemblemodels(myenergy, lstfeature)