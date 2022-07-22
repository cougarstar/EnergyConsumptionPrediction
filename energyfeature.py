from energypreprocess import *

import pandas as pd
import numpy as np

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

import sklearn
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import os

class EnergyFeature:
    """
        This class adds cluster and 
            auto encoded features to energy dataset.
    """

    lstcolcorr = ['denoised_avg_energy','temperatureMax','dewPoint', \
                    'cloudCover', 'windSpeed','pressure', 'visibility', \
                    'humidity','uvIndex', 'moonPhase']

    lstcolcluster = ['temperatureMax', 'dewPoint', \
                    'visibility', 'humidity','windSpeed', 'uvIndex']

    lstcolallweather = ['temperatureMax',
       'windBearing', 'dewPoint', 'cloudCover', 'windSpeed', 'pressure',
       'apparentTemperatureHigh', 'visibility', 'humidity',
       'apparentTemperatureLow', 'apparentTemperatureMax', 'uvIndex',
       'temperatureLow', 'temperatureMin', 'temperatureHigh',
       'apparentTemperatureMin', 'moonPhase']

    def __init__(self):
        
        if (os.path.exists(os.path.join("input", "energyweather.csv"))):
            self.df_weather_energy = pd.read_csv(os.path.join("input", "energyweather.csv"))
        else:
            self.df_weather_energy = \
                EnergyPreprocess().get_df_weather_energy()


        print("Relevant feature correlations: ")
        print(self.get_df_weather_energy()[self.lstcolcorr].corr())

        scaler = MinMaxScaler()
        weather_scaled = scaler.fit_transform(self.get_df_weather_energy()[self.lstcolcluster])

        # plot kmeans elbow to determine number of clusters
        self.plot_kmeans_elbow(weather_scaled)

        # n_clusters = 3 comes from kmeans elbow plot
        self.add_weather_cluster(weather_scaled, 3)

        print(self.get_df_weather_energy().columns)

    def get_encoded_dimension(self):
        return self.encoded_dimension 

    def set_encoded_dimension(self, encoded_dimension):
        self.encoded_dimension = encoded_dimension
    
    def get_df_weather_energy(self):
        return self.df_weather_energy

    def set_df_weather_energy(self, df_weather_energy):
        self.df_weather_energy = df_weather_energy

    def add_weather_cluster(self, weather_scaled, n_clusters):
        """
            This method adds cluster features to the dataset

            for example, if number of cluster is 3, 
                we add two sets of features:
                    one is weather_cluster which could have value 0, 1, and 2
                    the other is weather_cluster0 and weather_cluster1 
                        which are one-hot encoding
        """

        kmeans = KMeans(n_clusters=n_clusters, 
                            max_iter=600, 
                            algorithm = 'auto')
        kmeans.fit(weather_scaled)
        self.get_df_weather_energy()['weather_cluster'] = kmeans.labels_

        self.get_df_weather_energy()['weather_cluster0'] = \
            self.get_df_weather_energy()['weather_cluster'].apply(lambda x: 1 if x==0 else 0)
        self.get_df_weather_energy()['weather_cluster1'] = \
            self.get_df_weather_energy()['weather_cluster'].apply(lambda x: 1 if x==1 else 0)

    def plot_kmeans_elbow(self, weather_scaled):
        """
            This method plots kmeans elbow 
                which can be used to determine 
                number of clusters to add
        """

        ssd = []
        range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
        for num_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
            kmeans.fit(weather_scaled)
            
            ssd.append(kmeans.inertia_)
            
        # plot the SSDs for each n_clusters
        # ssd
        plt.plot(ssd)
        plt.show()

    def preprocess_encoder(self):
        """
            This method splits and scales dataset
                for autoencoder.
        """

        X = self.get_df_weather_energy()[self.lstcolallweather]
        y = self.get_df_weather_energy()['denoised_avg_energy']

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.33, random_state=1)

        t = MinMaxScaler()
        t.fit(X_train)
        X_train = t.transform(X_train)
        X_test = t.transform(X_test)
        return X, X_train, X_test

    def define_encoder(self, X, X_train, X_test):
        """
            This method defines autoencoder-decoder model, 
                trains the model onto the dataset
                gets the autoencoder part.
        """

        # define encoder
        n_inputs = X.shape[1]
        visible = Input(shape=(n_inputs,))
        # encoder level 1
        e = Dense(n_inputs*2)(visible)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # encoder level 2
        e = Dense(n_inputs)(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # bottleneck
        n_bottleneck = self.get_encoded_dimension()
        bottleneck = Dense(n_bottleneck)(e)
        # define decoder, level 1
        d = Dense(n_inputs)(bottleneck)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        # decoder level 2
        d = Dense(n_inputs*2)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        # output layer
        output = Dense(n_inputs, activation='linear')(d)
        # define autoencoder model
        model = Model(inputs=visible, outputs=output)
        # compile autoencoder model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        # plot the autoencoder
        # plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
        # fit the autoencoder model to reconstruct input
        history = model.fit(X_train, X_train, epochs=1000, batch_size=32, verbose=2, validation_data=(X_test,X_test))
        # plot loss
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        # define an encoder model (without the decoder)
        encoder = Model(inputs=visible, outputs=bottleneck)
        plot_model(encoder, 'encoder_no_compress.png', show_shapes=True)
        # save the encoder to file
        encoder.save('encoder.h5')
        return encoder

    def concatenate_encoded(self, encoder, X_train, X_test):
        """
            This method predicts dataset using autoencoder model,
                concatenates the prediction 
                    onto original dataset as encoded feature.
        """

        # encode the train data
        X_train_encode = encoder.predict(X_train)
        # encode the test data
        X_test_encode = encoder.predict(X_test)

        weather_encoded = \
            np.concatenate((X_train_encode, X_test_encode), axis=0)
        weather_encoded = \
            pd.DataFrame(weather_encoded, \
                columns=self.create_encoded_columns())
        self.set_df_weather_energy( \
            pd.concat([self.get_df_weather_energy(), weather_encoded], \
                axis=1, join='inner'))

    def create_encoded_columns(self):
        """
            This method creates columns corresponding to
                encoded features.
        """
        
        encoded_dimension = self.get_encoded_dimension()
        lstcols = []

        for count in range(encoded_dimension):
            str_col = 'weather_encoded' + str(encoded_dimension) + '_' + str(count)
            lstcols.append(str_col)

        return lstcols

    def auto_encode(self, encoded_dimension):
        """
            This method preprocess data for autoencoder,
                defines autoencoder model, 
                trains and predicts on the dataset
                adds encoded features onto original dataset.

            This method acts the entry point 
                for adding encoded features.
        """

        X, X_train, X_test = self.preprocess_encoder()
        self.set_encoded_dimension(encoded_dimension)

        print("\n\n---------------------------------------------------\n")
        print("Encoded dimension is set to " + str(self.get_encoded_dimension()))
        print("\n---------------------------------------------------\n\n")
        
        encoder = self.define_encoder(X, X_train, X_test)
        self.concatenate_encoded(encoder, X_train, X_test)


if __name__ == '__main__':

    myenergy = EnergyFeature()
    for dimension in range(3, 6):
        myenergy.auto_encode(dimension)
    myenergy.get_df_weather_energy().to_csv( \
            os.path.join("input", "energycombined.csv"), index=False)
            
    print(myenergy.get_df_weather_energy())
