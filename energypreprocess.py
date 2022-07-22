import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import denoise_wavelet

import os

class EnergyPreprocess:
    """
        This class reads energy data from file energy.csv, 
            weather from file weather_daily_cleaned.csv,
            holiday from file uk_bank_holidays.csv,
            combines them into one dataframe,
            adds denoised average energy consumption feature to the dataframe,
            finally writes the dataframe into file energyweather.csv

        Note: weather_daily_cleaned.csv file has been manually copied and pasted 
            from weather_daily_darksky.csv, 
            11 rows had incorrect date, 
            I manually modified them.

    """

    def __init__(self, 
                    energyfile = os.path.join("input", "energy.csv"), 
                    weatherfile = os.path.join("input", "weather_daily_cleaned.csv")):

        self.df_energy = pd.read_csv(energyfile)
        self.df_weather = pd.read_csv(weatherfile)

        self.df_housecount = \
            self.get_df_energy().groupby('day')[['LCLid']].nunique()

        self.set_df_energy(self.get_df_energy().groupby('day')[['energy_sum']].sum())
        self.set_df_energy(self.get_df_energy().merge(self.df_housecount, on=['day']))
        self.set_df_energy(self.get_df_energy().reset_index())

        self.get_df_energy()['day'] = pd.to_datetime(self.get_df_energy()['day'], format = '%Y-%m-%d')
        
        # add addtional day related features
        self.get_df_energy()['dayofweek'] = self.get_df_energy()['day'].dt.dayofweek
        self.get_df_energy()['quarter'] = self.get_df_energy()['day'].dt.quarter
        self.get_df_energy()['month'] = self.get_df_energy()['day'].dt.month
        self.get_df_energy()['year'] = self.get_df_energy()['day'].dt.year
        self.get_df_energy()['dayofyear'] = self.get_df_energy()['day'].dt.dayofyear
        self.get_df_energy()['dayofmonth'] = self.get_df_energy()['day'].dt.day
        self.get_df_energy()['weekofyear'] = self.get_df_energy()['day'].dt.isocalendar().week
        self.get_df_energy()['sin_day'] = np.sin(self.get_df_energy()['dayofyear'])
        self.get_df_energy()['cos_day'] = np.cos(self.get_df_energy()['dayofyear'])

        self.get_df_energy()['day'] = self.get_df_energy()['day'].dt.date
        self.get_df_energy()['avg_energy'] = self.get_df_energy()['energy_sum'] / self.get_df_energy()['LCLid']

        self.get_df_weather()['day'] = pd.to_datetime(self.get_df_weather()['day']).dt.date

        lstweathercols = ['temperatureMax', 'windBearing', 'dewPoint', 'cloudCover', 'windSpeed',
                        'pressure', 'apparentTemperatureHigh', 'visibility', 'humidity',
                        'apparentTemperatureLow', 'apparentTemperatureMax', 'uvIndex',
                        'temperatureLow', 'temperatureMin', 'temperatureHigh',
                        'apparentTemperatureMin', 'moonPhase','day']
        self.set_df_weather(self.get_df_weather()[lstweathercols])        

        self.df_weather_energy = self.get_df_energy().merge(self.get_df_weather(), on='day')

        # remove outlier, the last row
        self.set_df_weather_energy( \
            self.get_df_weather_energy().drop( \
                self.get_df_weather_energy().avg_energy.idxmin()))

        self.get_df_weather_energy()['weekday'] = self.get_df_weather_energy()['day'].apply(lambda x: x.weekday())
        self.get_df_weather_energy()['weekday'] = self.get_df_weather_energy()['weekday'].apply(lambda x: 0 if x >= 5 else 1)

        holiday = pd.read_csv('uk_bank_holidays.csv')
        holiday['Bank holidays'] = pd.to_datetime(holiday['Bank holidays'],format='%Y-%m-%d').dt.date
        self.set_df_weather_energy(self.get_df_weather_energy().merge(holiday, \
                    left_on = 'day',right_on = 'Bank holidays',how = 'left'))

        self.get_df_weather_energy()['holiday_ind'] = \
            np.where(self.get_df_weather_energy()['Bank holidays'].isna(),0,1)

        self.denoise_avg_energy()
        
        if (not os.path.exists(os.path.join("input", "energyweather.csv"))):
            self.get_df_weather_energy().to_csv(os.path.join("input", "energyweather.csv"), index=False)

    def set_df_weather(self, df_weather):
        self.df_weather = df_weather

    def get_df_weather(self):
        return self.df_weather

    def set_df_energy(self, df_energy):
        self.df_energy = df_energy

    def get_df_energy(self):
        return self.df_energy

    def get_df_weather_energy(self):
        return self.df_weather_energy

    def set_df_weather_energy(self, df_weather_energy):
        self.df_weather_energy = df_weather_energy

    def denoise_avg_energy(self):
        """
            Add denoised average energy feature to the dataset
                using wavelet transform
        """

        self.get_df_weather_energy()['denoised_avg_energy'] = \
            denoise_wavelet(self.get_df_weather_energy()['avg_energy'],
                            method = 'BayesShrink', 
                            mode = 'soft', 
                            wavelet_levels=3,
                            rescale_sigma=True) 

if __name__ == '__main__':
    myenergy = EnergyPreprocess()

    print(myenergy.get_df_weather_energy())
      