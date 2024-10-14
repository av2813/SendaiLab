import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

import scienceplots
from typing import List, Dict, Optional

plt.style.use(['science', 'ieee', 'no-latex'])


class MOKEAnalyzer:
    def __init__(self, path, name=''):
        self.path = path
        self.data = None
        self.name = name
        self.Hc_summary = {}
        self.background_params = None  # Store background fit parameters
        self.gaussian_fit_params = None  # Store Gaussian fit parameters

    def read_data(self):
        """Reads MOKE data from a CSV file."""
        try:
            self.data = pd.read_csv(self.path, names=['H', 'theta', 'H_ad', 'M_ad', 'M_dc'], header=22, encoding='shift_jis')
            
            self.data['H'] = self.data['H']*1000
            self.data['H_ad'] = self.data['H_ad']*1000
            #print(self.data)
        except Exception as e:
            raise ValueError(f"Error reading data from {self.path}: {e}")
        
    

    def fit_background(self, field_threshold: float) -> None:
        """
        Fit and subtract the background from the data.

        Args:
            field_threshold (float): The field threshold for background fitting.

        Raises:
            ValueError: If the data has not been read or if the field threshold is invalid.
        """
        if self.data is None:
            raise ValueError("Data has not been read. Call read_data() first.")

        if field_threshold <= 0:
            raise ValueError("Field threshold must be a positive number.")

        def linear(x, a, b):
            return a * x + b
        
        positive_mask = self.data['H'] > field_threshold
        negative_mask = self.data['H'] < -field_threshold
        # Perform the curve fit
        popt_pos, _ = curve_fit(linear, self.data.loc[positive_mask, 'H'], self.data.loc[positive_mask, 'theta'])
        popt_neg, _ = curve_fit(linear, self.data.loc[negative_mask, 'H'], self.data.loc[negative_mask, 'theta'])
        
        grad_ave = (popt_pos[0]+popt_neg[0])/2

        # Calculate background for all data points
        background = self.data['H']*grad_ave

        # Store the background-subtracted data
        self.data['M'] = self.data['theta'] - background

        # Store the fit parameters and covariance
        self.background_fit = {
            'popt_neg': popt_neg,
            'popt_pos': popt_pos,
            'background': background
        }

    def calc_Mr(self, field_threshold = 2000):
        

        field, magnetization = self.data['H'], self.data['M_norm']
        
        zero_crossings = np.where(np.diff(np.signbit(field)))[0]

        if len(zero_crossings) == 0:
            raise ValueError("No zero crossings found in magnetization data.")
        
        if len(zero_crossings) == 0:
            raise ValueError("No zero crossings found in magnetization data.")

        x1 = np.where(magnetization[zero_crossings] < 0)
        x2 = np.where(magnetization[zero_crossings] > 0)

        Mr1_values = magnetization[zero_crossings[x1]]
        Mr2_values = magnetization[zero_crossings[x2]]
        Mr1_mean =float(np.mean(Mr1_values))
        Mr2_mean =float(np.mean(Mr2_values))

        Mr1_std = np.std(magnetization[zero_crossings[x1]])
        Mr2_std = np.std(magnetization[zero_crossings[x2]])
        
        self.Mr_summary = {
            'Mr1': Mr1_mean,
            'Mr2': Mr2_mean,
            'Mr1_std': Mr1_std,
            'Mr2_std': Mr2_std,
            'Mr_mean': (abs(Mr1_mean) + abs(Mr2_mean)) / 2,
            'Mr_std': (Mr1_std + Mr2_std) / 2
        }
    
    def calc_squareness(self, field_threshold = 2000):
        positive_mask_H = self.data['H'] > field_threshold
        negative_mask_H = self.data['H'] < -field_threshold
        Mnorm_pos = self.data.loc[positive_mask_H, 'M_norm']
        Mnorm_neg = self.data.loc[negative_mask_H, 'M_norm']

        Ms_pos_mean = np.mean(Mnorm_pos)
        Ms_neg_mean = np.mean(Mnorm_neg)
        Ms_pos_std = np.std(Mnorm_pos)
        Ms_neg_std = np.std(Mnorm_neg)

        Ms_mean = (abs(Ms_pos_mean)+abs(Ms_neg_mean))/2

        squareness = self.Mr_summary['Mr_mean']/Ms_mean
        self.Mr_summary['MrMs'] = squareness
        self.Ms_summary = {
            'Ms1': Ms_pos_mean,
            'Ms2': Ms_neg_mean,
            'Ms1_std': Ms_pos_std,
            'Ms2_std': Ms_neg_std,
            'Ms_mean': (abs(Ms_pos_mean) + abs(Ms_neg_mean)) / 2,
            'Ms_std': (Ms_pos_std + Ms_neg_std) / 2
        }


    def normalize_data(self):
        """Normalizes the magnetization data."""
        mean_magnetization = np.mean(self.data['theta'])
        self.data['M_norm'] = self.data['theta'] - mean_magnetization

    def find_coercive_field(self):
        """Extracts coercive fields from the processed data."""
        field, magnetization = self.data['H'], self.data['M_norm']
        
        zero_crossings = np.where(np.diff(np.signbit(magnetization)))[0]
        if len(zero_crossings) == 0:
            raise ValueError("No zero crossings found in magnetization data.")
        
        if len(zero_crossings) == 0:
            raise ValueError("No zero crossings found in magnetization data.")

        x1 = np.where(field[zero_crossings] < 0)
        x2 = np.where(field[zero_crossings] > 0)
        
        Hc1 = np.mean(field[zero_crossings[x1]])
        Hc2 = np.mean(field[zero_crossings[x2]])
        
        Hc1_std = np.std(field[zero_crossings[x1]])
        Hc2_std = np.std(field[zero_crossings[x2]])
        
        # Store coercive field summary
        self.Hc_summary = {
            'Hc1': Hc1,
            'Hc2': Hc2,
            'Hc1_std': Hc1_std,
            'Hc2_std': Hc2_std,
            'Hc_mean': (abs(Hc1) + abs(Hc2)) / 2,
            'Hc_std': (Hc1_std + Hc2_std) / 2
        }

    def fit_switching_field_distribution(self):
        """Fits the switching field distribution with a Gaussian."""
        field, magnetization = self.data['H'], self.data['M_norm']
        
        # Calculate the derivative of magnetization to find switching fields
        mag_diff = np.diff(magnetization)
        field_diff = np.diff(field)

        mag_cdf = norm.cdf(mag_diff)
        sfd = np.divide(mag_diff, field_diff, out=np.zeros_like(mag_diff), where=field_diff!=0)
        sfd = np.abs(sfd)

        sfd_normalized = np.abs(sfd / np.trapz(sfd, field[:-1]))

        field_1, mag_cdf_1 = field[:500], sfd_normalized[:500]
        field_2, mag_cdf_2 = field[500:1000],sfd_normalized[500:1000]
        field_3, mag_cdf_3 = field[1000:1500],sfd_normalized[1000:1500]
        field_4, mag_cdf_4 = field[1500:-1],sfd_normalized[1500:]
        
        # Fit Gaussian to histogram data
        def gaussian(x, amp, mean, stddev, constant):
            return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))+ constant

        def lognormal(x, amp, mean, stddev, constant):
            return(amp* (1/(x*stddev*((2*np.pi)**0.5)))*np.exp(-(np.log(x)-mean)**2/(2*stddev**2))+constant)

        popt_neg, _ = curve_fit(gaussian, field_2, mag_cdf_2, p0=[max(mag_cdf_2), self.Hc_summary['Hc1'], 10, 0.])

        popt_pos, _ = curve_fit(gaussian, field_4, mag_cdf_4, p0=[max(mag_cdf_4), self.Hc_summary['Hc2'], 10, 0.])
        
        # Store Gaussian fit parameters
        self.sfd_fit_params = {'popt_neg':popt_neg, 'popt_pos':popt_pos}

    def gaussian(self,x, amp, mean, stddev, constant):
            return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))+constant

    def lognormal(self,x, amp, mean, stddev, constant):
        return(amp* (1/(x*stddev*((2*np.pi)**0.5)))*np.exp(-(np.log(x)-mean)**2/(2*stddev**2))+constant)

    def plot_hysteresis_loop(self, data_type = 'raw', color = 'k'):
        """Plots the hysteresis loop of raw and processed data."""
        fig, ax = plt.subplots(figsize=(10, 6))
        if data_type == 'raw':
            ax.plot(self.data['H'], self.data['theta'], label=self.name, color=color, alpha=0.5)
        elif data_type == 'processed':
            ax.plot(self.data['H'], self.data['M_norm'], label=self.name, color=color)
        elif data_type == 'normalised':
            ax.plot(self.data['H'], 2*self.data['M_norm']/(self.background_fit['popt_pos'][1]-self.background_fit['popt_neg'][1]), label=self.name, color=color)
        
        if 'M_ad_norm' in self.data.columns:
            ax.plot(self.data['H'], self.data['M_norm'], label='Normalized Data', color='green')
        
        if 'Hc_summary' in self.__dict__ and self.Hc_summary:
            ax.axvline(self.Hc_summary['Hc1'], color='red', linestyle='--', label='Hc1')
            ax.axvline(self.Hc_summary['Hc2'], color='purple', linestyle='--', label='Hc2')

        ax.set_xlabel('Field (Oe)')
        ax.set_ylabel('Magnetization (emu)')
        ax.legend()
        ax.set_title('Hysteresis Loop')
        
    def plot_switching_field_distribution(self):
        """Plots the switching field distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        field, magnetization = self.data['H'], self.data['M_norm']
        
        mag_diff = np.diff(magnetization)

        mag_cdf = norm.cdf(mag_diff)

        sfd = np.abs(mag_diff)

        sfd_normalized = np.abs(sfd / np.trapz(sfd, field[:-1]))

        ax.plot(field[:-1],sfd_normalized)

        ax.set_xlabel('Field (Oe)')
        ax.set_ylabel('Switching field distribution')
        
        # Fit Gaussian and plot it if parameters are available
        if self.gaussian_fit_params is not None:
            x_fit = np.linspace(min(field), max(field), 1000)
            y_fit_pos = self.gaussian(x_fit, self.sfd_fit_params['popt_pos'][0], self.sfd_fit_params['popt_pos'][1],self.sfd_fit_params['popt_pos'][2],self.sfd_fit_params['popt_pos'][3])
            y_fit_neg = self.gaussian(x_fit, self.sfd_fit_params['popt_neg'][0], self.sfd_fit_params['popt_neg'][1],self.sfd_fit_params['popt_neg'][2],self.sfd_fit_params['popt_neg'][3])
            ax.plot(x_fit, y_fit_pos, color='red', label='Pos Gaussian Fit')
            ax.plot(x_fit, y_fit_neg, color='blue', label='Neg Gaussian Fit')

# Usage Example:
# analyzer = MOKEAnalyzer('path_to_your_data.csv')
# analyzer.read_data()
# analyzer.subtract_background()
# analyzer.normalize_data()
# coercive_fields = analyzer.find_coercive_field()
# analyzer.plot_hysteresis_loop()
# analyzer.plot_switching_field_distribution()
# plt.show()