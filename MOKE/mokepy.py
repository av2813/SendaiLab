import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm


class MOKEAnalyzer:
    def __init__(self, path):
        self.path = path
        self.data = None
        self.Hc_summary = {}
        self.background_params = None  # Store background fit parameters
        self.gaussian_fit_params = None  # Store Gaussian fit parameters

    def read_data(self):
        """Reads MOKE data from a CSV file."""
        try:
            #self.data = pd.read_csv(self.path, encoding='shift_jis')
            #print(self.data)
            self.data = pd.read_csv(self.path, names=['H', 'theta', 'H_ad', 'M_ad', 'M_dc'], header=22, encoding='shift_jis')
            
            self.data['H'] = self.data['H']*1000
            self.data['H_ad'] = self.data['H_ad']*1000
            print(self.data)
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
        
        # Only fit to positive field data above the threshold
        #mask = self.data['H'] > field_threshold
        positive_mask = self.data['H'] > field_threshold
        negative_mask = self.data['H'] < -field_threshold
        # Perform the curve fit
        popt_pos, _ = curve_fit(linear, self.data.loc[positive_mask, 'H'], self.data.loc[positive_mask, 'theta'])
        popt_neg, _ = curve_fit(linear, self.data.loc[negative_mask, 'H'], self.data.loc[negative_mask, 'theta'])
        
        grad_ave = (popt_pos[0]+popt_neg[0])/2

        # Calculate background for all data points
        background = self.data['H']*grad_ave

        # Store the background-subtracted data
        #self.processed_data = self.data.copy()
        self.data['M'] = self.data['theta'] - background

        # Store the fit parameters and covariance
        self.background_fit = {
            'popt_neg': popt_neg,
            'popt_pos': popt_pos,
            'background': background
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
            'Hc_mean': (Hc1 + Hc2) / 2,
            'Hc_std': (Hc1_std + Hc2_std) / 2
        }

    def fit_switching_field_distribution(self):
        """Fits the switching field distribution with a Gaussian."""
        field, magnetization = self.data['H'], self.data['M_norm']
        
        # Calculate the derivative of magnetization to find switching fields
        mag_diff = np.diff(magnetization)

        mag_cdf = norm.cdf(mag_diff)

        sfd = np.abs(mag_diff)

        sfd_normalized = np.abs(sfd / np.trapz(sfd, field[:-1]))

        field_1, mag_cdf_1 = field[:500], sfd_normalized[:500]
        field_2, mag_cdf_2 = field[500:1000],sfd_normalized[500:1000]
        field_3, mag_cdf_3 = field[1000:1500],sfd_normalized[1000:1500]
        field_4, mag_cdf_4 = field[1500:-1],sfd_normalized[1500:]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(field_1,mag_cdf_1,'.',label='1')
        ax.plot(field_2,mag_cdf_2,'.',label='2')
        ax.plot(field_3,mag_cdf_3,'.',label='3')
        ax.plot(field_4,mag_cdf_4,'.',label='4')
        ax.axvline(self.Hc_summary['Hc1'])
        ax.axvline(self.Hc_summary['Hc2'], color = 'r')
        plt.legend()

        ax.set_xlabel('Field (Oe)')
        ax.set_ylabel('Switching field distribution')
        
        # Use histogram to estimate switching field distribution
        #hist, bin_edges = np.histogram(field[:-1][mag_diff > 0], bins=30)  # Only consider positive slopes
        
        #bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Fit Gaussian to histogram data
        def gaussian(x, amp, mean, stddev, constant):
            return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))+ constant

        def lognormal(x, amp, mean, stddev, constant):
            return(amp* (1/(x*stddev*((2*np.pi)**0.5)))*np.exp(-(np.log(x)-mean)**2/(2*stddev**2))+constant)

        popt_neg, _ = curve_fit(gaussian, field_2, mag_cdf_2, p0=[max(mag_cdf_2), self.Hc_summary['Hc1'], 10, 0.])

        popt_pos, _ = curve_fit(gaussian, field_4, mag_cdf_4, p0=[max(mag_cdf_4), self.Hc_summary['Hc2'], 10, 0.])
        
        # Store Gaussian fit parameters
        self.gaussian_fit_params = {'popt_neg':popt_neg, 'popt_pos':popt_pos}

    def gaussian(self,x, amp, mean, stddev, constant):
            return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))+constant

    def lognormal(self,x, amp, mean, stddev, constant):
        return(amp* (1/(x*stddev*((2*np.pi)**0.5)))*np.exp(-(np.log(x)-mean)**2/(2*stddev**2))+constant)

    def plot_hysteresis_loop(self):
        """Plots the hysteresis loop of raw and processed data."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.data['H'], self.data['theta'], label='Raw Data', color='blue', alpha=0.5)
        ax.plot(self.data['H'], self.data['M_norm'], label='Background Subtracted', color='orange')
        
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
        
        # Use histogram to estimate switching field distribution
        #hist, bin_edges = np.histogram(field[:-1][mag_diff > 0], bins=30)
        
        #bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        #ax.bar(bin_centers, hist, width=np.diff(bin_edges), align='center', alpha=0.5)
        
        # Fit Gaussian and plot it if parameters are available
        if self.gaussian_fit_params is not None:
            x_fit = np.linspace(min(field), max(field), 1000)
            y_fit_pos = self.gaussian(x_fit, self.gaussian_fit_params['popt_pos'][0], self.gaussian_fit_params['popt_pos'][1],self.gaussian_fit_params['popt_pos'][2],self.gaussian_fit_params['popt_pos'][3])
            y_fit_neg = self.gaussian(x_fit, self.gaussian_fit_params['popt_neg'][0], self.gaussian_fit_params['popt_neg'][1],self.gaussian_fit_params['popt_neg'][2],self.gaussian_fit_params['popt_neg'][3])
            #y_fit_pos = self.gaussian_fit_params['popt_pos'][0] * np.exp(-((x_fit - self.gaussian_fit_params['popt_pos'][1]) ** 2) / (2 * self.gaussian_fit_params['popt_pos'][2] ** 2))
            #y_fit_neg = self.gaussian_fit_params['popt_neg'][0] * np.exp(-((x_fit - self.gaussian_fit_params['popt_neg'][1]) ** 2) / (2 * self.gaussian_fit_params['popt_neg'][2] ** 2))
            ax.plot(x_fit, y_fit_pos, color='red', label='Pos Gaussian Fit')
            ax.plot(x_fit, y_fit_neg, color='blue', label='Neg Gaussian Fit')

        #    ax.set_title('Switching Field Distribution with Gaussian Fit')

# Usage Example:
# analyzer = MOKEAnalyzer('path_to_your_data.csv')
# analyzer.read_data()
# analyzer.subtract_background()
# analyzer.normalize_data()
# coercive_fields = analyzer.find_coercive_field()
# analyzer.plot_hysteresis_loop()
# analyzer.plot_switching_field_distribution()
# plt.show()