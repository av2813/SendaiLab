import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import scienceplots
from typing import List, Dict, Optional

plt.style.use(['science', 'ieee', 'no-latex'])

class MOKEAnalyzer:
    """
    A class for analyzing Magneto-Optic Kerr Effect (MOKE) data.

    Attributes:
        material (str): The material being analyzed.
        sample_name (str): The name of the sample.
        array_name (str): The name of the array.
        filename (str): The name of the data file.
        folder (str): The folder containing the data file.
        path (str): The full path to the data file.
        data (pd.DataFrame): The raw data.
        data_ref (pd.DataFrame): Reference data.
        processed_data (pd.DataFrame): Processed data.
        metadata (Dict[str, str]): Metadata extracted from the file.
        background_fit (Dict): Background fit parameters.
        coercive_field (float): Calculated coercive field.
        saturation_magnetization (float): Calculated saturation magnetization.
    """

    def __init__(self, material: str, sample_name: str, array_name: str, filename: str, folder: str):
        """
        Initialize the MOKEAnalyzer.

        Args:
            material (str): The material being analyzed.
            sample_name (str): The name of the sample.
            array_name (str): The name of the array.
            filename (str): The name of the data file.
            folder (str): The folder containing the data file.
        """
        self.material = material
        self.sample_name = sample_name
        self.array_name = array_name
        self.name = sample_name+array_name
        self.filename = filename
        self.folder = folder
        self.path = os.path.join(folder, filename)
        self.data = None
        self.data_ref = None
        self.processed_data = None
        self.metadata = {}
        self.background_fit = None
        self.coercive_field = None
        self.saturation_magnetization = None
        self.properties = {}

    def add_property(self, key, prop):
        self.properties[key] = prop

    def read_data(self, extract_metadata: bool = True) -> None:
        """
        Read data from the file and optionally extract metadata.

        Args:
            extract_metadata (bool): Whether to extract metadata from the file. Defaults to True.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            pd.errors.EmptyDataError: If the file is empty or contains no data.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"The file {self.path} does not exist.")

        try:
            with open(self.path, 'r', encoding='latin1') as file:
                lines = file.readlines()

            if extract_metadata:
                for line in lines[:31]:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        self.metadata[key.strip()] = value.strip()

            self.data = pd.read_csv(self.path, header=31, skipfooter=2002, encoding='latin1')
            self.data_ref = pd.read_csv(self.path, header=2033, encoding='latin1')
            self.data['H'] = self.data['H']*1000
            if self.data.empty or self.data_ref.empty:
                raise pd.errors.EmptyDataError("The file contains no data.")

        except pd.errors.EmptyDataError as e:
            raise pd.errors.EmptyDataError(f"Error reading data from {self.path}: {str(e)}")
        except Exception as e:
            raise Exception(f"An error occurred while reading the file {self.path}: {str(e)}")

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
        mask = self.data['H'] > field_threshold
        positive_mask = self.data['H'] > field_threshold
        negative_mask = self.data['H'] < -field_threshold
        # Perform the curve fit
        popt_pos, _ = curve_fit(linear, self.data.loc[positive_mask, 'H'], self.data.loc[positive_mask, 'Theta k'])
        popt_neg, _ = curve_fit(linear, self.data.loc[negative_mask, 'H'], self.data.loc[negative_mask, 'Theta k'])
        
        grad_ave = (popt_pos[0]+popt_neg[0])/2

        # Calculate background for all data points
        background = self.data['H']*grad_ave

        # Store the background-subtracted data
        #self.processed_data = self.data.copy()
        self.data['M'] = self.data['Theta k'] - background

        # Store the fit parameters and covariance
        self.background_fit = {
            'popt_neg': popt_neg,
            'popt_pos': popt_pos,
            'background': background
        }

        

    def smooth_data(self, window_length: int = 11, polyorder: int = 3) -> None:
        """
        Smooth the processed data using Savitzky-Golay filter.

        Args:
            window_length (int): The length of the filter window. Defaults to 11.
            polyorder (int): The order of the polynomial used to fit the samples. Defaults to 3.

        Raises:
            ValueError: If the data has not been processed or if the window length or polyorder are invalid.
        """
        if 'M' not in self.data.columns:
            raise ValueError("Data not processed. Call fit_background() first.")

        if window_length % 2 == 0 or window_length < 3:
            raise ValueError("Window length must be an odd number greater than or equal to 3.")

        if polyorder >= window_length:
            raise ValueError("Polyorder must be less than window length.")

        self.data['M Smooth'] = savgol_filter(self.data['M'], window_length, polyorder)

    def calculate_coercive_field(self):
        # Ensure we have background-subtracted data
        print(self.data.columns)
        if 'M' not in self.data.columns:
            raise ValueError("Background subtraction has not been performed. Call fit_background() first.")

        # Create interpolation functions for positive and negative branches
        positive_branch = self.data[self.data['H'] >= 0]
        negative_branch = self.data[self.data['H'] <= 0]

        interp_func_pos = interp1d(positive_branch['M'], positive_branch['H'])
        interp_func_neg = interp1d(negative_branch['M'], negative_branch['H'])

        # Find the zero crossings
        self.positive_coercive_field = float(interp_func_pos(0))
        self.negative_coercive_field = float(interp_func_neg(0))

        # Calculate the average coercive field (absolute value)
        self.coercive_field = (abs(self.positive_coercive_field) + abs(self.negative_coercive_field)) / 2

    def calculate_saturation_magnetization(self) -> None:
        """
        Calculate the saturation magnetization of the sample.

        Raises:
            ValueError: If the data has not been processed.
        """
        if 'M' not in self.data.columns:
            raise ValueError("Data not processed. Call fit_background() and smooth_data() first.")

        self.saturation_magnetization = max(abs(self.data['M']))

    def plot_data(self, ax, data_types=['raw'], color='k'):
        if 'raw' in data_types:
            ax.plot(self.data['H'], self.data['Theta k'], '-', label='Raw Data', color=color)
        if 'processed' in data_types and 'M' in self.data.columns:
            ax.plot(self.data['H'], self.data['M Smooth'], '-', 
                    label=self.name, color=color)
        
        ax.set_xlabel('Applied Field (Oe)')
        ax.set_ylabel('MOKE Signal (a.u.)')
        ax.legend()

    def save_processed_data(self, filename):
        if 'M' in self.data.columns:
            self.data.to_csv(os.path.join(self.folder, filename+'_pro.csv'), index=False)
        else:
            print("No processed data to save. Run analyze() first.")
    
    def analyze(self, field_threshold: float, window_length: int = 3, polyorder: int = 1) -> None:
        """
        Perform the full analysis pipeline.

        Args:
            field_threshold (float): The field threshold for background fitting.
            window_length (int): The length of the filter window for smoothing. Defaults to 11.
            polyorder (int): The order of the polynomial used for smoothing. Defaults to 3.

        Raises:
            Exception: If any step in the analysis pipeline fails.
        """
        try:
            self.read_data()
            self.fit_background(field_threshold)
            self.smooth_data(window_length, polyorder)
            self.calculate_coercive_field()
            self.calculate_saturation_magnetization()
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")

def compare_samples(samples, data_types=['processed'], colors=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(samples)))

    for sample, color in zip(samples, colors):
        sample.plot_data(ax, data_types, color)

    plt.title(f"Comparison of MOKE Data ({', '.join(data_types)})")
    return fig, ax

class BatchProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.samples = []
        self.summary_data = []

    def process_files(self, field_threshold):
        for filename in os.listdir(self.input_folder):
            if filename.endswith('.csv'):
                sample = MOKEAnalyzer(material="Unknown", sample_name=filename[:-4], 
                                      array_name="Unknown", filename=filename, 
                                      folder=self.input_folder)
                sample.analyze(field_threshold)
                self.samples.append(sample)
                self.summary_data.append({
                    'Sample': filename[:-4],
                    'Coercive Field': sample.coercive_field,
                    'Saturation Magnetization': sample.saturation_magnetization
                })

    def generate_summary_report(self):
        summary_df = pd.DataFrame(self.summary_data)
        summary_df.to_csv(os.path.join(self.output_folder, 'summary_report.csv'), index=False)

    def generate_comparison_plots(self):
        fig, ax = compare_samples(self.samples)
        plt.savefig(os.path.join(self.output_folder, 'comparison_plots.png'))
        plt.close()

    def run_batch_analysis(self, field_threshold):
        self.process_files(field_threshold)
        self.generate_summary_report()
        self.generate_comparison_plots()
        print("Batch analysis complete. Summary report and comparison plots generated.")

# Example usage
if __name__ == "__main__":
    input_folder = "moke_data"
    output_folder = "moke_results"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    batch_processor = BatchProcessor(input_folder, output_folder)
    batch_processor.run_batch_analysis(field_threshold=8)