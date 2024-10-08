import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import scienceplots

plt.style.use(['science','ieee', 'no-latex'])

class VSMAnalyzer:
    def __init__(self, mass, volume, material, folder, filename, sample_name):
        self.mass = mass  # in grams
        self.volume = volume  # in cm^3
        self.material = material
        self.folder = folder
        self.filename = filename
        self.data = None
        self.background_subtracted_data = None
        self.scaled_data = None
        self.sample_name = sample_name
        self.metadata = {}

    def read_data(self, extract_metadata=False):
        file_path = os.path.join(self.folder, self.filename)
        
        # Read metadata and data
        with open(file_path, 'r', encoding='latin1') as file:
            lines = file.readlines()

        # Process metadata
        for line in lines[:40]:  # Assuming metadata is in the first 40 lines
            if '=' in line:
                key, value = line.strip().split('=', 1)
                self.metadata[key.strip()] = value.strip().strip(',')
            elif line.startswith('DATE'):
                self.metadata['header'] = line.strip()
        if extract_metadata:
            # Extract important metadata
            self.sample_name = self.metadata.get('sample name', '')
            self.sample_volume = float(self.metadata.get('sample volume', '1E-07').split(',')[0])
            self.sample_weight = float(self.metadata.get('sample weight', '1').split(',')[0])
            self.max_field = float(self.metadata.get('max magnetic field', '2000').split(',')[0])

        # Read actual data
        self.data = pd.read_csv(file_path, skiprows=41, names=['Date', 'Field', 'Moment', 'Angle'], 
                                encoding='latin1', parse_dates=['Date'])
        
        # Convert Field to float (removing any potential commas)
        #self.data['Field'] = self.data['Field'].str.replace(',', '').astype(float)
        self.data['Field'] = pd.to_numeric(self.data['Field'], errors='coerce')
        # Convert Moment to float (it's already in scientific notation)
        self.data['Moment'] = pd.to_numeric(self.data['Moment'], errors='coerce')
        #self.data['Moment'] = self.data['Moment'].astype(float)


    def fit_background(self, field_threshold):
        def linear(x, a, b):
            return a * x + b

        # Only fit to positive field data above the threshold
        mask = self.data['Field'] > field_threshold
        positive_mask = self.data['Field'] > field_threshold
        negative_mask = self.data['Field'] < -field_threshold
        # Perform the curve fit
        popt_pos, _ = curve_fit(linear, self.data.loc[positive_mask, 'Field'], self.data.loc[positive_mask, 'Moment'])
        popt_neg, _ = curve_fit(linear, self.data.loc[negative_mask, 'Field'], self.data.loc[negative_mask, 'Moment'])
        print(popt_neg, popt_pos)
        grad_ave = (popt_pos[0]+popt_neg[0])/2

        # Calculate background for all data points
        background = self.data['Field']*grad_ave

        # Store the background-subtracted data
        self.data['Moment_background_sub'] = self.data['Moment'] - background

        # Store the fit parameters and covariance
        self.background_fit = {
            'popt_neg': popt_neg,
            'popt_pos': popt_pos,
            'background': background
        }

    def calculate_coercive_field(self):
        interp_func = interp1d(self.data['Moment_background_sub'], self.data['Field'])
        self.coercive_field = abs(interp_func(0))

    def calculate_saturation_magnetization(self):
        self.saturation_magnetization_raw = (self.background_fit['popt_pos'][1]-self.background_fit['popt_neg'][1])/2#max(abs(self.data['Moment_background_sub']))
        self.saturation_magnetization_volume = self.saturation_magnetization_raw / self.volume
        self.saturation_magnetization_mass = self.saturation_magnetization_raw / self.mass

    def scale_data(self):
        """Scale the data by volume and mass, for both raw and background-subtracted data."""
        # Scale raw data
        self.data['Moment_Volume'] = self.data['Moment'] / self.volume
        self.data['Moment_Mass'] = self.data['Moment'] / self.mass
        
        # Scale background-subtracted data if it exists
        if 'Moment_background_sub' in self.data.columns:
            self.data['Moment_Volume_sub'] = self.data['Moment_background_sub'] / self.volume
            self.data['Moment_Mass_sub'] = self.data['Moment_background_sub'] / self.mass

    def get_moment_data(self, data_types):
        """
        Get moment data based on specified data types.
        
        :param data_types: List of data types (e.g., ['raw', 'background_subtracted', 'scaled_volume'])
        :return: Pandas Series of moment data
        """
        if 'background_subtracted' in data_types:
            base_moment = self.data['Moment_background_sub']
        else:
            base_moment = self.data['Moment']
        
        if 'scaled_volume' in data_types:
            return base_moment / self.volume
        elif 'scaled_mass' in data_types:
            return base_moment / self.mass
        else:
            return base_moment
        
    def plot_data(self, ax, label, data_types=['raw'], color = 'k'):
        moment = self.get_moment_data(data_types)

        ax.plot(self.data['Field'], moment,'-', label=self.sample_name, color = color)
        ax.set_xlabel('Field (Oe)')
        if 'scaled_volume' in data_types:
            ax.set_ylabel(r'Moment (emu/$cm^{3}$)')
        elif 'scaled_mass' in data_types:
            ax.set_ylabel('Moment (emu/g)')
        else:
            ax.set_ylabel('Moment (emu)')
        ax.legend()

    def save_plot(self, fig, filename):
        fig.savefig(os.path.join(self.folder, filename))

    def save_processed_data(self, filename):
        self.scaled_data.to_csv(os.path.join(self.folder, filename), index=False)

    def analyze(self, field_threshold):
        self.read_data()
        self.fit_background(field_threshold)
        self.scale_data()
        self.calculate_coercive_field()
        self.calculate_saturation_magnetization()
        

def compare_samples(samples,colors, data_type='raw'):
    fig, ax = plt.subplots(figsize=(5, 3))
    for sample,color in zip(samples,colors):
        sample.plot_data(ax, f"{sample.sample_name}", data_type, color)
    plt.title(f"Comparison of VSM Data ({data_type})")
    return fig, ax


'''
# Usage example
sample1 = VSMAnalyzer(mass=0.1, volume=0.05, material="Fe", folder="data", filename="sample1.txt")
sample2 = VSMAnalyzer(mass=0.15, volume=0.07, material="Co", folder="data", filename="sample2.txt")

samples = [sample1, sample2]

for sample in samples:
    sample.analyze(field_threshold=0.8)

# Plot and save raw data comparison
fig_raw, ax_raw = compare_samples(samples, 'raw')
fig_raw.savefig("raw_data_comparison.png")

# Plot and save background subtracted data comparison
fig_bg, ax_bg = compare_samples(samples, 'background_subtracted')
fig_bg.savefig("background_subtracted_comparison.png")

# Plot and save volume-scaled data comparison
fig_vol, ax_vol = compare_samples(samples, 'scaled_volume')
fig_vol.savefig("volume_scaled_comparison.png")

# Plot and save mass-scaled data comparison
fig_mass, ax_mass = compare_samples(samples, 'scaled_mass')
fig_mass.savefig("mass_scaled_comparison.png")

# Save processed data for each sample
for sample in samples:
    sample.save_processed_data(f"{sample.material}_processed_data.csv")

print("Analysis complete. Plots and processed data have been saved.")


'''

# ... (previous VSMAnalyzer class code remains the same)

class BatchProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.samples = []
        self.summary_data = []

    def process_files(self):
        for filename in os.listdir(self.input_folder):
            if filename.endswith('.txt'):  # Assuming VSM data files are .txt
                file_path = os.path.join(self.input_folder, filename)
                sample = VSMAnalyzer(mass=0.1, volume=0.05, material=filename[:-4], 
                                     folder=self.input_folder, filename=filename)
                sample.analyze(field_threshold=0.8)
                self.samples.append(sample)
                self.summary_data.append({
                    'Sample': filename[:-4],
                    'Coercive Field': sample.coercive_field,
                    'Saturation Magnetization (emu)': sample.saturation_magnetization_raw,
                    'Saturation Magnetization (emu/cm³)': sample.saturation_magnetization_volume,
                    'Saturation Magnetization (emu/g)': sample.saturation_magnetization_mass
                })

    def generate_summary_report(self):
        summary_df = pd.DataFrame(self.summary_data)
        summary_df.to_csv(os.path.join(self.output_folder, 'summary_report.csv'), index=False)
        
        # Generate PDF report
        pdf_path = os.path.join(self.output_folder, 'summary_report.pdf')
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 16)
        c.drawString(1*inch, height - 1*inch, "VSM Analysis Summary Report")

        c.setFont("Helvetica", 12)
        y = height - 2*inch
        for index, row in summary_df.iterrows():
            c.drawString(1*inch, y, f"Sample: {row['Sample']}")
            y -= 0.3*inch
            c.drawString(1.2*inch, y, f"Coercive Field: {row['Coercive Field']:.4f} T")
            y -= 0.3*inch
            c.drawString(1.2*inch, y, f"Saturation Magnetization: {row['Saturation Magnetization (emu)']:.4f} emu")
            y -= 0.3*inch
            c.drawString(1.2*inch, y, f"Volume Normalized: {row['Saturation Magnetization (emu/cm³)']:.4f} emu/cm³")
            y -= 0.3*inch
            c.drawString(1.2*inch, y, f"Mass Normalized: {row['Saturation Magnetization (emu/g)']:.4f} emu/g")
            y -= 0.5*inch

            if y < 2*inch:
                c.showPage()
                y = height - 1*inch

        c.save()

    def generate_comparison_plots(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        for sample in self.samples:
            sample.plot_data(ax1, f"{sample.material}", 'background_subtracted')
            sample.plot_data(ax2, f"{sample.material}", 'scaled_volume')
        
        ax1.set_title("Background Subtracted Data Comparison")
        ax2.set_title("Volume Normalized Data Comparison")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'comparison_plots.png'))
        plt.close()

    def run_batch_analysis(self):
        self.process_files()
        self.generate_summary_report()
        self.generate_comparison_plots()
        print("Batch analysis complete. Summary report and comparison plots generated.")


'''
# Usage example
input_folder = "vsm_data"
output_folder = "vsm_results"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

batch_processor = BatchProcessor(input_folder, output_folder)
batch_processor.run_batch_analysis()

'''