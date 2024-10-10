import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches
from mokepy import MOKEAnalyzer  # Assuming MOKEAnalyzer is in a separate file

from importlib import reload


class BatchAnalyzer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.analyzers = {}
        self.summary_data = []

    def process_files(self):
        """Process all .csv files in the specified folder."""
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.csv') and 'coercive_fields_summary' not in filename:
                file_path = os.path.join(self.folder_path, filename)
                analyzer = MOKEAnalyzer(file_path)
                analyzer.read_data()
                analyzer.subtract_background()
                analyzer.normalize_data()
                analyzer.find_coercive_field()
                analyzer.fit_switching_field_distribution()
                self.analyzers[filename] = analyzer
                
                # Collect summary data
                self.summary_data.append({
                    'Filename': filename,
                    'Hc1': analyzer.Hc_summary['Hc1'],
                    'Hc2': analyzer.Hc_summary['Hc2'],
                    'Hc_mean': analyzer.Hc_summary['Hc_mean'],
                    'Area': self.extract_parameter(filename, 'Area'),  # Extract Area from filename or data
                    'Thickness': self.extract_parameter(filename, 'Thickness')  # Extract Thickness from filename or data
                })

    def extract_parameter(self, filename, param_name):
        """Extracts a specified parameter from the filename or data."""
        # This is a placeholder implementation. You may need to adjust it based on your actual data structure.
        # For demonstration purposes, let's assume we can extract parameters from the filename.
        if param_name == 'Area':
            return float(filename.split('-')[3])  # Example extraction logic (adjust as needed)
        elif param_name == 'Thickness':
            return float(filename.split('-')[4])  # Example extraction logic (adjust as needed)
        else:
            return None

    def plot_combined_hysteresis_loops(self):
        """Plot hysteresis loops from all files on a single graph."""
        plt.figure(figsize=(12, 8))
        for filename, analyzer in self.analyzers.items():
            plt.plot(analyzer.data['H_ad'], analyzer.data['M_ad_norm'], label=filename)
        plt.xlabel('Field (Oe)')
        plt.ylabel('Normalized Magnetization')
        plt.title('Combined Hysteresis Loops')
        plt.legend()
        plt.savefig(os.path.join(self.folder_path, 'combined_hysteresis_loops.png'))
        plt.close()

    def plot_combined_switching_field_distributions(self):
        """Plot switching field distributions from all files on a single graph."""
        plt.figure(figsize=(12, 8))
        for filename, analyzer in self.analyzers.items():
            field, magnetization = analyzer.data['H_ad'], analyzer.data['M_ad_norm']
            mag_diff = np.diff(magnetization)
            hist, bin_edges = np.histogram(field[:-1][mag_diff > 0], bins=30, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.plot(bin_centers, hist, label=filename)
        plt.xlabel('Field (Oe)')
        plt.ylabel('Normalized Frequency')
        plt.title('Combined Switching Field Distributions')
        plt.legend()
        plt.savefig(os.path.join(self.folder_path, 'combined_switching_field_distributions.png'))
        plt.close()

    def save_coercive_fields(self):
        """Save coercive fields to a CSV file."""
        csv_path = os.path.join(self.folder_path, 'coercive_fields_summary.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Filename', 'Hc1', 'Hc2', 'Hc_mean', 'Area', 'Thickness']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in self.summary_data:
                writer.writerow(data)

    def create_summary_presentation(self):
        """Create a PowerPoint presentation summarizing the analysis."""
        prs = Presentation()
        
        # Title slide
        title_slide_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(title_slide_layout)
        title = slide.shapes.title
        subtitle = slide.placeholders[1]
        title.text = "MOKE Analysis Summary"
        subtitle.text = f"Folder: {self.folder_path}"

        # Combined Hysteresis Loops slide
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = slide.shapes.title
        title.text = "Combined Hysteresis Loops"
        img_path = os.path.join(self.folder_path, 'combined_hysteresis_loops.png')
        slide.shapes.add_picture(img_path, Inches(1), Inches(1.5), height=Inches(5))

        # Combined Switching Field Distributions slide
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = slide.shapes.title
        title.text = "Combined Switching Field Distributions"
        img_path = os.path.join(self.folder_path, 'combined_switching_field_distributions.png')
        slide.shapes.add_picture(img_path, Inches(1), Inches(1.5), height=Inches(5))

        # Coercive Fields Summary slide
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = slide.shapes.title
        title.text = "Coercive Fields Summary"
        
        # Add a table with coercive field data
        rows = len(self.summary_data) + 1
        cols = 6  # Updated to include Area and Thickness
        left = Inches(1)
        top = Inches(1.5)
        width = Inches(8)
        height = Inches(0.8 * rows)
        
        table = slide.shapes.add_table(rows, cols, left, top, width, height).table
        
        # Set column headers
        headers = ['Filename', 'Hc1 (Oe)', 'Hc2 (Oe)', 'Hc_mean (Oe)', 'Area (cm²)', 'Thickness (cm)']
        
        for i, header in enumerate(headers):
            table.cell(0, i).text = header
        
            # Fill in the data
            for i, data in enumerate(self.summary_data, start=1):
                table.cell(i, 0).text = data['Filename']
                table.cell(i, 1).text = f"{data['Hc1']:.2f}"
                table.cell(i, 2).text = f"{data['Hc2']:.2f}"
                table.cell(i, 3).text = f"{data['Hc_mean']:.2f}"
                table.cell(i, 4).text = f"{data['Area']:.4f}"
                table.cell(i, 5).text = f"{data['Thickness']:.4e}"  # Scientific notation for thickness

        # Save the presentation
        prs.save(os.path.join(self.folder_path, 'MOKE_Analysis_Summary.pptx'))

    def plot_coercive_fields_vs_parameter(self, parameter=[]):
        """Plots coercive fields against a specified parameter."""
        valid_parameters = ['Area', 'Thickness']
        Hc1_values = [data['Hc1'] for data in self.summary_data]
        Hc2_values = [data['Hc2'] for data in self.summary_data]
        if len(parameter) != Hc1_values:
            raise ValueError(f"Invalid parameter '{parameter}'. Choose from {valid_parameters}.")

        x_values = [data[parameter] for data in self.summary_data]
        

        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, Hc1_values, label='Hc1', color='blue', marker='o')
        plt.scatter(x_values, Hc2_values, label='Hc2', color='red', marker='x')

        plt.xlabel(parameter + " (cm² or cm)")
        plt.ylabel("Coercive Field (Oe)")
        plt.title(f"Coercive Fields vs {parameter}")
        plt.legend()
        plt.grid(True)

        # Save plot to file
        plt.savefig(os.path.join(self.folder_path, f'coercive_fields_vs_{parameter}.png'))
        plt.close()

    def run_batch_analysis(self):
       """Run the complete batch analysis process."""
       self.process_files()
       self.plot_combined_hysteresis_loops()
       self.plot_combined_switching_field_distributions()
       self.save_coercive_fields()
       self.create_summary_presentation()

# Usage example:
# batch_analyzer = BatchAnalyzer('/path/to/your/data/folder')
# batch_analyzer.run_batch_analysis()
# batch_analyzer.plot_coercive_fields_vs_parameter(parameter='Area')  # Plot coercive fields vs Area or Thickness