# TR-MOKE Data Analysis Tool

This Python module provides a comprehensive tool for analyzing Time-Resolved Magneto-Optic Kerr Effect (TR-MOKE) data. It offers functionality for reading, processing, fitting, and visualizing TR-MOKE measurements.

## Features

- Data import from text files
- Single and double exponential fitting of TR-MOKE signals
- Two-temperature model (2TM) fitting
- Batch processing of multiple data files
- Comparative analysis of different datasets
- Customizable plotting and visualization

## Installation

To use this tool, ensure you have Python 3.x installed along with the following dependencies:

```bash
pip install numpy matplotlib scipy lmfit natsort
```

## Usage

1. Import the module:

```python
import trmoke
from importlib import reload
reload(trmoke)
```

2. Create an instance of the `TRMOKEAnalyzer` class:

```python
analyzer = trmoke.TRMOKEAnalyzer()
```

3. Define paths to your data folders:

```python
folder_al_5x_0mT = r'path/to/Al-0mT-data'
folder_al_5x_500mT = r'path/to/Al-500mT-data'
folder_si_5x_0mT = r'path/to/Si-0mT-data'
folder_si_5x_500mT = r'path/to/Si-500mT-data'
```

4. Set normalization factors for each dataset:

```python
normalisation_factors = [35/np.pi/2*1000, 35/np.pi/2*1000, 92.8/np.pi/2*1000, 92.8/np.pi/2*1000]
```

5. Analyze the folders:

```python
analyzer.analyze_folders(
    [folder_al_5x_500mT, folder_al_5x_0mT, folder_si_5x_500mT, folder_si_5x_0mT],
    ['Al-500mT', 'Al-0mT', 'Si-500mT', 'Si-0mT'],
    ['Reds', 'Greens', 'Greys', 'Blues'],
    double_exp=True,
    use_2tm=False,
    normalisation_factors=normalisation_factors
)
```

## Key Functions

- `read_data`: Reads TR-MOKE data from text files
- `moke_signal`: Single exponential model for TR-MOKE signal
- `moke_signal_double`: Double exponential model for TR-MOKE signal
- `two_temperature_model`: Implements the two-temperature model
- `fit_moke_signal`: Fits TR-MOKE data to single or double exponential models
- `fit_two_temperature_model`: Fits TR-MOKE data to the two-temperature model
- `process_file`: Processes a single data file
- `process_folder`: Processes all files in a folder
- `plot_fit_summary`: Generates summary plots of fitting parameters
- `analyze_folders`: Performs comparative analysis of multiple datasets

## Customization

- Adjust fitting parameters in `fit_moke_signal` and `fit_two_temperature_model` methods
- Modify plot aesthetics in `plot_fit_summary` and `analyze_folders` methods
- Extend the `TRMOKEAnalyzer` class to add new analysis techniques or visualization options

## Notes

- Ensure your data files are in the correct format (comma-separated values with appropriate headers)
- Adjust normalization factors based on your experimental setup
- Use `double_exp=True` for double exponential fitting and `use_2tm=True` for two-temperature model fitting

This tool provides a flexible framework for TR-MOKE data analysis, allowing researchers to efficiently process and visualize their experimental results[1].

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/181803/2a6e21a4-6a94-43ec-a3a4-08b679d8f532/paste.txt
