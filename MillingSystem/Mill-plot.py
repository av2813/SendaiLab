import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import logging
from ipywidgets import Button
from typing import Tuple, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return json.load(f)

# Constants (move these to config file for more flexibility)
CONFIG = load_config('config.json')

# Configuration
PLOT_CONFIG = {'FIG_SIZE':(12,6),
    'line_color': 'k',
    'line_style': '--',
    'text_rotation': 90,
    'text_va': 'top',
    'text_ha': 'right',
    'bbox_props': dict(facecolor='white', edgecolor='none', alpha=0.7)
}

def read_asc_file(file_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    metadata = {}
    try:
        with open(file_path, 'r') as file:
            for _ in range(CONFIG['METADATA_LINES']):
                line = file.readline().strip()
                if line:
                    key, value = line.split('\t', 1)
                    metadata[key] = value
        
        data = pd.read_csv(file_path, sep='\t', header=CONFIG['HEADER_ROWS'], skiprows=CONFIG['METADATA_LINES'])
        new_columns = []
        count = 0
        for col in data.columns:
            if 'Unnamed:' in col[0]:
                new_columns.append(f"{count}_{col[1]}")
            else:
                new_columns.append(f"{col[0]} {col[1]}")
                count += 1
        data.columns = new_columns
        
        return data, metadata
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None, None

def plot_ioncount(data: pd.DataFrame, ax: plt.Axes, x_l: str, y_l: str) -> None:
    ax.plot(data[x_l], np.log10(data[y_l]), label=y_l)

def setup_plot(data: pd.DataFrame, ax: plt.Axes) -> None:
    for ion in CONFIG['IONS_TO_PLOT']:
        plot_ioncount(data, ax, ion['x'], ion['y'])
    
    ax.legend()
    ax.set_xlabel('Time relative (s)')
    ax.set_ylabel('Ion count (cps) - log scale')
    ax.set_xlim(CONFIG['X_LIMIT'])

def calculate_statistics(data: pd.DataFrame) -> Dict[str, float]:
    stats = {}
    for ion in CONFIG['IONS_TO_PLOT']:
        stats[ion['y']] = {
            'mean': data[ion['y']].mean(),
            'std': data[ion['y']].std(),
            'max': data[ion['y']].max(),
            'min': data[ion['y']].min()
        }
    return stats

def export_data(data: pd.DataFrame, stats: Dict, output_path: str) -> None:
    data.to_csv(f"{output_path}_data.csv", index=False)
    with open(f"{output_path}_stats.json", 'w') as f:
        json.dump(stats, f, indent=4)

def add_vline(event, ax: plt.Axes, vlines: List) -> None:
    """Add a vertical line to the plot when clicked."""
    if event.inaxes == ax:
        x = event.xdata
        line = ax.axvline(x, color=PLOT_CONFIG['line_color'], linestyle=PLOT_CONFIG['line_style'])
        text = ax.text(x, ax.get_ylim()[1], f'Interface at {x:.2f}', 
                       rotation=PLOT_CONFIG['text_rotation'], va=PLOT_CONFIG['text_va'], ha=PLOT_CONFIG['text_ha'])
        vlines.append((x, line, text))
        vlines.sort(key=lambda v: v[0])
        
        if len(vlines) > 1:
            estimate_time_between_vlines(ax, vlines)
        
        plt.draw()

def estimate_time_between_vlines(ax: plt.Axes, vlines: List) -> None:
    """Estimate and display time differences between vertical lines."""
    for i in range(len(vlines) - 1):
        x1, _, _ = vlines[i]
        x2, _, _ = vlines[i + 1]
        time_diff = x2 - x1
        mid_point = (x1 + x2) / 2
        ax.text(mid_point, ax.get_ylim()[0], f'Δt = {time_diff:.2f} s', 
                ha='center', va='bottom', bbox=PLOT_CONFIG['bbox_props'])

def clear_vlines(b, ax: plt.Axes, vlines: List) -> None:
    """Clear all vertical lines and associated text from the plot."""
    for _, line, text in vlines:
        line.remove()
        text.remove()
    for text in ax.texts:
        text.remove()
    vlines.clear()
    plt.draw()


def main(args):
    data, metadata = read_asc_file(args.file_path)
    if data is None:
        return

    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(1, 1, figsize=PLOT_CONFIG['FIG_SIZE'])
    setup_plot(data, ax)

    vlines = []

    fig.canvas.mpl_connect('button_press_event', lambda event: add_vline(event, ax, vlines))

    button_clear = Button(description="Clear Lines")
    button_clear.on_click(lambda b: clear_vlines(b, ax, vlines))

    plt.show()
    
    #setup_plot(data, ax)

    stats = calculate_statistics(data)
    logging.info(f"Data statistics: {stats}")

    if args.export:
        export_data(data, stats, args.output)
        logging.info(f"Data exported to {args.output}")

    plt.show(block=True)  # Keep the window open until closed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and plot ASC file data.")
    parser.add_argument("file_path", help="Path to the ASC file")
    parser.add_argument("--export", action="store_true", help="Export data and statistics")
    parser.add_argument("--output", default="output", help="Output file prefix for exported data")
    args = parser.parse_args()

    main(args)
