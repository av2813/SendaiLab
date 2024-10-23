import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from lmfit import Model, Parameters
from natsort import natsorted
from scipy.integrate import odeint

class TRMOKEAnalyzer:
    def __init__(self):
        self.metadata_keys = [
            'Fieldstrength(mT)', 'Pumppower(mW)', 'Addnumber', 'Originalstageposition',
            'FieldID', 'Expectedfield(mT)', 'HWPangle(probe)', 'HWPangle(pump)',
            'QWPangle(pump)', 'reflectance'
        ]

    def read_data(self, filename: str) -> Tuple[Dict[str, float], List[str], np.ndarray]:
        metadata = {}
        column_headers = []

        with open(filename, 'r') as file:
            header = file.readline().strip().replace('"','').replace(' ','').replace(',,',',').replace('=','')
            header_parts = header.split(',')
            
            for i in range(0, len(header_parts), 2):
                key_part = header_parts[i]
                value_part = header_parts[i+1] if i+1 < len(header_parts) else ''
                metadata[key_part] = float(value_part)

            column_headers = [header.strip(' "') for header in file.readline().strip().split(',')]
            data = np.loadtxt(file, delimiter=',', usecols=(1, 2, 3, 4))
        print(metadata)
        return metadata, column_headers, data

    @staticmethod
    def moke_signal(t, A, tau_r, tau_d, y0):
        return A * (1 - np.exp(-t/tau_r)) * np.exp(-t/tau_d) + y0

    @staticmethod
    def moke_signal_double(t, A1, tau_r, tau_d1, A2, tau_d2, y0):
        return A1 * (1 - np.exp(-t/tau_r)) * np.exp(-t/tau_d1) + A2 * np.exp(-t/tau_d2) + y0
    
    @staticmethod
    def two_temperature_model(t, Te0, Tl0, G, Ce, Cl, P, tau):
        """
        Two-temperature model function.
        
        Parameters:
        t : array, time points
        Te0 : float, initial electron temperature
        Tl0 : float, initial lattice temperature
        G : float, electron-phonon coupling constant
        Ce : float, electron heat capacity
        Cl : float, lattice heat capacity
        P : float, laser power
        tau : float, laser pulse duration
        
        Returns:
        array: Electron temperature at each time point
        """
        def dTdt(T, t, G, Ce, Cl, P, tau):
            Te, Tl = T
            dTedt = -G/Ce * (Te - Tl) + P/(Ce*tau) * np.exp(-t/tau)
            dTldt = G/Cl * (Te - Tl)
            return [dTedt, dTldt]

        T0 = [Te0, Tl0]
        sol = odeint(dTdt, T0, t, args=(G, Ce, Cl, P, tau))
        return sol[:, 0]  # Return electron temperature

    def fit_two_temperature_model(self, t, signal):
        """
        Fit the two-temperature model to TR-MOKE data.
        
        Parameters:
        t : array, time points
        signal : array, TR-MOKE signal
        
        Returns:
        lmfit.ModelResult: Fit result
        """
        model = Model(self.two_temperature_model)
        params = Parameters()
        params.add('Te0', value=300, min=200, max=10000)
        params.add('Tl0', value=300, min=200, max=10000)
        params.add('G', value=1e17, min=1e16, max=1e18)
        params.add('Ce', value=1e3, min=1e2, max=1e4)
        params.add('Cl', value=1e6, min=1e5, max=1e7)
        params.add('P', value=1e-3, min=1e-4, max=1e-2)
        params.add('tau', value=100e-15*1e12, min=10e-15*1e12, max=1e-12*1e12)

        result = model.fit(signal, params, t=t)
        return result

    def fit_moke_signal(self, t, signal, double_exp=False):
        if double_exp:
            model = Model(self.moke_signal_double)
            params = Parameters()
            params.add('A1', value=np.ptp(signal))
            params.add('tau_r', value=0.1, min=0)
            params.add('tau_d1', value=1, min=0)
            params.add('A2', value=np.ptp(signal)/3)
            params.add('tau_d2', value=10, min=0)
            params.add('y0', value=0, vary=False)
        else:
            model = Model(self.moke_signal)
            params = Parameters()
            params.add('A', value=np.ptp(signal), min=0)
            params.add('tau_r', value=0.1, min=0)
            params.add('tau_d', value=1, min=0)
            params.add('y0', value=np.mean(signal))

        result = model.fit(signal, params, t=t)
        return result

    def process_file(self, filename: str, ax: Optional[plt.Axes] = None, color: str = 'b', double_exp: bool = False, use_2tm: bool = False) -> Dict:
        metadata, headers, data = self.read_data(filename)
        delay_time = data[:, 0]
        kerr_rotation = data[:, 1]

        fit_mask = delay_time > 0
        fit_delay_time = delay_time[fit_mask]
        fit_kerr_rotation = kerr_rotation[fit_mask]

        if use_2tm:
            result = self.fit_two_temperature_model(fit_delay_time, fit_kerr_rotation)
        else:
            result = self.fit_moke_signal(fit_delay_time, fit_kerr_rotation, double_exp)

        if ax:
            ax.plot(fit_delay_time, fit_kerr_rotation, 'o', color=color, markersize=3, alpha=0.5,
                    label=f"Data: {metadata['Pumppower(mW)']:.2f} mW, {metadata['Expectedfield(mT)']:.2f} mT")
            ax.plot(fit_delay_time, result.best_fit, '-', color=color, linewidth=1)

        fit_results = {
            'Pumppower(mW)': metadata['Pumppower(mW)'],
            'Expectedfield(mT)': metadata['Expectedfield(mT)'],
            **result.best_values
        }
        return fit_results

    def process_folder(self, folder_path: str, ax: Optional[plt.Axes] = None, color_map: str = 'viridis', double_exp: bool = False, use_2tm: bool = False) -> List[Dict]:
        data_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        data_files_sorted = natsorted(data_files)
        
        colors = plt.get_cmap(color_map)(np.linspace(0.1, 0.9, len(data_files_sorted)))
        
        fit_results = []
        for file, color in zip(data_files_sorted, colors):
            file_path = os.path.join(folder_path, file)
            result = self.process_file(file_path, ax, color, double_exp, use_2tm)
            fit_results.append(result)
        
        return fit_results

    def plot_fit_summary(self, fit_results: List[Dict], labels: List[str],colors: List[str], double_exp: bool = False):
        params = ['A', 'tau_r', 'tau_d', 'y0'] if not double_exp else ['A1', 'tau_r', 'tau_d1', 'A2', 'tau_d2', 'y0']
        fig, axs = plt.subplots(len(params)//2 + len(params)%2, 2, figsize=(15, 5*len(params)//2))
        axs = axs.flatten()

        for i, param in enumerate(params):
            ax = axs[i]
            for j, (results, label) in enumerate(zip(fit_results, labels)):
                pump_powers = [r['Pumppower(mW)'] for r in results if param in r]
                param_values = [r[param] for r in results if param in r]
                ax.scatter(pump_powers, param_values, label=label,color = colors[j])
            
            ax.set_xlabel('Pump Power (mW)')
            ax.set_ylabel(param)
            ax.set_title(f'{param} vs Pump Power')
            ax.legend()

        plt.tight_layout()
        plt.show()

    def analyze_folders(self, folder_paths: List[str], labels: List[str], color_maps: List[str], double_exp: bool = False, use_2tm: bool = False):
        fig, ax = plt.subplots(figsize=(12, 8))
        all_fit_results = []

        for folder_path, label, color_map in zip(folder_paths, labels, color_maps):
            fit_results = self.process_folder(folder_path, ax, double_exp=double_exp, use_2tm=use_2tm, color_map=color_map)
            all_fit_results.append(fit_results)

        ax.set_xlabel('Delay Time (ps)')
        ax.set_ylabel('Kerr Rotation Signal (μV)')
        ax.set_title('TR-MOKE Data Comparison')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        self.plot_fit_summary(all_fit_results, labels, double_exp=double_exp, use_2tm=use_2tm, colors=['r','purple','orange','blue'])
# Usage example:
#if __name__ == "__main__":
#    analyzer = TRMOKEAnalyzer()
#    
#    folder1 = r'path/to/folder1'
#    folder2 = r'path/to/folder2'
#    
#    analyzer.analyze_folders([folder1, folder2], ['500mT', '6mT'], double_exp=True)
