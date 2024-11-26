import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean
from natsort import natsorted
import multiprocessing


columns_to_change = {
    "Voltage (V)":"Voltage(V)",
    "Current (A)": "Current(A)",
    "Charge Capacity (Ah)": "Charge_Capacity(Ah)",
    "Discharge Capacity (Ah)": "Discharge_Capacity(Ah)",
    "Charge Energy (Wh)": "Charge_Energy(Wh)",
    "Discharge Energy (Wh)": "Discharge_Energy(Wh)",
    "dV/dt (V/s)": "dV/dt(V/s)",
    "Step Time (s)": "Step_Time(s)",
    "Step Index": "Step_Index"
    }
columns_to_drop = [
    "ACR (Ohm)",
    "ACR(Ohm)",
    "Internal Resistance (Ohm)",
    "Internal Resistance(Ohm)",
    "Aux_Temperature(¡æ)_1",
    "Test Time (s)",
    "Test_Time(s)",
    "Cycle Index",
    "Cycle_Index",
    "Current",
    "Voltage",
    "Aux_Temperature(¡æ)_2",
    "Aux_Temperature(¡æ)_3"
]

class RANSAC_Outlier_Detector:

    def __init__(self, clean_cell_path):
        self.parquet_folder = clean_cell_path

    def main(self):
        # Arranging matplotlib settings
        fig, axes = plt.subplots(4, 3, figsize=(16, 12))
        axes = axes.flatten()
        ransac_label_data = pd.DataFrame()
        all_data = pd.DataFrame()

        for i, parquet_name in enumerate(os.listdir(self.parquet_folder)[:]):
            print(parquet_name)
            cell_name = parquet_name.split(".")[0]
            parquet_path = os.path.join(self.parquet_folder, parquet_name)
            df_cell = self.parquet_to_dataframe(parquet_path)
            df_label = self.create_label_df(df_cell)
            X, y, X_clean, y_clean, df_cell, df_label_cleaned = self.clean_outliers(df_cell, df_label, cell_name)

            ransac_label_data  = pd.concat([ransac_label_data, df_label_cleaned], ignore_index=True)
            all_data = pd.concat([all_data, df_cell], ignore_index=True)

            # axes[i].plot(X, y, label = cell_name)
            # axes[i].plot(X, np.full_like(X, 4), color = 'white')
            axes[i].scatter(X, y, color = "red", label = "Actual Max Disch")
            axes[i].scatter(X, np.full_like(X, 4), color = "white", label = "Zero")
            axes[i].scatter(X_clean, y_clean, color = "green", label = "RANSAC cleaned Max Disch")

            axes[i].set_title(f'Aging graph for {cell_name}')
            axes[i].set_xlabel('Cycle number')
            axes[i].set_ylabel('Max Discharge Capacity (Ah)')
            # axes[i].legend(loc = "lower right")

        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

        return ransac_label_data, all_data

    def parquet_to_dataframe(self, parquet_path):
        # read uncleaned parquet file and drop unnecessary columns
        df_cell = pd.read_parquet(parquet_path)
        columns_tobe_dropped = list(set(columns_to_drop).intersection(df_cell.columns))
        df_cell = df_cell.drop(columns=columns_tobe_dropped)
        for k, v in columns_to_change.items():
            if k in df_cell.columns:
                df_cell[columns_to_change[k]] = df_cell[columns_to_change[k]].fillna(df_cell[k])
                df_cell = df_cell.drop(columns=[k])
                df_cell[columns_to_change[k]] = df_cell[columns_to_change[k]].apply(lambda x: float(x) if x != 'NaN' else pd.NA)
        df_cell = df_cell.dropna() # drop null columns
        df_cell['Cycle'] = df_cell['Cycle'].str.split('_').str[1].astype(int)
        return df_cell
        
    def create_label_df(self, df_cell):
        label_data = []
        for cycle in range(1, int(df_cell["Cycle"].max())+1):
            cycle_data = df_cell[df_cell["Cycle"]==cycle]
            min_disch_cap = cycle_data["Discharge_Capacity(Ah)"].min()
            max_disch_cap = cycle_data["Discharge_Capacity(Ah)"].max()
            CDR = max_disch_cap - min_disch_cap
            time_index = cycle_data.index[0]
            label_data.append({"Time_Index": time_index, "Cycle": cycle, "CDR": CDR})
        df_label = pd.DataFrame(label_data).set_index("Time_Index")
        return df_label

    def clean_outliers(self, df_cell, df_label, cell_name):
        X, y = df_label["Cycle"], df_label["CDR"]
        df_label = df_label[df_label["CDR"]>=3] # low pass filter

        # RANSAC prediction
        X_reshaped, y_reshaped = df_label["Cycle"].to_numpy().reshape(-1, 1), df_label["CDR"].to_numpy().reshape(-1,1)
        reg = RANSACRegressor(random_state=0).fit(X_reshaped, y_reshaped)
        y_predicted = reg.predict(X_reshaped)

        # confidence interval (manual)

        # mean, std_dev = np.mean(y_predicted), np.std(y_predicted)
        # err = 1.15035 * (std_dev / np.sqrt(len(y_predicted)))
        err = 0.3
        lower_bound = y_predicted - err
        upper_bound = y_predicted + err

        # chose CDR rates only in confidence interval
        df_label['lower bound'], df_label['upper bound'] = lower_bound, upper_bound
        df_label_cleaned = df_label[(df_label['CDR'] >= df_label['lower bound']) & (df_label['CDR'] <=  df_label['upper bound'])]
        X_clean, y_clean = df_label_cleaned['Cycle'].to_numpy().reshape(-1, 1), df_label_cleaned['CDR'].to_numpy().reshape(-1, 1)
        df_label_cleaned.loc[:, ["Cell"]] = cell_name
        df_cell.loc[:, ["Cell"]] = cell_name

        return X, y, X_clean, y_clean, df_cell, df_label_cleaned
            

if __name__ == "__main__":
    clean_cell_path = r'C:\Users\PC_4766\Desktop\Sanan\Codes\Bitirme\cycle_parquets'
    ransac_label_data, all_data = RANSAC_Outlier_Detector(clean_cell_path).main()
    print(ransac_label_data)
    print(all_data)