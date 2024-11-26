# Import Libraries
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

class EUC_Cleaner:
    '''
    Cleans dataset and extracts cycles using euclidean distance formula.

    Input: 
        - cell_parquet_path -> ../path/to/cell.parquet
        - parquet_name -> cell name (e.g., G1)
    
    Output:
        - parquet folders with cleaned and cycle extracted version

    Note: parquet is an alternative format for .csv files. Previously, csv or xlsx files are read,
        resampled, and corresponding parquet files are created. Parquets can be found in
        "main parquets" folder. The logic behind using parquet folders is file reading and writing procedure.
        It is like zipping the files to low memories which ease processing the dataset as a pandas dataframe.
    '''

    def __init__(self, cell_parquet_path, parquet_name):
        # self.parquet_file = r"D:\Proje_dosyalari\Arda\Batarya_Datalar\parquet_folder\main_parquets\G1.parquet"
        self.parquet_file = cell_parquet_path
        self.parquet_name = parquet_name

        self.df_cell = self.parquet_to_dataframe()
    

    def parquet_to_dataframe(self):
        '''
        Read parquet file for a specified cell name
        Perform pre-cleaning and return pre-cleaned dataframe
        '''
        df_cell = pd.read_parquet(self.parquet_file)
        columns_tobe_dropped = list(set(columns_to_drop).intersection(df_cell.columns))
        df_cell = df_cell.drop(columns=columns_tobe_dropped)
        for k, v in columns_to_change.items():
            if k in df_cell.columns:
                df_cell[columns_to_change[k]] = df_cell[columns_to_change[k]].fillna(df_cell[k])
                df_cell = df_cell.drop(columns=[k])
                df_cell[columns_to_change[k]] = df_cell[columns_to_change[k]].apply(lambda x: float(x) if x != 'NaN' else pd.NA)
        df_cell = df_cell.dropna()
        return df_cell
    
    def define_first_cycle(self, df_cell):
        """
        Define first cycle for each cell manually
        """
        if self.parquet_name == 'G1':
            first_cycle = df_cell[300:7900]
            initial = 300
            cycle_length = 7600
        if self.parquet_name  == "V4":
            first_cycle = df_cell[5180:16300]
            initial = 5180
            cycle_length = 11120
        if self.parquet_name == "W10":
            first_cycle = df_cell[550:8250]
            initial = 550
            cycle_length = 7700
        if self.parquet_name == "W5":
            first_cycle = df_cell[550:8350]
            initial = 550
            cycle_length = 7800
        if self.parquet_name == "W8":
            first_cycle = df_cell[520:8450]
            initial = 520
            cycle_length = 7930
        if self.parquet_name == "W9":
            first_cycle = df_cell[550:8280]
            initial = 550
            cycle_length = 7730
        if self.parquet_name == "V5":
            first_cycle = df_cell[16050:30900]
            initial = 16050
            cycle_length = 14850
        if self.parquet_name == "W3":
            first_cycle = df_cell[50:7860]
            initial = 50
            cycle_length = 7810
        if self.parquet_name == "W4":
            first_cycle = df_cell[620:9200]
            initial = 620
            cycle_length = 8580
        if self.parquet_name == "W7":
            first_cycle = df_cell[480:8800]
            initial = 480
            cycle_length = 8320

        return first_cycle, initial, cycle_length


    def create_sliding_windows(self, slide):
        """
        Takes pre-cleaned cell dataframe 
        returns generated sliding windows with defined slide value 
        """
        df_cell = self.parquet_to_dataframe()
        first_cycle, initial, cycle_length = self.define_first_cycle(df_cell)
        last = initial+cycle_length

        first_cycle = first_cycle.reset_index(drop=False)
        disch_data = df_cell[df_cell["Cycle"]>=2][:].reset_index(drop=False)
        sliding_windows = pd.DataFrame()
        window_num = 1

        euclidean_distances = {}

        while last <= len(disch_data):
            if window_num % 100 == 0:
                selected_windows = self.filter_euc(euclidean_distances)
                clean_window_names = selected_windows.keys()
                sliding_windows = sliding_windows[sliding_windows["Window"].isin(clean_window_names)]
                euclidean_distances = {window: euclidean_distances[window] for window in euclidean_distances if window in clean_window_names}

            subset = disch_data.loc[initial:last-1]
            window_name = "Window_{}".format(window_num)
            subset.loc[:, ["Window"]] = window_name
            if len(subset) == len(first_cycle):
                euc = np.linalg.norm(subset["Discharge_Capacity(Ah)"].to_numpy()-first_cycle["Discharge_Capacity(Ah)"].to_numpy())
                subset.loc[:, ["EUC"]] = euc
                euclidean_distances[window_name] = euc
            sliding_windows = pd.concat([sliding_windows, subset], ignore_index=True)
            initial += slide
            last += slide
            window_num+=1
        del subset, disch_data, first_cycle
        return sliding_windows, euclidean_distances

    def filter_euc(self, euclidean_distances):
        selected_windows = {}
        last_window = "Window_0"
        for window, distance in natsorted(euclidean_distances.items()):
            if distance<=50 and distance > 0:
                selected_windows[window] = distance
                print("{} value is written in dictionary".format(window))
                if int(window.split("_")[-1]) - int(last_window.split("_")[-1]) == 1:
                    print("{} is sequential".format(window))
                    if distance < last_distance:
                        print("{} distance is less than {} distance".format(window, last_window))
                        del selected_windows[last_window]
                    else:
                        del selected_windows[window]
                        print("{} distance is more and deleted".format(window))   
                last_window = window
                last_distance = euclidean_distances[last_window]
        return selected_windows
    
    
    def choose_cycles(self, parquet_path):
        sliding_windows, euclidean_distances = self.create_sliding_windows(slide = 50)
        selected_windows = self.filter_euc(euclidean_distances)

        clean_window_names = selected_windows.keys()
        df_cell_cleaned = sliding_windows[sliding_windows["Window"].isin(clean_window_names)].set_index("Date_Time", inplace = False)
        # df_cleaned_cycle = df_cleaned_cycle.dropna()
        cycles = {}
        for i, window_name in enumerate(natsorted(clean_window_names)):
            cycle_name = "Cycle_{}".format(i+1)
            cycles[window_name] = cycle_name

        def cycle_name_generator(window_name):
            return cycles[window_name]
        
        df_cell_cleaned.loc[:, ["Cycle"]] = df_cell_cleaned["Window"].apply(cycle_name_generator)
        df_cell_cleaned["Cycle"] = df_cell_cleaned["Cycle"].astype(str)

        df_cell_cleaned.to_parquet(parquet_path)

        return df_cell_cleaned

    
if __name__ == "__main__":
    main_parquet_path = r"C:\Users\PC_4766\Desktop\Sanan\Codes\Bitirme\main_parquets"
    write_parquet_path = r"C:\Users\PC_4766\Desktop\Sanan\Codes\Bitirme\cycle_parquets"

    g1_path = r'C:\Users\PC_4766\Desktop\Sanan\Codes\Bitirme\cycle_parquets\G1.parquet'
    v4_path = r'C:\Users\PC_4766\Desktop\Sanan\Codes\Bitirme\cycle_parquets\V4.parquet'
    w5_path = r'C:\Users\PC_4766\Desktop\Sanan\Codes\Bitirme\cycle_parquets\W5.parquet'
    w8_path = r'C:\Users\PC_4766\Desktop\Sanan\Codes\Bitirme\cycle_parquets\W8.parquet'
    w9_path = r'C:\Users\PC_4766\Desktop\Sanan\Codes\Bitirme\cycle_parquets\W9.parquet'
    w10_path = r'C:\Users\PC_4766\Desktop\Sanan\Codes\Bitirme\cycle_parquets\W10.parquet'

    v5_path = r'C:\Users\PC_4766\Desktop\Sanan\Codes\Bitirme\cycle_parquets\V5.parquet'
    w3_path = r'C:\Users\PC_4766\Desktop\Sanan\Codes\Bitirme\cycle_parquets\W3.parquet'
    w4_path = r'C:\Users\PC_4766\Desktop\Sanan\Codes\Bitirme\cycle_parquets\W4.parquet'
    w7_path = r'C:\Users\PC_4766\Desktop\Sanan\Codes\Bitirme\cycle_parquets\W7.parquet'


    cell_names = ['G1', 'V4', 'W5', 'W8', 'W9', 'W10']
    new_cell_names = ['V5', 'W3', 'W4', 'W7']
    output_paths = []
    targets = {}
    # sliding_windows, euclidean_distances = EUC_Cleaner(parquet_path, parquet_name).create_sliding_windows(slide = 100)
    # df_cell_cleaned, sliding_windows, euclidean_distances = EUC_Cleaner(G1_parquet_path, parquet_name).choose_cycles()

    for i, parquet in enumerate(natsorted((os.listdir(main_parquet_path)))):
        parquet_name = parquet.split(".")[0]
        print(parquet_name)
        if parquet_name in new_cell_names:
            parquet_path = os.path.join(main_parquet_path, parquet)
            print(parquet_path)
            output_parquet_path = os.path.join(write_parquet_path, parquet)
            output_paths.append(output_parquet_path)
            print(write_parquet_path)

            my_target = EUC_Cleaner(parquet_path, parquet_name)
            targets[parquet_name] = my_target

    # p1 = multiprocessing.Process(target=targets["G1"].choose_cycles, args=(g1_path,))
    # p2 = multiprocessing.Process(target=targets["V4"].choose_cycles, args=(v4_path,))
    # p3 = multiprocessing.Process(target=targets["W5"].choose_cycles, args=(w5_path,))
    # p4 = multiprocessing.Process(target=targets["W8"].choose_cycles, args=(w8_path,))
    # p5 = multiprocessing.Process(target=targets["W9"].choose_cycles, args=(w9_path,))
    # p6 = multiprocessing.Process(target=targets["W10"].choose_cycles, args=(w10_path,))
            
    p1 = multiprocessing.Process(target=targets["V5"].choose_cycles, args=(v5_path,))
    p2 = multiprocessing.Process(target=targets["W3"].choose_cycles, args=(w3_path,))
    p3 = multiprocessing.Process(target=targets["W4"].choose_cycles, args=(w4_path,))
    p4 = multiprocessing.Process(target=targets["W7"].choose_cycles, args=(w7_path,))


    p1.start()
    p2.start()
    p3.start()
    p4.start()