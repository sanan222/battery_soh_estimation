import os
import pandas as pd
import numpy as np
import re
from natsort import natsorted
import multiprocessing



class DatasetGenerator:
    '''
    Reads the whole dataset from the folder path, processes excel files, performs resampling and general cleaning
    and returns parquet dataset files for each cell.

    Input:
        - datase_path -> ../path/to/dataset_folder (e.g., folders with excel files downloaded directly from the Stanford dataset website)
    
    Output:
        - parquet_folders -> cleaned pandas dataframes saves with .parquet extension.

    Note: parquet file types are used to make reading and writing files more efficient and speedy.
    '''

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.cells_list = ['G1', 'V4', 'V5', 'W3', 'W4', 'W5', 'W7', 'W8', 'W9', 'W10']
    def process_dataset(self):
        # all_paths = [os.path.join(root, d) for root, dirs, _ in os.walk(self.dataset_path) for d in dirs]
        cached_cell_paths = []
        cell_paths_dict = self.organize_folders()
        for cell_name, cell_paths in cell_paths_dict.items():
            # if cell_name == given_cell_name:    
            cycle_number = 0
            parquets_df = pd.DataFrame()
            for cell_path in cell_paths:
                print(cell_path)
                    # read excels in a folder, resample and collect them in one parquet fle
                    # return nonprocessed folder paths
                    # cached_cell_paths.append(self.process_one_folder(cell_path))
         
                # collect parquet paths in each folder in a sorted order and combine them in one parquet
                for filename in os.listdir(cell_path):
                    filepath = os.path.join(cell_path, filename)
                    if filepath.endswith(".parquet"):
                        print(filepath)
                        parquet_df = pd.read_parquet(filepath)
                        parquet_df["Cycle"] = pd.to_numeric(parquet_df["Cycle"], 'coerce') + cycle_number
                        parquets_df = pd.concat([parquets_df, parquet_df])

                cycle_number = parquets_df["Cycle"].max()
                print(cycle_number)
            parquet_folder = r"D:\Proje_dosyalari\Arda\Batarya_Datalar\parquet_folder\main_parquets"
            parquet_path = os.path.join(parquet_folder, "{}.parquet".format(cell_name))
            # if "{}.parquet".format(cell_name) in os.listdir(parquet_folder):
            #     print("Parquet path is already created")
            # else:
            parquets_df.to_parquet(parquet_path)
            
    def organize_folders(self):
        cell_paths_dict = {cell_name: None for cell_name in self.cells_list}

        all_paths = [os.path.join(root, d) for root, dirs, _ in os.walk(self.dataset_path) for d in dirs]
        # extractin sorted version of cycling folders
        cycling_paths = [path for path in all_paths if path.split('\\')[-1].split('_')[0] == "Cycling"]
        cycling_paths = natsorted(cycling_paths)

        # extraction of cell files as a dict
        for cell_name in self.cells_list:
            one_cell_paths = [path for path in all_paths if cell_name in path.split('\\')]
            one_cell_paths = natsorted(one_cell_paths)
            for path in one_cell_paths:
                if path.split("\\")[-1] == "Part1" or path.split("\\")[-1] == "Channel_5":
                    repeated_path = os.path.dirname(path)
                    one_cell_paths.remove(repeated_path)
            cell_paths_dict[cell_name] = one_cell_paths
        return cell_paths_dict
    
    def process_one_folder(self, one_cell_path):
        # if one_cell_path.split('\\')[-1] in self.cells_list:
        parquet_path = os.path.join(one_cell_path, "Cell{}.parquet".format(one_cell_path.split('\\')[-1]))
        excel_paths = [os.path.join(one_cell_path, excel_file) for excel_file in os.listdir(one_cell_path) if os.path.join(one_cell_path, excel_file)]
        
        if parquet_path in excel_paths:
            # os.remove(parquet_path)
            print('{} parquet path has already been processed'.format(one_cell_path.split('\\')[-1]))
        else:
            try: 
                excel_paths = natsorted(excel_paths)
                df_cycles = pd.DataFrame()
                for excel_path in excel_paths[:]:
                    if re.search("Wb", excel_path):
                        # "INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_3C_N225_W4_Channel_2_Wb_2.xlsx"
                        channel_num = excel_path.split("\\")[-1].split("_")[-3] # 2
                        cycle_num = excel_path.split("\\")[-1].split("_")[-1].split(".")[0] # 2
                        try:
                            sheet_name = "Channel_{}_1".format(channel_num) # Channel_4_1
                            df_cycle = pd.read_excel(excel_path, sheet_name=sheet_name, index_col="Date_Time", parse_dates=True)
                        except:
                            sheet_name = "Channel{}_1".format(channel_num) # Channel4_1
                            df_cycle = pd.read_excel(excel_path, sheet_name=sheet_name, index_col="Date_Time", parse_dates=True)
                    else:
                        # "INR21700_M50T_T23_Aging_UDDS_SOC20_80_CC_0_5C_N225_W5_Channel_2.1.xlsx"
                        channel_num = excel_path.split("\\")[-1].split("_")[-1].split(".")[0] # 1
                        cycle_num = excel_path.split("\\")[-1].split("_")[-1].split(".")[1] # 2
                        try:
                            sheet_name = "Channel_{}_1".format(channel_num) # Channel_4_1
                            df_cycle = pd.read_excel(excel_path, sheet_name=sheet_name, index_col="Date_Time", parse_dates=True)
                        except:
                            sheet_name = "Channel{}_1".format(channel_num) # Channel4_1
                            df_cycle = pd.read_excel(excel_path, sheet_name=sheet_name, index_col="Date_Time", parse_dates=True)
                    
                    df_cycle.index = pd.to_datetime(df_cycle.index, format="ISO8601")
                    df_cycle = df_cycle.resample("5s").mean()
                    df_cycle["Cycle"] = cycle_num
                    df_cycles = pd.concat([df_cycles, df_cycle])
                print(parquet_path)
                df_cycles.to_parquet(parquet_path)
                print("{} path is processed".format(one_cell_path))
                        
            except:
                print("{} path could not be processed".format(one_cell_path))
            return one_cell_path
  

if __name__ == "__main__":
    dataset_path = r"D:\Proje_dosyalari\Arda\Batarya_Datalar\cycling_tests"
    one_cell_path = r"D:\Proje_dosyalari\Arda\Batarya_Datalar\cycling_tests\Cycling_2\G1\Part1"
    cells_list = ['G1', 'V4', 'V5', 'W3', 'W4', 'W5', 'W7', 'W8', 'W9', 'W10']
    dataset_generator = DatasetGenerator(dataset_path)
    dataset_generator.process_dataset()
    

    p1 = multiprocessing.Process(target=dataset_generator.process_dataset, args=("G1",))
    p2 = multiprocessing.Process(target=dataset_generator.process_dataset, args=("V4",))
    p3 = multiprocessing.Process(target=dataset_generator.process_dataset, args=("V5",))
    p4 = multiprocessing.Process(target=dataset_generator.process_dataset, args=("W3",))
    p5 = multiprocessing.Process(target=dataset_generator.process_dataset, args=("W4",))
    p6 = multiprocessing.Process(target=dataset_generator.process_dataset, args=("W5",))
    p7 = multiprocessing.Process(target=dataset_generator.process_dataset, args=("W7",))
    p8 = multiprocessing.Process(target=dataset_generator.process_dataset, args=("W8",))
    p9 = multiprocessing.Process(target=dataset_generator.process_dataset, args=("W9",))
    p10 = multiprocessing.Process(target=dataset_generator.process_dataset, args=("W10",))

    p1.start() 
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()
    p10.start()

