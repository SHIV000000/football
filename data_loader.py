# data_loader.py

import pandas as pd
import os

def load_data(data_dir):
    all_data = []
    print(f"Attempting to load data from: {data_dir}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist")
    
    for year_folder in sorted(os.listdir(data_dir)):
        year_path = os.path.join(data_dir, year_folder)
        if os.path.isdir(year_path):
            for file in os.listdir(year_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(year_path, file)
                    print(f"Loading file: {file_path}")
                    try:
                        df = pd.read_csv(file_path, parse_dates=['Date'])
                        df['season'] = year_folder
                        df['league'] = file.split('.')[0]  # Extract league from filename
                        
                        # Ensure required columns are present
                        required_columns = ['Date', 'Team 1', 'Team 2', 'FT']
                        if all(col in df.columns for col in required_columns):
                            all_data.append(df)
                        else:
                            print(f"Skipping {file_path} due to missing required columns")
                    except Exception as e:
                        print(f"Error loading {file_path}: {str(e)}")
    
    if not all_data:
        raise ValueError("No valid data files found in the specified directory")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Columns in combined DataFrame: {combined_df.columns.tolist()}")
    
    return combined_df

def load_additional_data(additional_data_dir):
    print(f"Attempting to load additional data from: {additional_data_dir}")
    if not os.path.exists(additional_data_dir):
        raise FileNotFoundError(f"The directory {additional_data_dir} does not exist")
    
    additional_data = []
    if not os.listdir(additional_data_dir):  # Check if the directory is empty
        print("The additional data directory is empty")
        return None  # Return None if the directory is empty

    for file in os.listdir(additional_data_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(additional_data_dir, file)
            print(f"Loading additional file: {file_path}")
            try:
                df = pd.read_csv(file_path)
                additional_data.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
    
    if not additional_data:
        print("No valid additional data files found in the specified directory")
        return None
    
    combined_additional_df = pd.concat(additional_data, ignore_index=True)
    print(f"Combined additional DataFrame shape: {combined_additional_df.shape}")
    print(f"Columns in combined additional DataFrame: {combined_additional_df.columns.tolist()}")
    
    return combined_additional_df


