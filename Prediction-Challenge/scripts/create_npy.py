import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import h5py  # For saving large arrays in memory-efficient HDF5 format
from tqdm import tqdm 
from npy_append_array import NpyAppendArray
from tqdm import tqdm


'''
This script creates the pre processed Data :
- named X_train_LSTM : with n_observations sequences of 100*n_features for the LSTM 
- named X_train_additionnal : with n_observations of n_features for the independant analysis '''


### PARAMETERS
date = '05-12'
x_train_path = "..\Data\X_train.csv"
y_train_path = "..\Data\y_train.csv"





### Functions 
def encode_df(df_to_encode):
    categorical_columns = ['venue','action','side','trade']
    df_pandas_encoded = pd.get_dummies(df_to_encode,columns=categorical_columns,drop_first=True,dtype=int)

    return df_pandas_encoded

def correct_df(
    df, 
    column_names=[
        'price', 'bid', 'ask', 'bid_size', 'ask_size', 'flux', 
        'venue_1', 'venue_2', 'venue_3', 'venue_4', 'venue_5', 'action_D', 'action_U', 'side_B',
        'trade_True'
    ]
):
    """
    Ensures the DataFrame has columns in a specified order, adding missing columns with zeros.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to correct.
        column_names (list): List of column names in the desired order (default provided).
    
    Returns:
        pd.DataFrame: The corrected DataFrame.
    """
    # Add missing columns with zeros
    for column in column_names:
        if column not in df.columns:
            df[column] = 0
    
    # Reorder columns to match the specified order
    df = df[column_names]
    
    return df

def transform_df(df):
    #we want to drop obs_id, order_id 
    df = df.drop(['order_id','obs_id'],axis=1) #dropping obs id because they did so in the benchmark
    return df


def create_lstm_data(data, k):
    '''
    input:
        data - the pandas object of (n_observations x 100 , p) shape, where n is the number of rows,
               p is the number of predictors
        k    - the length of the sequences, namely, the number of previous rows 
               (including current) we want to use to predict the target.
    output:
        X_data - the predictors numpy matrix of (n-k, k, p) shape
    '''


    # initialize zero matrix of (n-k, k, p) shape to store the n-k number
    # of sequences of k-length and zero array of (n-k, 1) to store targets
    X_data = np.zeros((data.shape[0]//k, k, data.shape[1]))
    
    # run loop to slice k-number of previous rows as 1 sequence to predict
    # 1 target and save them to X_data matrix and y_data list
    for i in range(data.shape[0]//k):
        cur_sequence = data.iloc[k*i: k*(i+1), :]
                
        X_data[i,:,:] = cur_sequence
    
    return X_data

def create_additional_data(df,seq_len):
    #input df the df is made of sequences of seq_len rows, for each sequence, we return the max + min of column price as a proxy of the volatility 

    # initialize zero matrix of ( n_rows, n_faetures=1) shape to store the features for each row
    # n_rows = df.shape[0]//seq_len
    X_data = np.zeros((df.shape[0]//seq_len,1))
    
    # run loop to slice k-number of previous rows as 1 sequence to predict
    for i in range(df.shape[0]//seq_len):
        cur_sequence = df.iloc[seq_len*i: seq_len*(i+1), :]

        max = cur_sequence['price'].max()
        min = cur_sequence['price'].min()


        X_data[i] = max+min
    
    return X_data


def process_data_chunked(x_train_path, y_train_path, output_prefix, chunk_size=10_000, seq_len=100):
    """
    Process data chunk by chunk and save the results incrementally.
    
    Args:
    - x_train_path: Path to the X_train CSV file.
    - y_train_path: Path to the y_train CSV file.
    - output_prefix: Prefix for the output files.
    - chunk_size: Number of rows to process in each chunk.
    - seq_len: Length of each sequence for LSTM.
    """
    # Read y_train (the target file) entirely as it's small and doesn't need chunking
    y_train_full = pd.read_csv(y_train_path).drop('obs_id', axis=1)

    # Use tqdm to show progress
    total_rows = sum(1 for _ in open(x_train_path)) - 1  # Get total rows excluding header
    num_chunks = (total_rows + chunk_size - 1) // chunk_size  # Calculate total chunks

    X_train_LSTM_npy_name = f"..\Data\{output_prefix}_X_train_LSTM.npy"
    X_train_add_npy_name = f"..\Data\{output_prefix}_X_train_add.npy"
    y_train_npy_name = f"..\Data\{output_prefix}_y_train.npy"


    # Process the X_train file in chunks
    with NpyAppendArray(X_train_LSTM_npy_name, delete_if_exists=True) as npaa, \
     NpyAppendArray(X_train_add_npy_name, delete_if_exists=True) as npaa_add:
        for i, chunk in enumerate(tqdm(pd.read_csv(x_train_path, chunksize=chunk_size), desc="Processing Chunks", total=num_chunks)):
            # Apply the transformation functions
            chunk_bis = correct_df(transform_df(encode_df(chunk)))

            # Create LSTM-compatible data for this chunk
            X_data = create_lstm_data(chunk_bis, seq_len)
            npaa.append(X_data)

            X_data_add = create_additional_data(chunk,seq_len)
            npaa_add.append(X_data_add)

            # Clear memory for the current chunk
            del X_data, X_data_add, chunk, chunk_bis


    np.save(y_train_npy_name,y_train_full)
    print(f"Processing completed. X_train and y_train saved as {output_prefix}_X_train.npy and {output_prefix}_y_train.npy.")


if __name__ == '__main__':
    process_data_chunked(x_train_path, y_train_path, date, chunk_size=10_000, seq_len=100)


