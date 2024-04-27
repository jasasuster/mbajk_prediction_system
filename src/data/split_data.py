import pandas as pd

def main():
    # Read data from reference_data.csv into a dataframe
    df = pd.read_csv('data/processed/reference_data.csv')

    # Split the dataframe into train and test dataframes
    test_size = int(0.1 * len(df))
    test_df = df.head(test_size)
    train_df = df.iloc[test_size:]

    # Save train_df and test_df to train.csv and test.csv respectively
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
