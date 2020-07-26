from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np

# Learning : In case of numerical dataset, when splitting in training and test split and also when
# creating the folds, we should create some kind of categories and then do a stratified sampling 
# because randomly choosing the data can lead to bias / non representation of the whole dataset

# To achive this we use the pd.cut method --> check the usage below, and then we perform a
# StratifiedShuffleSplit() on the dataset.

if __name__ == '__main__':
    df = pd.read_csv('../input/housing.csv')

    # Creating a cut for the median household income
    df['income_cat'] = pd.cut(df['median_income'], bins=[0.,1.5,3.0,4.5,6.,np.inf], labels=[1,2,3,4,5])

    # Split the data into train and test --> as that has not been explicitly done for us right now
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for tr_idx, te_idx in strat_split.split(df, df['income_cat']):
        df_train = df.loc[tr_idx]
        df_test = df.loc[te_idx]

    df_train.drop('income_cat', axis=1, inplace=True)
    df_test.drop('income_cat', axis=1, inplace=True)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_train.to_csv("../input/train.csv",index=False)
    df_test.to_csv("../input/test.csv",index=False)