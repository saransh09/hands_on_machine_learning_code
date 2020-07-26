import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

# Custom transformer to add the additional features that we want
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:,rooms_ix] / X[:,households_ix]
        population_per_household = X[:,population_ix] / X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix] / X[:,rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


if __name__ == '__main__':
    household = pd.read_csv('../input/train.csv')
    household_train = household.drop('median_house_value', axis=1)

    num_pipeline = Pipeline([
                        ('impute', SimpleImputer(strategy='median')),
                        ('attribs_adder', CombinedAttributesAdder()),
                        ('std_scaler', StandardScaler())
                    ])
    """
    # If one wantes to just apply these transforms
    household_num_tr = num_pipeline.fit_transform(housing_num)
    joblib.dump(num_pipeline, '../save_files/num_pipeline.bin')
    """

    # But we can go one step further and compose all the transformation into a single file
    num_attribs = [c for c in household_train.columns if c not in ['ocean_proximity']]
    cat_attribs = ['ocean_proximity']

    full_pipeline = ColumnTransformer([
                        ('num', num_pipeline, num_attribs),
                        ('cat', OneHotEncoder(), cat_attribs)
                    ])
    
    housing_prepared = full_pipeline.fit_transform(household_train)
    joblib.dump(full_pipeline, '../save_files/full_pipeline.bin')

    cat_names = list(full_pipeline.transformers_[1][1].categories_[0])
    cat_names = [c.replace(' ','_') for c in cat_names]
    additional_names = ['room_per_household', 'population_per_household', 'bedrooms_per_room']
    tran_columns = num_attribs + additional_names + cat_names
    df_transform = pd.DataFrame(housing_prepared, columns=tran_columns, index=household_train.index)
    df_transform.loc[:,'median_house_value'] = household.loc[:,'median_house_value']

    df_transform.to_csv('../input/train_clean.csv', index=False)