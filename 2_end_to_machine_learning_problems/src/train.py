from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.tree import DecisionTreeRegressor


if __name__ == '__main__':

    df = pd.read_csv('../input/train_clean.csv')
    X = df.loc[:,df.columns[:-1]].values
    y = df.loc[:,df.columns[-1]].values 

    """
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X,y)
    pred = lr.predict(X)
    lin_mse = mean_squared_error(y,pred)
    lin_rmse = np.sqrt(lin_mse)
    print(f'The RMSE value : {lin_rmse}')
    # The RMSE value : 68368.60339648873
    # The RMSE value is actually quite underwhelming, therefore, we can see that the model in
    # underfitting, let's try some more complex values
    """

    # Decision Trees
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X,y)
    pred = tree_reg.predict(X)
    tree_mse = mean_squared_error(y,pred)
    tree_rmse = np.sqrt(tree_mse)
    print(f'The RMSE value : {tree_rmse}')
    # The RMSE value : 0.0 