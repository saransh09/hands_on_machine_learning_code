import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor


if __name__ == '__main__':

    df = pd.read_csv('../input/train_clean.csv')
    X = df.loc[:,df.columns[:-1]].values
    y = df.loc[:,df.columns[-1]].values 

    
    # Linear Regression
    """
    lr = LinearRegression()
    lr.fit(X,y)
    pred = lr.predict(X)
    lin_mse = mean_squared_error(y,pred)
    lin_rmse = np.sqrt(lin_mse)
    """
    # print(f'The RMSE value : {lin_rmse}')
    # The RMSE value : 68368.60339648873
    # The RMSE value is actually quite underwhelming, therefore, we can see that the model in
    # underfitting, let's try some more complex values
    

    
    # Decision Trees
    """
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X,y)
    pred = tree_reg.predict(X)
    tree_mse = mean_squared_error(y,pred)
    tree_rmse = np.sqrt(tree_mse)
    """
    # print(f'The RMSE value : {tree_rmse}')
    # The RMSE value : 0.0 
    # This is classic overfitting, however this is not the way to go and we should probably go for
    # cross validation scheme

    # Random Forest Regressor
    """    
    forest_reg = RandomForestRegressor()
    forest_reg.fit(X,y)
    """
    
    """
    tree_scores = cross_val_score(tree_reg, X, y, scoring='neg_mean_squared_error', cv=10)
    tree_rmse_score_cross_val = np.sqrt(-tree_scores)
    print(f'tree RMSE cross-val score : {tree_rmse_score_cross_val}')
    print(f'Mean RMSE : {tree_rmse_score_cross_val.mean()}')
    print(f'STD RMSE : {tree_rmse_score_cross_val.std()}')
    # Mean RMSE : 72003.49690654696
    # STD RMSE : 2557.4937280284803
    """

    """
    lin_scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=10)
    lin_rmse_scores_cross_val = np.sqrt(-lin_scores)
    print(f'lr RMSE cross-val score : {lin_rmse_scores_cross_val}')
    print(f'Mean RMSE : {lin_rmse_scores_cross_val.mean()}')
    print(f'STD RMSE : {lin_rmse_scores_cross_val.std()}')
    # Mean RMSE : 68898.49528960025
    # STD RMSE : 2197.174486205656
    """

    # It is an evidence of overfitting, let's try and apply a random forest classifier

    """
    forest_scores = cross_val_score(forest_reg, X, y, scoring='neg_mean_squared_error', cv=10)
    forest_rmse_scores_cross_val = np.sqrt(-forest_scores)
    print(f'forest RMSE cross-val score : {forest_rmse_scores_cross_val}')
    print(f'Mean RMSE : {forest_rmse_scores_cross_val.mean()}')
    print(f'STD RMSE : {forest_rmse_scores_cross_val.std()}')
    Mean RMSE : 50359.00690383506
    STD RMSE : 1641.6507543863681    
    """

    # We can try more models too, however, let's try a GridSearchCV to find the best hyperparameters
    # for our model that we have with us.
    param_grid = [
        {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
        {'bootstrap': [False], 'n_estimators':[3,10], 'max_features':[2,3,4]},
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X,y)
    joblib.dump(grid_search, '../save_files/grid_search.pkl')
    print(f'Best hyper-parameters : {grid_search.best_params_}')
    # To fetch the best estimator
    best_estimator = grid_search.best_estimator_

    """When you have no idea what value a hyperparameter should have, a simple approach is
    to try out consecutive powers of 10 (or a smaller number if you want a more fine-grained
    search, as shown in this example with the n_estimators hyperparameter). """

    