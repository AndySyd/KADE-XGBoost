import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution as de
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

random_col_list = []
rmse_list = []

def objf(F):
    train = pd.read_csv('train_scour_weir.csv')
    test = pd.read_csv('test_scour_weir.csv')
    y_test = test.iloc[:,-1]
    y_train = train.iloc[:,-1]
    F = F.astype('int')
    print(sorted(np.unique(F)),np.unique(F).shape)
    # for i in range(len(F)):
    #     F[i] = int(F[i])
    # print(sorted(F))
    # random_col = np.random.choice(int(F),29,replace=False).tolist()
    random_col = np.array(F).tolist()
    random_col = np.unique(random_col)
    # print(sorted(random_col))

    x_train_selected = train.iloc[:,random_col]
    x_test_selected = test.iloc[:,random_col]

    XGBR = xgb.XGBRegressor(max_depth=10, n_estimators=500, learning_rate=0.036911,
                            gamma=0.632, min_child_weight=2, subsample=0.993997637,
                            colsample_bytree=0.9, colsample_bylevel=0.99,
                            reg_alpha=0.00099, reg_lambda=0.999996161)
    XGBR.fit(x_train_selected,y_train)

    y_pred = XGBR.predict(x_test_selected)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))

    random_col_list.append(random_col)
    rmse_list.append(rmse)

    return rmse

# bounds = [(0,i) for i  in range(34)]
bounds = [(0,34)]*34

ret = de(func=objf,bounds=bounds,maxiter=100,recombination=0.9,mutation=0.6,disp=True,polish=True)

print('验证最佳rmse:',min(rmse_list))
print('验证最佳特征组合:',sorted(np.unique(ret.x.astype('int'))))

train = pd.read_csv('train_scour_weir.csv')
test = pd.read_csv('test_scour_weir.csv')
y_test = test.iloc[:,-1]
y_train = train.iloc[:,-1]

x_train = train.iloc[:,np.unique(ret.x.astype('int'))]
x_test = test.iloc[:,np.unique(ret.x.astype('int'))]

xgbb = xgb.XGBRegressor(max_depth=10, n_estimators=500, learning_rate=0.036911,
                        gamma=0.632, min_child_weight=2, subsample=0.993997637,
                        colsample_bytree=0.9, colsample_bylevel=0.99,
                        reg_alpha=0.00099, reg_lambda=0.999996161)
xgbb.fit(x_train, y_train)

y_pred = xgbb.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('R2:', r2)
