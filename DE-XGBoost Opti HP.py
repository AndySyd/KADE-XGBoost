import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score

train = pd.read_csv('train_scour_weir.csv')
test = pd.read_csv('test_scour_weir.csv')
y_test = test.iloc[:, -1]
y_train = train.iloc[:, -1]
x_train = train.drop(labels='class', axis=1)
x_test = test.drop(labels='class', axis=1)

def objf(HP):
    HP_dic = {
        'max_depth': int(HP[0])
        ,'n_estimators': int(HP[1])
        ,'learning_rate': float(HP[2])
        ,'gamma': float(HP[3])
        ,'min_child_weight': int(HP[4])
        ,'subsample': float(HP[5])
        ,'colsample_bytree': float(HP[6])
        ,'colsample_bylevel': float(HP[7])
        ,'reg_alpha': float(HP[8])
        ,'reg_lambda': float(HP[9])
    }

    XGBR = xgb.XGBRegressor(**HP_dic)
    XGBR.fit(x_train, y_train)

    y_pred = XGBR.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return rmse

r2_list = []

for i in range(1):
    print('这是第'+str(i+1)+'次独立运行')
    bounds = [(1, 15)     # max_depth
              ,(200, 600)  # n_estimators
              ,(0, 1.0)    # learning_rate
              ,(0, 1)      # gamma
              ,(1, 15)    # min_child_weight
              ,(0, 1.0)    # subsample
              ,(0, 1.0)    # colsample_bytree
              ,(0, 1.0)    # colsample_bylevel
              ,(0, 1.0)    # reg_alpha
              ,(0, 1.0)]    # reg_lambda

    ret = differential_evolution(func=objf, bounds=bounds, maxiter=100,recombination=0.9,mutation=0.6,polish=True)

    Best_HP = {
        'max_depth': int(ret.x[0])
        ,'n_estimators': int(ret.x[1])
        ,'learning_rate': float(ret.x[2])
        ,'gamma': float(ret.x[3])
        ,'min_child_weight': int(ret.x[4])
        ,'subsample': float(ret.x[5])
        ,'colsample_bytree': float(ret.x[6])
        ,'colsample_bylevel': float(ret.x[7])
        ,'reg_alpha': float(ret.x[8])
        ,'reg_lambda': float(ret.x[9])
    }

    print("Best HP = ", Best_HP)

    XGBR1 = xgb.XGBRegressor(**Best_HP)
    XGBR1.fit(x_train, y_train)

    y_pred = XGBR1.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    print("R2 :",r2)
    r2_list.append(r2)
print("结束所有运行")
print(r2_list)
print("R2最大值 =",np.max(r2_list))
print("R2平均值 =",np.mean(r2_list))
print("R2最小值 =",np.min(r2_list))
