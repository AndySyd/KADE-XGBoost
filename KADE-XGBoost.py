"""
Author: Yangdi Shen
Email: shenyd98@163.com
Coding date: 10:04 on 24 April 2024
Related to paper: https://doi.org/10.1016/j.asoc.2025.112838
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Dataset loading
train = pd.read_csv('file path')
test = pd.read_csv('file path')

# Split into training and testing data
y_test = test.iloc['Rows and columns of y_test']
y_train = train.iloc['Rows and columns of y_train']
x_train = train.iloc['Rows and columns of x_train']
x_test = test.iloc['Rows and columns of x_test']

# Experiments with 30 Runs
for m in range(30):

    # Parameters initialization
    NP = 50
    HP_Dim = 10
    Ftrs_Dim = 26
    Dim = HP_Dim + Ftrs_Dim
    F = 0.5
    Cr = 0.5
    pop = []
    Total_Iterations = 100
    Zeta1 = 0.4
    Zeta2 = 0.6
    Delta1 = 0.6
    Delta2 = 0.8

    # Initialize population
    for i in range(NP):
        HP = [
            np.random.randint(1, 16)
            , np.random.randint(100, 200)
            , np.random.uniform(0, 1)
            , np.random.uniform(0, 1)
            , np.random.randint(1, 16)
            , np.random.uniform(0, 1)
            , np.random.uniform(0, 1)
            , np.random.uniform(0, 1)
            , np.random.uniform(0, 1)
            , np.random.uniform(0, 1)
              ]
        Ftrs = [np.round(np.random.rand(1), 1)[0] for i in range(Ftrs_Dim)]
        HP_F = HP + Ftrs
        pop.append(HP_F)

    # Boundaries of XGBoost hyperparameters
    HP_bounds = [
        (1, 15)
        , (100, 200)
        , (0, 1)
        , (0, 1)
        , (1, 15)
        , (0, 1)
        , (0, 1)
        , (0, 1)
        , (0, 1)
        , (0, 1)
                 ]
    Ftrs_bounds = [(0, 1)] * Ftrs_Dim
    bounds = HP_bounds + Ftrs_bounds

    # Object function
    def objf(HP_F):
        HP_dic = {
            'max_depth': int(HP_F[0])
            , 'n_estimators': int(HP_F[1])
            , 'learning_rate': float(HP_F[2])
            , 'gamma': float(HP_F[3])
            , 'min_child_weight': int(HP_F[4])
            , 'subsample': float(HP_F[5])
            , 'colsample_bytree': float(HP_F[6])
            , 'colsample_bylevel': float(HP_F[7])
            , 'reg_alpha': float(HP_F[8])
            , 'reg_lambda': float(HP_F[9])
                }

        # Feature value transformation
        for o in range(len(HP_F[len(HP):])):
            if HP_F[len(HP) + o] > 0.4:
                HP_F[len(HP) + o] = 1
            else:
                HP_F[len(HP) + o] = 0
        random_col = np.where(np.array([round(i) for i in HP_F[len(HP):]]) == 1)[0].tolist()
        if len(random_col) == 0:
            random_col = np.where(np.array([round(i) for i in HP_F[len(HP):]]) == 0)[0].tolist()
        x_train_selected = train.iloc[:, random_col]
        x_test_selected = test.iloc[:, random_col]

        # Calculate fitness (RMSE)
        XGBR = xgb.XGBRegressor(**HP_dic)
        XGBR.fit(x_train_selected, y_train)
        y_pred = XGBR.predict(x_test_selected)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse

    # KADE
    v = [[0] * Dim] * NP
    u = [[0] * Dim] * NP
    stable_featrues = []

    # Total_Iterations * Delta1
    for n in range(round(Total_Iterations * Delta1)):
        best = []
        pop_features_list = []
        for i in range(NP):

            # Mutation
            for j in range(Dim):
                r1, r2, r3, r4, r5 = np.random.choice(NP , 5, replace=False)
                v[i][j] = pop[r1][j] + (pop[r2][j] - pop[r3][j]) * F # DE/rand/1
                if v[i][j] < bounds[j][0]:
                    v[i][j] = bounds[j][0]
                elif v[i][j] > bounds[j][1]:
                    v[i][j] = bounds[j][1]

            # Crossover
            for j in range(Dim):
                if np.random.random() < Cr:
                    u[i][j] = v[i][j]
                else:
                    u[i][j] = pop[i][j]

            # Selection
            if objf(u[i]) < objf(pop[i]):
                for j in range(Dim):
                    pop[i][j] = u[i][j]

            best.append(objf(pop[i]))
            pop_features_list.extend(np.where(np.array([round(j) for j in pop[i][len(HP):]]) == 1)[0].tolist())
        print(np.min(best), n)

        # KAS (Extracting promising features at first stage)
        sorted_dic = {k: v for k, v in sorted(Counter(pop_features_list).items(), key=lambda item: item[1], reverse=True)}
        sorted_key = list(sorted_dic.keys())
        sorted_values = list(sorted_dic.values())
        index2 = round(Ftrs_Dim*Zeta1)
        stable_featrues = sorted_key[:index2]
        for i in range(len(stable_featrues)):
            stable_featrues[i] = stable_featrues[i] + len(HP)
    print('Promising featrues at first stage：', stable_featrues)


    # Total_Iterations * (Delta2-Delta1)
    for n in range(round(Total_Iterations * (Delta2-Delta1))):
        best = []
        pop_features_list = []
        for i in range(NP):

            # Mutation
            for j in range(Dim):
                r1, r2, r3, r4, r5 = np.random.choice(NP , 5, replace=False)
                v[i][j] = pop[r1][j] + (pop[r2][j] - pop[r3][j]) * F # DE/rand/1
                if v[i][j] < bounds[j][0]:
                    v[i][j] = bounds[j][0]
                elif v[i][j] > bounds[j][1]:
                    v[i][j] = bounds[j][1]

            # Crossover
            for j in range(Dim):
                if j not in stable_featrues:
                    if np.random.random() < Cr:
                        u[i][j] = v[i][j]
                    else:
                        u[i][j] = pop[i][j]
                else:
                    u[i][j] = 1 # Notably, the promising features are retained here.

            # Selection
            if objf(u[i]) < objf(pop[i]):
                for j in range(Dim):
                    pop[i][j] = u[i][j]

            best.append(objf(pop[i]))
            pop_features_list.extend(np.where(np.array([round(j) for j in pop[i][len(HP):]]) == 1)[0].tolist())
        print(np.min(best), n)

        # KAS (Extracting promising features at second stage)
        sorted_dic = {k: v for k, v in sorted(Counter(pop_features_list).items(), key=lambda item: item[1], reverse=True)}
        sorted_key = list(sorted_dic.keys())
        sorted_values = list(sorted_dic.values())
        index = None
        for i, value in enumerate(sorted_values):
            if value <= np.mean(np.array(sorted_values)):
                index = i
                break
        index = round(Ftrs_Dim*Zeta2)
        stable_featrues = sorted_key[:index]
        for i in range(len(stable_featrues)):
            stable_featrues[i] = stable_featrues[i] + len(HP)
    print('Promising featrues at second stage：', stable_featrues)

    # Total_Iterations * (1-Delta2)
    for n in range(round(Total_Iterations * (1-Delta2))):
        best = []
        pop_features_list = []
        for i in range(NP):

            # Mutation
            for j in range(Dim):
                r1, r2, r3, r4, r5 = np.random.choice(NP , 5, replace=False)
                v[i][j] = pop[r1][j] + (pop[r2][j] - pop[r3][j]) * F + (pop[r4][j] - pop[r5][j]) * F # DE/rand/2
                if v[i][j] < bounds[j][0]:
                    v[i][j] = bounds[j][0]
                elif v[i][j] > bounds[j][1]:
                    v[i][j] = bounds[j][1]

            # Crossover
            for j in range(Dim):
                if j not in stable_featrues:
                    if np.random.random() < Cr:
                        u[i][j] = v[i][j]
                    else:
                        u[i][j] = pop[i][j]
                else:
                    u[i][j] = 1 # Notably, the promising features are retained here.

            # Selection
            if objf(u[i]) < objf(pop[i]):
                for j in range(Dim):
                    pop[i][j] = u[i][j]

            best.append(objf(pop[i]))
            pop_features_list.extend(np.where(np.array([round(j) for j in pop[i][len(HP):]]) == 1)[0].tolist())
        print(np.min(best), n)

    print('Best solution', pop[np.argmin(best)])

# Performance metrics calculation
    HP_F = pop[np.argmin(best)]
    HP_dic = {
        'max_depth': int(HP_F[0])
        , 'n_estimators': int(HP_F[1])
        , 'learning_rate': float(HP_F[2])
        , 'gamma': float(HP_F[3])
        , 'min_child_weight': int(HP_F[4])
        , 'subsample': float(HP_F[5])
        , 'colsample_bytree': float(HP_F[6])
        , 'colsample_bylevel': float(HP_F[7])
        , 'reg_alpha': float(HP_F[8])
        , 'reg_lambda': float(HP_F[9])
            }

    for o in range(len(HP_F[len(HP):])):
        if HP_F[len(HP) + o] > 0.4:
            HP_F[len(HP) + o] = 1
        else:
            HP_F[len(HP) + o] = 0
    random_col = np.where(np.array([round(i) for i in HP_F[len(HP):]]) == 1)[0].tolist()
    if len(random_col) == 0:
        random_col = np.where(np.array([round(i) for i in HP_F[len(HP):]]) == 0)[0].tolist()
    x_train_selected = train.iloc[:, random_col]
    x_test_selected = test.iloc[:, random_col]

    XGBR = xgb.XGBRegressor(**HP_dic)
    XGBR.fit(x_train_selected, y_train)

    y_pred = XGBR.predict(x_test_selected)
    y_ypred = y_test - y_pred
    var_y = statistics.variance(y_test)
    var_yy = statistics.variance(y_ypred, statistics.mean(y_ypred))
    vaf = 1 - var_yy / var_y
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    r2 = r2_score(y_test, y_pred)
    aic = np.log10(mean_squared_error(y_test, y_pred)) * len(y_train) + 2 * (len(y_test) + 1)
    bic = np.log10(mean_squared_error(y_test, y_pred)) * len(y_train) + np.log10(len(y_train)) * (len(y_test) + 1)

    print('RMSE = ', rmse)
    print('MAE = ', mae)
    print('R2:', r2)
    print('correlation:', corr)
    print('VAF:', vaf)
    print("AIC:", aic)
    print("BIC:", bic)

