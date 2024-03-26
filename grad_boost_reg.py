import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import os
from xgboost import XGBRegressor

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength")
concrete = pd.read_csv("Concrete_Data.csv")
X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

xgb_model = XGBRegressor(random_state=2022)
print(xgb_model.get_params())
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
params = {'n_estimators':[50,100,150],
          'max_depth':[1,2,3,4],
          'learning_rate':[0.01,0.1,0.15, 0.2,0.3]}

gcv = GridSearchCV(xgb_model, param_grid=params, cv=kfold,
                   verbose=3, scoring='r2')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


###### Feature Importance Plot ############
import matplotlib.pyplot as plt
best_model = gcv.best_estimator_
imps = best_model.feature_importances_

i_sorted = np.argsort(-imps)
n_sorted = X.columns[i_sorted]
imp_sort = imps[i_sorted]
plt.barh(n_sorted, imp_sort)
plt.title("Sorted Feature Importances")
plt.show()

