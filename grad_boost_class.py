import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import os
from sklearn.ensemble import GradientBoostingClassifier

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")

brupt = pd.read_csv("Bankruptcy.csv", index_col=0)
X = brupt.drop(['D', 'YR'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    train_size=0.7,
                                                    random_state=2022)

clf = GradientBoostingClassifier(random_state=2022)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = clf.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

################ Grid Search CV ####################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
params = {'n_estimators':[50,100,150],
          'max_depth':[1,2,3,4],
          'learning_rate':[0.01,0.1,0.15, 0.2,0.3]}
clf = GradientBoostingClassifier(random_state=2022)
print(clf.get_params())
gcv = GridSearchCV(clf, param_grid=params, cv=kfold,
                   verbose=3, scoring='roc_auc')
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

################### XGB ###########################
from xgboost import XGBClassifier

xgb_model = XGBClassifier(random_state=2022)
print(xgb_model.get_params())
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
params = {'n_estimators':[50,100,150],
          'max_depth':[1,2,3,4],
          'learning_rate':[0.01,0.1,0.15, 0.2,0.3]}

gcv = GridSearchCV(xgb_model, param_grid=params, cv=kfold,
                   verbose=3, scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

############## Image Segmentation ##########################
from sklearn.preprocessing import LabelEncoder
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Image Segmentation")

image_seg = pd.read_csv("Image_Segmention.csv")
X = image_seg.drop('Class', axis=1)
y = image_seg['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)

xgb_model = XGBClassifier(random_state=2022)
print(xgb_model.get_params())
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
params = {'n_estimators':[50,100,150],
          'max_depth':[1,2,3,4],
          'learning_rate':[0.01,0.1,0.15, 0.2,0.3]}
gcv = GridSearchCV(xgb_model, param_grid=params, cv=kfold,
                   verbose=3, scoring='neg_log_loss')
gcv.fit(X,le_y)
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

