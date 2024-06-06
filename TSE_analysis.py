## Import importants libraries
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyemma
from pyemma.util.contexts import settings

A = msm.metastable_sets[2] ## define source state
B = msm.metastable_sets[4] ## define sink state
flux = pyemma.msm.tpt(msm, A, B) ## calculate flux 
cg, cgflux = flux.coarse_grain(msm.metastable_sets)
paths, path_fluxes = cgflux.pathways(fraction=0.99)
print('percentage       \tpath')
print('-------------------------------------')
for i in range(len(paths)):
    print(np.round(path_fluxes[i] / np.sum(path_fluxes), 3),' \t', paths[i] + 1)

## Committor probability estimation

df=pd.DataFrame(flux.committor[dtrajs_concatenated])
df.columns=['Comittor Probability']
df.reset_index()
df = df.rename(columns={"index":"Time Frame"})
df['Time Frame'] = df.index
## Merging committor probability file and OPs file together
df3=pd.merge(tsne_df1,df, on='Time Frame')
df4=pd.merge(df3,tsne_df2,on='Time Frame')
df4=df4.iloc[:,[0,2,3,4,6,7,8]]
df5=df4[(df4['Comittor Probability'] >=0) & (df4['Comittor Probability'] <=1)]
train_data=df5.to_numpy()
np.random.shuffle(train_data)
test_dataset=train_data[:25000]
train_dataset=train_data[25000:]
train_dataset_pd=pd.DataFrame(train_dataset)
#train_dataset_pd
test_dataset_pd=pd.DataFrame(test_dataset)
test_dataset_pd.columns =['N\u0063', 'end-to-end distance','Rg', 'Nw','committor probability']
train_dataset_pd.columns=['N\u0063', 'end-to-end distance','Rg', 'Nw','committor probability']

### we have taken testing dataset in the committor probability range (0.47-0.53) to make prediction in the TSE region

df1=test_dataset_pd[(test_dataset_pd['committor probability'] >=0.47) & (test_dataset_pd['committor probability'] <=0.53)]
X_test=df1.iloc[:,[0,1,2,3]] 
Y_test=df1.iloc[:,[4]]
X_train=train_dataset_pd.iloc[:,[0,1,2,3]] ## features in training set
Y_train=train_dataset_pd.iloc[:,[4]]       ## target variable in training set
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
xgb_r = xg.XGBRegressor(objective ='reg:linear',
                  n_estimators = 50, learning_rate = 0.05)

# Fitting the model
xgb_r.fit(X_train,Y_train)

# Predict the model
pred = xgb_r.predict(X_test)
#y_pred = model.predict(x_test).sum(axis=1)

# RMSE Computation
rmse = np.sqrt(MSE(Y_test, pred))
print("RMSE : % f" %(rmse))

import shap
import matplotlib.pylab as pl
explainer = shap.Explainer(xgb_r, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values,X_test,plot_type='bar')
#plt.savefig('c40-shap-comittor.png')

from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, X, y, cv, train_sizes=np.linspace(0.1, 1.0, 10)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='neg_mean_squared_error'
    )

    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title("Random Forest Regression - Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Mean Squared Error")
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation error")

    plt.legend(loc="best")
    plt.show()
from sklearn.model_selection import ShuffleSplit
def plot_learning_curve(estimator, X, y, cv=None, train_sizes=np.linspace(0.1, 1.0, 10)):
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='r2', n_jobs=-1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

# Create a cross-validation strategy
cv = ShuffleSplit(n_splits=10, random_state=42)
plot_learning_curve(xgb_r, X_train, Y_train, cv=cv)
xgb_r = xg.XGBRegressor(objective ='reg:linear',
                  n_estimators = 50, learning_rate = 0.05)

# Plot learning curve
plot_learning_curve(xgb_r, X_train, Y_train, cv=5)
import matplotlib.pyplot as plt
axes = plt.gca()

from yellowbrick.regressor import PredictionError
visualizer = PredictionError(xgb_r)

visualizer.fit(X_train, Y_train)
visualizer.score(X_test, Y_test)
visualizer.ax.set_xlim([0, 1])
visualizer.ax.set_ylim([0,1])
visualizer.ax.set_xlabel("True Committor Probability")
visualizer.ax.set_ylabel("Calculated committor probability")
visualizer.show(xlabel='$True Committor Probability$',ylabel='$Calculated committor probability$')
