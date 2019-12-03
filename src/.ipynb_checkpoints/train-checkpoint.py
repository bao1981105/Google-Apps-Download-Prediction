# -*- coding: utf-8 -*-
pip install xgboost

import pandas as pd
import numpy as np

df = pd.read_csv('../data/googleplaystore.csv')
df2 = pd.read_csv('../data/googleplaystore_user_reviews.csv')

from mcar import mcar_test
from preprocess import preprocess

df_temp = preprocess(df,df2)
mcar_test(df_temp[['Rating','Size']])

import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from sklearn.model_selection import ParameterGrid
cont_frs_standard = ['Reviews','Size','Price','Last Updated','Num_of_Characters','Rating']
cat_frs_onehot = ['Category','Current_Ver_truncated','Android_Ver_truncated']
cat_frs_ordinal = ['Type','Content Rating']

X = df_temp.drop(['Installs'], axis=1)
y = df_temp['Installs']
print(y.value_counts()/len(y))
classes, counts = np.unique(y,return_counts=True)
print('balance:',np.max(counts/len(y)))


# +
def preprocess_with_impute(X,y,random_state,n_folds):
    '''
    
    '''
    X_other, X_test, y_other, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state,stratify=y)
    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=random_state)
    imputer = IterativeImputer(estimator = RandomForestRegressor(),random_state=random_state)
    standard_transformer = Pipeline(steps=[('imputer', imputer),('standard', StandardScaler())])
    onehot_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse=False,handle_unknown='ignore'))])
    ordinal_transformer1 = Pipeline(steps=[('ordinal1', OrdinalEncoder(categories = [['Free','Paid']]))])
    ordinal_transformer2 = Pipeline(steps=[('ordinal2', OrdinalEncoder(categories = [['Everyone', 'Everyone 10+', 'Teen', 'Mature 17+', 'Adults only 18+','Unrated']]))])
    preprocessor = ColumnTransformer(
    transformers=[
        ('standard', standard_transformer, cont_frs_standard),
        ('ordinal1',ordinal_transformer1,['Type']),
        ('ordinal2',ordinal_transformer2,['Content Rating']),
        ('onehot', onehot_transformer, cat_frs_onehot)])
    return X_other, X_test, y_other, y_test, kf, preprocessor

def ML_pipeline_rf_GridSearchCV(X_other, X_test, y_other, y_test, kf, preprocessor, random_state):
    rf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier())])
    param_grid = { 
    'classifier__max_features': ['auto'],
    'classifier__max_depth' : range(30,65,5),
    'classifier__min_samples_split' : range(2,5),
    'classifier__n_estimators' : [100],
    'classifier__random_state' : [random_state]}
    grid = GridSearchCV(rf, param_grid=param_grid,scoring = make_scorer(accuracy_score),
                        cv=kf, return_train_score = True,iid=True)
    grid.fit(X_other, y_other)
    return grid, grid.score(X_test, y_test)


# -

test_scores_rf = []
for i in range(8):
    random_state = (i + 1) * 42
    X_other, X_test, y_other, y_test, kf, preprocessor = preprocess_with_impute(X,y,random_state = random_state,n_folds=5)
    grid, test_score = ML_pipeline_rf_GridSearchCV(X_other, X_test, y_other, y_test, kf, preprocessor, random_state)
    print(grid.best_params_)
    print('test score:',test_score)
    test_scores_rf.append(test_score)
print('test accuracy:',np.around(np.mean(test_scores_rf),3),'+/-',np.around(np.std(test_scores_rf),3))


def ML_pipeline_svc_GridSearchCV(X_other, X_test, y_other, y_test, kf, preprocessor, random_state):
    estimators = Pipeline([('reduce_dim', PCA()), ('clf', SVC())])
    svc = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', estimators)])
    Cs = np.logspace(3,7,5)
    γs = np.logspace(-4,2,3)
    param_grid = { 
    'classifier__reduce_dim__n_components': [25],
    'classifier__clf__C': Cs,
    'classifier__clf__gamma' : γs}
    grid = GridSearchCV(svc, param_grid=param_grid,scoring = make_scorer(accuracy_score),
                        cv=kf, return_train_score = True,iid=True,n_jobs=-1)
    grid.fit(X_other, y_other)
    return grid, grid.score(X_test, y_test)


test_scores_svc = []
for i in range(2):
    random_state = (i + 1) * 42
    X_other, X_test, y_other, y_test, kf, preprocessor = preprocess_with_impute(X,y,random_state = random_state,n_folds=4)
    grid, test_score = ML_pipeline_svc_GridSearchCV(X_other, X_test, y_other, y_test, kf, preprocessor, random_state)
    print(grid.best_params_)
    print('test score:',test_score)
    test_scores_svc.append(test_score)
print('test accuracy:',np.around(np.mean(test_scores_svc),3),'+/-',np.around(np.std(test_scores_svc),3))


def ML_pipeline_knn_GridSearchCV(X_other, X_test, y_other, y_test, kf, preprocessor, random_state):
    knn = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', KNeighborsClassifier())])
    param_grid = {'classifier__n_neighbors': [20,25,30,35,40,45]}
    grid = GridSearchCV(knn, param_grid=param_grid,scoring = make_scorer(accuracy_score),
                    cv=kf, return_train_score = True,iid=True,n_jobs=-1)
    grid.fit(X_other, y_other)
    return grid, grid.score(X_test, y_test)


test_scores_knn = []
for i in range(8):
    random_state = (i + 1) * 42
    X_other, X_test, y_other, y_test, kf, preprocessor = preprocess_with_impute(X,y,random_state = random_state,n_folds=5)
    grid, test_score = ML_pipeline_knn_GridSearchCV(X_other, X_test, y_other, y_test, kf, preprocessor, random_state)
    print(grid.best_params_)
    print('test score:',test_score)
    test_scores_knn.append(test_score)
print('test accuracy:',np.around(np.mean(test_scores_knn),3),'+/-',np.around(np.std(test_scores_knn),3))


def ML_pipeline_xgb_GridSearchCV(X, y, random_state, n_folds):
    X_other, X_test, y_other, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state,stratify=y)
    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state = random_state)
    standard_transformer = Pipeline(steps=[('standard', StandardScaler())])
    onehot_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse=False,handle_unknown='ignore'))])
    ordinal_transformer1 = Pipeline(steps=[('ordinal1', OrdinalEncoder(categories = [['Free','Paid']]))])
    ordinal_transformer2 = Pipeline(steps=[('ordinal2', OrdinalEncoder(categories = [['Everyone', 'Everyone 10+', 'Teen', 'Mature 17+', 'Adults only 18+','Unrated']]))])                              
    XGB = xgboost.XGBClassifier()
    preprocessor = ColumnTransformer(
    transformers=[
        ('standard', standard_transformer, cont_frs_standard),
        ('ordinal1',ordinal_transformer1,['Type']),
        ('ordinal2',ordinal_transformer2,['Content Rating']),
        ('onehot', onehot_transformer, cat_frs_onehot)])

    param_grid = {'classifier__learning_rate': [0.03],
              'classifier__n_estimators': [100],
              'classifier__random_state': [random_state],
              'classifier__missing': [np.nan], 
              'classifier__max_depth': [20,30,40,50],
              'classifier__colsample_bytree': [0.7],              
              'classifier__subsample': [0.68]}

    xgb = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', XGB)])

    grid = GridSearchCV(xgb, param_grid=param_grid,scoring = make_scorer(accuracy_score),
                    cv=kf, return_train_score = True,iid=True,n_jobs=-1)
    grid.fit(X_other, y_other)
    return grid, grid.score(X_test, y_test)


test_scores_xgb = []
for i in range(8):
    random_state = (i + 1) * 42
    grid, test_score = ML_pipeline_xgb_GridSearchCV(X, y, random_state, 5)
    print(grid.best_params_)
    print('test score:',test_score)
    test_scores_xgb.append(test_score)
print('test accuracy:',np.around(np.mean(test_scores_xgb),3),'+/-',np.around(np.std(test_scores_xgb),3))


