#!/usr/bin/env python
# coding: utf-8


#%matplotlib qt 

import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from imblearn.metrics import specificity_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from scipy.stats import iqr
from scipy.stats import t

# import data
data = pd.read_csv('bm_combat_spectral_changed.csv', index_col=[0])

splitted_data = {'non-harm': data.drop(data.iloc[:, 6:209], axis=1), 
                 'harm': data.drop(data.iloc[:, 209:], axis=1)}
# train/test split
np.random.seed(0)
random.seed(0)

data_types = ['non-harm','harm']
# center- and gender-wise train/test split
center_names = ['all','california','finland','iowa','medellin']
genders = ['both','m','f']

# train
X_train_sets = {data_type: {center_name: {gender: None for gender in genders} for center_name in center_names} for data_type in data_types} 
y_train_sets = {data_type: {center_name: {gender: None for gender in genders} for center_name in center_names} for data_type in data_types}

# test
X_test_sets = {data_type: {center_name: {gender: None for gender in genders} for center_name in center_names} for data_type in data_types}
X_test_sets_strat = {data_type: {center_name: {gender: None for gender in genders} for center_name in center_names} for data_type in data_types}
y_test_sets = {data_type: {center_name: {gender: None for gender in genders} for center_name in center_names} for data_type in data_types}

for data_type in data_types:
    X = splitted_data[data_type]
    y = splitted_data[data_type][['center','label','gender']]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.7,
                                                    random_state=1,
                                                    stratify=X[['center','group']])
    for name in center_names:
    
        if name == 'all':

            for gender in genders:

                if gender == 'both':

                    # train

                    X_train_sets[data_type][name][gender] = X_train.drop(['center','group','age','gender','batch','label'], axis=1)

                    y_train_sets[data_type][name][gender] = y_train.drop(['center','gender'], axis=1)

                    # test

                    X_test_sets_strat[data_type][name][gender] = X_test
                    X_test_sets[data_type][name][gender] = X_test.drop(['center','group','age','gender','batch','label'], axis=1)

                    y_test_sets[data_type][name][gender] = y_test.drop(['center','gender'], axis=1)

                else:

                    # train

                    train_sets = X_train.loc[X_train['gender'] == gender] 
                    X_train_sets[data_type][name][gender] = train_sets.drop(['center','group','age','gender','batch','label'], axis=1)

                    train_label_sets = y_train.loc[y_train['gender'] == gender]
                    y_train_sets[data_type][name][gender] = train_label_sets.drop(['center','gender'], axis=1)    

                    # test

                    test_sets = X_test.loc[X_test['gender'] == gender]
                    X_test_sets_strat[data_type][name][gender] = test_sets 
                    X_test_sets[data_type][name][gender] = test_sets.drop(['center','group','age','gender','batch','label'], axis=1)

                    test_label_sets = y_test.loc[y_test['gender'] == gender]
                    y_test_sets[data_type][name][gender] = test_label_sets.drop(['center','gender'], axis=1)


        else:

            for gender in genders:

                if gender == 'both':

                    # train

                    X_train_temp = X_train.loc[X_train['center'] == name]
                    X_train_sets[data_type][name][gender] = X_train_temp.drop(['center','group','age','gender','batch','label'], axis=1)

                    y_train_temp = y_train.loc[y_train['center'] == name]                
                    y_train_sets[data_type][name][gender] = y_train_temp.drop(['center','gender'], axis=1)

                    # test

                    X_test_temp = X_test.loc[X_test['center'] == name]
                    X_test_sets_strat[data_type][name][gender] = X_test_temp
                    X_test_sets[data_type][name][gender] = X_test_temp.drop(['center','group','age','gender','batch','label'], axis=1)

                    y_test_temp = y_test.loc[y_test['center'] == name]
                    y_test_sets[data_type][name][gender] = y_test_temp.drop(['center','gender'], axis=1)

                else:

                    # train

                    X_train_temp = X_train.loc[(X_train['center'] == name) & (X_train['gender'] == gender)]
                    X_train_sets[data_type][name][gender] = X_train_temp.drop(['center','group','age','gender','batch','label'], axis=1)

                    y_train_temp = y_train.loc[(y_train['center'] == name) & (y_train['gender'] == gender)]                
                    y_train_sets[data_type][name][gender] = y_train_temp.drop(['center','gender'], axis=1)

                    # test

                    X_test_temp = X_test.loc[(X_test['center'] == name) & (X_test['gender'] == gender)]
                    X_test_sets_strat[data_type][name][gender] = X_test_temp
                    X_test_sets[data_type][name][gender] = X_test_temp.drop(['center','group','age','gender','batch','label'], axis=1)

                    y_test_temp = y_test.loc[(y_test['center'] == name) & (y_test['gender'] == gender)]
                    y_test_sets[data_type][name][gender] = y_test_temp.drop(['center','gender'], axis=1)


# Initializing Classifiers

clf1 = LogisticRegression(multi_class='multinomial',
                          solver='newton-cg',
                          random_state=1)

clf2 = KNeighborsClassifier(algorithm='ball_tree',
                            leaf_size=50)

clf3 = DecisionTreeClassifier(random_state=1)

clf4 = SVC(random_state=1)


# Building the pipelines

pipe1 = Pipeline([#('std', StandardScaler()),
                  #('fs', SequentialFeatureSelector(estimator = clf1)),
                  ('clf1', clf1)])

pipe2 = Pipeline([#('std', StandardScaler()),
                  #('fs', SequentialFeatureSelector(estimator = clf2)),
                  ('clf2', clf2)])

#pipe3 = Pipeline([('std', StandardScaler()),
#                  ('fs', RFE(estimator = clf3)),
#                  ('clf3', clf3)])

pipe4 = Pipeline([#('std', StandardScaler()),
                  #('fs', SequentialFeatureSelector(estimator = clf4)),
                  ('clf4', clf4)])


# Setting up the parameter grids
param_grid1 = [{
                #'fs__n_features_to_select': np.linspace(1, 203, num=203, dtype=int),
                #'fs__step': [1],
                'clf1__penalty': ['l2'],
                'clf1__C': np.power(10., np.arange(-4, 4))}]

param_grid2 = [{
                #'fs__n_features_to_select': np.linspace(1, 203, num=203, dtype=int),
                #'fs__step': [1],
                'clf2__n_neighbors': list(range(1, 10)),
                'clf2__p': [1, 2]}]

param_grid3 = [{
                'max_depth': list(range(1, 10)) + [None],
                'criterion': ['gini', 'entropy']}]

# 'model__C': [0.1, 1, 10, 25, 50, 75, 100, 150, 1000, 10000, 100000]
# 'model__gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

param_grid4 = [
               {
                #'fs__n_features_to_select': np.linspace(1, 203, num=203, dtype=int),
                #'fs__step': [1],
                'clf4__kernel': ['rbf'],
                'clf4__C': np.power(10., np.arange(-4, 4)),
                'clf4__gamma': np.power(10., np.arange(-5, 0))},
               {
                #'fs__n_features_to_select': np.linspace(1, 203, num=203, dtype=int),
                #'fs__step': [1],
                'clf4__kernel': ['linear'],
                'clf4__C': np.power(10., np.arange(-4, 4))}
              ]

param_grid5 = [{
                'max_depth': list(range(1, 10)) + [None],
                'criterion': ['gini', 'entropy']}]

# Setting up multiple GridSearchCV objects, 1 for each algorithm

gridcvs = {}

#for pgrid, est, name in zip((param_grid1, param_grid2,
#                             param_grid3, param_grid4),
#                           (pipe1, pipe2, clf3, pipe4),
#                            ('Softmax', 'KNN', 'DTree', 'SVM')):
    
for pgrid, est, name in zip((param_grid1, param_grid4),
                            (pipe1, pipe4),
                            ('LR', 'SVM')):
    
    #print(pgrid)
    #print(est)
    #print(name)
    
    gcv = GridSearchCV(estimator=est,
                       param_grid=pgrid,
                       scoring='accuracy',
                       n_jobs=1,
                       cv=5,
                       verbose=0,
                       refit=True)
    gridcvs[name] = gcv


# # FEATURE SELECTION METHOD: SELECTKBEST


list_of_k = np.linspace(1, 203, num=203, dtype=int)
metrics = ['accuracy','recall','specificity','precision','F1','auc']

cv_scores = {data_type: {k: {center_name: {gender: {name: {metric: [] for metric in metrics} for name, gs_est in gridcvs.items()} for gender in genders} for center_name in center_names} for k in list_of_k} for data_type in data_types}
cv_scores_without_zeros = {data_type: {k: {center_name: {gender: {name: {metric: [] for metric in metrics} for name, gs_est in gridcvs.items()} for gender in genders} for center_name in center_names} for k in list_of_k} for data_type in data_types}

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

features_ids = {data_type: {k: {center_name: {gender: [] for gender in genders} for center_name in center_names} for k in list_of_k} for data_type in data_types}
features_names = {data_type: {k: {center_name: {gender: [] for gender in genders} for center_name in center_names} for k in list_of_k} for data_type in data_types}

for data_type in data_types:

    for k in list_of_k:

        for center_name in center_names:

            if center_name != center_names[0]: continue

            for gender in genders:           

                if gender != genders[0]: continue     

                print('Cross-validation + feature selection for number of features of ' + str(k) + ' for ' + center_name.upper() + ' for ' + gender.upper())

                # The outer loop for algorithm selection
                c = 1
                for outer_train_idx, outer_valid_idx in skfold.split(X_train_sets[data_type][center_name][gender],y_train_sets[data_type][center_name][gender]):

                    # feature selection: starts here

                    selector = SelectKBest(score_func=f_classif, k=k)

                    # run score function on (X,y) and get the appropriate features
                    fit = selector.fit(X_train_sets[data_type][center_name][gender].iloc[outer_train_idx.tolist()], y_train_sets[data_type][center_name][gender].iloc[outer_train_idx.tolist()].values.ravel())

                    # Get columns to keep and create new dataframe with those only
                    selected_features = fit.get_support(indices=True)

                    features = X_train_sets[data_type][center_name][gender].columns[selected_features].to_list()

                    features_ids[data_type][k][center_name][gender].append(selected_features)
                    features_names[data_type][k][center_name][gender].append(features)

                    # feature selection: ends here

                    for name, gs_est in sorted(gridcvs.items()):
                        print('outer fold %d/5 | tuning %-8s\n' % (c, name), end='')

                        # The inner loop for hyperparameter tuning

                        if len(features) != 0:

                            gs_est.fit(X_train_sets[data_type][center_name][gender].iloc[outer_train_idx.tolist()].iloc[:,selected_features], y_train_sets[data_type][center_name][gender].iloc[outer_train_idx.tolist()].values.ravel())
                            y_pred = gs_est.predict(X_train_sets[data_type][center_name][gender].iloc[outer_valid_idx.tolist()].iloc[:,selected_features])

                            for metric in metrics: 
                                if metric == 'accuracy':
                                    calc_metric = accuracy_score(y_true=y_train_sets[data_type][center_name][gender].iloc[outer_valid_idx.tolist()], y_pred=y_pred)
                                    #print(' | inner ACC %.2f%% | outer ACC %.2f%%' %
                                    #      (gs_est.best_score_ * 100, calc_metric * 100))
                                    cv_scores[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                    if calc_metric != 0:
                                        cv_scores_without_zeros[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                elif metric == 'recall':
                                    calc_metric = recall_score(y_true=y_train_sets[data_type][center_name][gender].iloc[outer_valid_idx.tolist()], y_pred=y_pred)
                                    cv_scores[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                    if calc_metric != 0:
                                        cv_scores_without_zeros[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                elif metric == 'specificity':
                                    calc_metric = specificity_score(y_true=y_train_sets[data_type][center_name][gender].iloc[outer_valid_idx.tolist()], y_pred=y_pred)
                                    cv_scores[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                    if calc_metric != 0:
                                        cv_scores_without_zeros[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                elif metric == 'precision':
                                    calc_metric = precision_score(y_true=y_train_sets[data_type][center_name][gender].iloc[outer_valid_idx.tolist()], y_pred=y_pred)
                                    cv_scores[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                    if calc_metric != 0:
                                        cv_scores_without_zeros[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                elif metric == 'F1':
                                    calc_metric = f1_score(y_true=y_train_sets[data_type][center_name][gender].iloc[outer_valid_idx.tolist()], y_pred=y_pred)
                                    cv_scores[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                    if calc_metric != 0:
                                        cv_scores_without_zeros[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                else:
                                    calc_metric = roc_auc_score(y_true=y_train_sets[data_type][center_name][gender].iloc[outer_valid_idx.tolist()], y_score=y_pred, average = 'macro')
                                    cv_scores[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                    if calc_metric != 0:
                                        cv_scores_without_zeros[data_type][k][center_name][gender][name][metric].append(calc_metric)
                        else:

                            print('!!!NO FEATURES WERE SELECTED!!!')

                            gs_est.fit(X_train_sets[data_type][center_name][gender].iloc[outer_train_idx.tolist()], y_train_sets[data_type][center_name][gender].iloc[outer_train_idx.tolist()].values.ravel())
                            y_pred = gs_est.predict(X_train_sets[data_type][center_name][gender].iloc[outer_valid_idx.tolist()])

                            for metric in metrics: 
                                #print(center_name,name,metric)
                                if metric == 'accuracy':
                                    calc_metric = accuracy_score(y_true=y_train_sets[data_type][center_name][gender].iloc[outer_valid_idx.tolist()], y_pred=y_pred)
                                    #print(' | inner ACC %.2f%% | outer ACC %.2f%%' %
                                    #      (gs_est.best_score_ * 100, calc_metric * 100))
                                    cv_scores[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                    if calc_metric != 0:
                                        cv_scores_without_zeros[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                elif metric == 'recall':
                                    calc_metric = recall_score(y_true=y_train_sets[data_type][center_name][gender].iloc[outer_valid_idx.tolist()], y_pred=y_pred)
                                    cv_scores[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                    if calc_metric != 0:
                                        cv_scores_without_zeros[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                elif metric == 'specificity':
                                    calc_metric = specificity_score(y_true=y_train_sets[data_type][center_name][gender].iloc[outer_valid_idx.tolist()], y_pred=y_pred)
                                    cv_scores[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                    if calc_metric != 0:
                                        cv_scores_without_zeros[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                elif metric == 'precision':
                                    calc_metric = precision_score(y_true=y_train_sets[data_type][center_name][gender].iloc[outer_valid_idx.tolist()], y_pred=y_pred)
                                    cv_scores[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                    if calc_metric != 0:
                                        cv_scores_without_zeros[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                elif metric == 'F1':
                                    calc_metric = f1_score(y_true=y_train_sets[data_type][center_name][gender].iloc[outer_valid_idx.tolist()], y_pred=y_pred)
                                    cv_scores[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                    if calc_metric != 0:
                                        cv_scores_without_zeros[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                else:
                                    calc_metric = roc_auc_score(y_true=y_train_sets[data_type][center_name][gender].iloc[outer_valid_idx.tolist()], y_score=y_pred, average = 'macro')
                                    cv_scores[data_type][k][center_name][gender][name][metric].append(calc_metric)
                                    if calc_metric != 0:
                                        cv_scores_without_zeros[data_type][k][center_name][gender][name][metric].append(calc_metric)


                    c += 1


# Looking at the validation results

mean_std_cv_scores = {data_type: {k: {center_name: {gender: {name: {metric: {'mean': None, 'SD': None} for metric in metrics} for name, gs_est in gridcvs.items()} for gender in genders} for center_name in center_names} for k in list_of_k} for data_type in data_types}
mean_std_cv_scores_without_zeros = {data_type: {k: {center_name: {gender: {name: {metric: {'mean': None, 'SD': None} for metric in metrics} for name, gs_est in gridcvs.items()} for gender in genders} for center_name in center_names} for k in list_of_k} for data_type in data_types}
median_iqr_cv_scores = {data_type: {k: {center_name: {gender: {name: {metric: {'median': None, 'IQR': None} for metric in metrics} for name, gs_est in gridcvs.items()} for gender in genders} for center_name in center_names} for k in list_of_k} for data_type in data_types}
median_iqr_cv_scores_without_zeros = {data_type: {k: {center_name: {gender: {name: {metric: {'median': None, 'IQR': None} for metric in metrics} for name, gs_est in gridcvs.items()} for gender in genders} for center_name in center_names} for k in list_of_k} for data_type in data_types}

for data_type in data_types:
    
    for k in list_of_k:

        for center_name in center_names:

            for gender in genders:

                for name, gs_est in sorted(gridcvs.items()):
                    #print('Results for ' + center_name.upper() + gender.upper() + ' ' + str(k) + ' ' + name.upper())
                    for metric in metrics: 
                        if metric != 'auc':
                            #print('%-8s | outer CV ' + metric + ' %.2f%% +\- %.3f' % (
                            #  100 * np.mean(cv_scores[k][center_name][name][metric]), 100 * np.std(cv_scores[k][center_name][name][metric])))
                            mean_std_cv_scores[data_type][k][center_name][gender][name][metric]['mean'] = np.mean(cv_scores[data_type][k][center_name][gender][name][metric])
                            mean_std_cv_scores_without_zeros[data_type][k][center_name][gender][name][metric]['mean'] = np.mean(cv_scores_without_zeros[data_type][k][center_name][gender][name][metric])
                            mean_std_cv_scores[data_type][k][center_name][gender][name][metric]['SD'] = np.std(cv_scores[data_type][k][center_name][gender][name][metric])
                            mean_std_cv_scores_without_zeros[data_type][k][center_name][gender][name][metric]['SD'] = np.std(cv_scores_without_zeros[data_type][k][center_name][gender][name][metric])

                            median_iqr_cv_scores[data_type][k][center_name][gender][name][metric]['median'] = 100*np.median(cv_scores[data_type][k][center_name][gender][name][metric])
                            median_iqr_cv_scores_without_zeros[data_type][k][center_name][gender][name][metric]['median'] = 100*np.median(cv_scores_without_zeros[data_type][k][center_name][gender][name][metric])
                            median_iqr_cv_scores[data_type][k][center_name][gender][name][metric]['IQR'] = 100*iqr(cv_scores[data_type][k][center_name][gender][name][metric])
                            median_iqr_cv_scores_without_zeros[data_type][k][center_name][gender][name][metric]['IQR'] = 100*iqr(cv_scores_without_zeros[data_type][k][center_name][gender][name][metric])
                        else: 
                            #print('%-8s | outer CV ' + metric + ' %.2f +\- %.3f' % (
                            #  np.mean(cv_scores[k][center_name][name][metric]), np.std(cv_scores[k][center_name][name][metric])))
                            mean_std_cv_scores[data_type][k][center_name][gender][name][metric]['mean'] = np.mean(cv_scores[data_type][k][center_name][gender][name][metric])
                            mean_std_cv_scores_without_zeros[data_type][k][center_name][gender][name][metric]['mean'] = np.mean(cv_scores_without_zeros[data_type][k][center_name][gender][name][metric])
                            mean_std_cv_scores[data_type][k][center_name][gender][name][metric]['SD'] = np.std(cv_scores[data_type][k][center_name][gender][name][metric])
                            mean_std_cv_scores_without_zeros[data_type][k][center_name][gender][name][metric]['SD'] = np.std(cv_scores_without_zeros[data_type][k][center_name][gender][name][metric])

                            median_iqr_cv_scores[data_type][k][center_name][gender][name][metric]['median'] = 100*np.median(cv_scores[data_type][k][center_name][gender][name][metric])
                            median_iqr_cv_scores_without_zeros[data_type][k][center_name][gender][name][metric]['median'] = 100*np.median(cv_scores_without_zeros[data_type][k][center_name][gender][name][metric])
                            median_iqr_cv_scores[data_type][k][center_name][gender][name][metric]['IQR'] = 100*iqr(cv_scores[data_type][k][center_name][gender][name][metric])
                            median_iqr_cv_scores_without_zeros[data_type][k][center_name][gender][name][metric]['IQR'] = 100*iqr(cv_scores_without_zeros[data_type][k][center_name][gender][name][metric])
                #print('*********************')

cv_acc_models = {data_type: {gender: {metric: {name: {'mean': [], 'SD': []} for name, gs_est in sorted(gridcvs.items())} for metric in metrics} for gender in genders} for data_type in data_types}

for data_type in data_types:
    for k in list_of_k:
        for gender in genders:
            for name, gs_est in sorted(gridcvs.items()):
                for metric in metrics:
                    cv_acc_models[data_type][gender][metric][name]['mean'].append(mean_std_cv_scores[data_type][k]['all'][gender][name][metric]['mean'])
                    cv_acc_models[data_type][gender][metric][name]['SD'].append(mean_std_cv_scores[data_type][k]['all'][gender][name][metric]['SD'])


best_features = {data_type: {gender: {name: None for name, gs_est in sorted(gridcvs.items())} for gender in genders} for data_type in data_types}

for data_type in data_types:
    for gender in genders:
        for name, gs_est in sorted(gridcvs.items()):
            f = cv_acc_models[data_type][gender]['accuracy'][name]['mean'].index(max(cv_acc_models[data_type][gender]['accuracy'][name]['mean']))+1
            best_features[data_type][gender][name] = f
            print('Number of features chosen for ' + data_type.upper() + ' data ' + gender.upper() + ' for ' + name + ' is ' + str(best_features[data_type][gender][name]))
            print('giving the following validation results')
            print(mean_std_cv_scores[data_type][f]['all'][gender][name])


merged_features_ids = {data_type: {k: {center_name: {gender: [] for gender in genders} for center_name in center_names} for k in list_of_k} for data_type in data_types}
merged_features_names = {data_type: {k: {center_name: {gender: [] for gender in genders} for center_name in center_names} for k in list_of_k} for data_type in data_types}

for data_type in data_types:

    for k in list_of_k:

        for center_name in center_names:

            for gender in genders:

                #print(center_name)
                for i in range(len(features_ids[data_type][k][center_name][gender])):
                    #print(i)
                    length = len(features_ids[data_type][k][center_name][gender][i])
                    #print(length)
                    for j in range(length):
                        merged_features_ids[data_type][k][center_name][gender].append(features_ids[data_type][k][center_name][gender][i][j])
                        merged_features_ids[data_type][k][center_name][gender] = list(set(merged_features_ids[data_type][k][center_name][gender]))
                        merged_features_names[data_type][k][center_name][gender].append(features_names[data_type][k][center_name][gender][i][j])


# # TESTING, BOOTSTRAPING


# stratified BOOTSTRAPPING

from sklearn.utils import resample

confusion_matrices = {center_name: {gender: {name: [] for name, gs_est in gridcvs.items()} for gender in genders} for center_name in center_names}
strat_bootstrap = {center_name: {gender: {name: {metric: [] for metric in metrics} for name, gs_est in gridcvs.items()} for gender in genders} for center_name in center_names}
strat_bootstrap_without_zeros = {center_name: {gender: {name: {metric: [] for metric in metrics} for name, gs_est in gridcvs.items()} for gender in genders} for center_name in center_names}

for name, gs_est in sorted(gridcvs.items()):
    
    # Fitting a model to the whole training set
    # using the "best" algorithm
    best_algo = gridcvs[name]
    print(best_algo.best_params_)
    #print(best_algo)
    
    if name != 'LR': continue

    for center_name in center_names:

        for gender in genders:
        
            if center_name != 'all' and gender != 'both': continue        

            f = best_features['harm']['both'][name]                     

            n_iterations = 100  # No. of bootstrap samples to be repeated (created)
            n_size = int(len(X_test_sets['harm'][center_name][gender]) * 1.0) # Size of sample, picking 100% of the given data in every bootstrap sample

            print('Test results for ' + center_name.upper() + ' for ' + gender.upper() + ' for ' + name)
            #print(n_size)

            for i in range(n_iterations):

                if center_name == 'all':

                    test_values = resample(X_test_sets_strat['harm'][center_name][gender],n_samples=n_size,stratify=X_test_sets_strat['harm'][center_name][gender][['center','gender','group']])

                else: 

                    test_values = resample(X_test_sets_strat['harm'][center_name][gender],n_samples = n_size,stratify=X_test_sets_strat['harm'][center_name][gender][['gender','group']])

                test_values = test_values.drop(['center','group','age','gender','batch','label'], axis=1)

                test_labels = y_test_sets['harm'][center_name][gender].loc[test_values.index.to_list()]

                best_algo.fit(X_train_sets['harm']['all']['both'].iloc[:,merged_features_ids['harm'][f]['all']['both']],y_train_sets['harm']['all']['both'].values.ravel())

                if i == 0:
                    #print(confusion_matrix(y_true=test_labels, y_pred=best_algo.predict(test_values.iloc[:,merged_features_ids[202]['all']])))
                    confusion_matrices[center_name][gender][name] = confusion_matrix(y_true=test_labels, y_pred=best_algo.predict(test_values.iloc[:,merged_features_ids['harm'][f]['all']['both']]))
                    #print(confusion_matrices_both[center_name])
                else:
                    #print(confusion_matrix(y_true=test_labels, y_pred=best_algo.predict(test_values.iloc[:,merged_features_ids[202]['all']])))
                    confusion_matrices[center_name][gender][name] += confusion_matrix(y_true=test_labels, y_pred=best_algo.predict(test_values.iloc[:,merged_features_ids['harm'][f]['all']['both']]))
                    #print(confusion_matrices_both[center_name])

                predicted_labels = best_algo.predict(test_values.iloc[:,merged_features_ids['harm'][f]['all']['both']])

                # metrics

                for metric in metrics: 

                    try:
                        roc_auc_score(y_true=test_labels, y_score=predicted_labels, average = 'macro')

                        if metric == 'accuracy':
                            calc_metric = accuracy_score(y_true=test_labels, y_pred=predicted_labels)
                            #print('Test Accuracy: %.2f%%' % (100 * calc_metric))
                            strat_bootstrap[center_name][gender][name][metric].append(calc_metric)
                            if calc_metric != 0:
                                strat_bootstrap_without_zeros[center_name][gender][name][metric].append(calc_metric)
                        elif metric == 'recall':
                            calc_metric = recall_score(y_true=test_labels, y_pred=predicted_labels)
                            #print('Recall/sensitivity: %.2f%%' % (100 * calc_metric))
                            strat_bootstrap[center_name][gender][name][metric].append(calc_metric)
                            if calc_metric != 0:
                                strat_bootstrap_without_zeros[center_name][gender][name][metric].append(calc_metric)
                        elif metric == 'specificity':
                            calc_metric = specificity_score(y_true=test_labels, y_pred=predicted_labels)
                            #print('Specificity: %.2f%%' % (100 * calc_metric))
                            strat_bootstrap[center_name][gender][name][metric].append(calc_metric)
                            if calc_metric != 0:
                                strat_bootstrap_without_zeros[center_name][gender][name][metric].append(calc_metric)
                        elif metric == 'precision':
                            calc_metric = precision_score(y_true=test_labels, y_pred=predicted_labels)
                            #print('Precision: %.2f%%' % (100 * calc_metric))
                            strat_bootstrap[center_name][gender][name][metric].append(calc_metric)
                            if calc_metric != 0:
                                strat_bootstrap_without_zeros[center_name][gender][name][metric].append(calc_metric)
                        elif metric == 'F1':
                            calc_metric = f1_score(y_true=test_labels, y_pred=predicted_labels)
                            #print('F1 score: %.2f%%' % (100 * calc_metric))
                            strat_bootstrap[center_name][gender][name][metric].append(calc_metric)
                            if calc_metric != 0:
                                strat_bootstrap_without_zeros[center_name][gender][name][metric].append(calc_metric)
                        else:
                            calc_metric = roc_auc_score(y_true=test_labels, y_score=predicted_labels, average = 'macro')
                            #print('ROC AUC: %.2f' % calc_metric)
                            strat_bootstrap[center_name][gender][name][metric].append(calc_metric)
                            if calc_metric != 0:
                                strat_bootstrap_without_zeros[center_name][gender][name][metric].append(calc_metric)
                    except ValueError:

                        pass


for gender in genders:
    for name, gs_est in sorted(gridcvs.items()):
        
        if name != 'LR': continue
                               
        print(gender)
        print(name)

        color = 'white'
        disp = ConfusionMatrixDisplay(confusion_matrix=np.rint(confusion_matrices['all'][gender][name]/100),display_labels=best_algo.classes_)
        disp.plot(cmap=plt.cm.Blues,values_format='g')
        plt.title('Confusion matrix for ' + gender + ' genders, ' + name)
        plt.show()


for center_name in center_names:

    for gender in genders:
        
        if center_name != 'all' and gender != 'both': continue
        
        for name, gs_est in sorted(gridcvs.items()):
            
            print('Test results for ' + center_name.upper() + ' for ' + gender.upper() + ' for ' + name)
            
            for metric in metrics: 
                if metric != 'auc':
                    print(metric + ' %.2f%% +\- %.3f' % (
                      100 * np.mean(strat_bootstrap[center_name][gender][name][metric]), 100 * np.std(strat_bootstrap[center_name][gender][name][metric])))
                else: 
                    print(metric + ' %.2f +\- %.3f' % (
                      np.mean(strat_bootstrap[center_name][gender][name][metric]), np.std(strat_bootstrap[center_name][gender][name][metric])))

