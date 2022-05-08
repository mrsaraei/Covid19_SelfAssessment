# -*- coding: utf-8 -*-
print("------------------------------------------------------")
print("---------------- Metadata Information ----------------")
print("------------------------------------------------------")
print("")

print("In the name of God")
print("Project: Self-Assessment of COVID_19 Using Machine Learning")
print("Creator: Mohammad Reza Saraei")
print("Contact: m.r.saraei@seraj.ac.ir")
print("University: Seraj Institute of Higher Education")
print("Supervisor: Dr. Saman Rajebi")
print("Created Date: February 12, 2022")
print("") 

# print("----------------------------------------------------")
# print("------------------ Import Libraries ----------------")
# print("----------------------------------------------------")
# print("")

import time
import pandas as pd
import numpy as np
import random
import os
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# print("----------------------------------------------------")
# print("------------------ Data Ingestion ------------------")
# print("----------------------------------------------------")
# print("")

# Load DataFrame
# df = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\[ISI] Health & Technology Journal (2022)\CoV19 Dataset\Rebalanced Data\CoV19_PhS_ClS_Data_Rebalanced.csv")
df = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\[ISI] Health & Technology Journal (2022)\CoV19 Dataset\Balanced Data\CoV19_PhS_ClS_Data_Balanced.csv")

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1].values
t = df.iloc[:, -1].values                                              # [0 = 'Malignant', 1= 'Benign']

# print("------------------------------------------------------")
# print("-------------- Tune-up Seed for ML Models ------------")
# print("------------------------------------------------------")
# print("")

# Set a Random State value
RANDOM_STATE = 42

# Set Python random a fixed value
random.seed(RANDOM_STATE)

# Set environment a fixed value
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)

# Set numpy random a fixed value
np.random.seed(RANDOM_STATE)

print("------------------------------------------------------")
print("------------------- Data Splitting -------------------")
print("------------------------------------------------------")
print("")

# Split Train and Test Data in Proportion of 70:30 %
f_train, f_test, t_train, t_test = train_test_split(f, t, test_size = 0.33, random_state = RANDOM_STATE)

print('Feature Train Set:', f_train.shape)
print('Feature Test Set:', f_test.shape)
print('Target Train Set:', t_train.shape)
print('Target Test Set:', t_test.shape)
print("")

print("------------------------------------------------------")
print("-------------- Ensemble Models Building --------------")
print("------------------------------------------------------")
print("")

print("RFC = RandomForestClassifier")
print("GBC = GradientBoostingClassifier")
print("XGB = XGBClassifier")
print("ADB = AdaBoostClassifier")
print("ETC = ExtraTreesClassifier")
print("CBC = CatBoostClassifier")
print("")

# Creating Machine Learning Models
RFC = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = RANDOM_STATE)
GBC = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.5, random_state = RANDOM_STATE)
XGB = XGBClassifier(n_estimators = 100, eval_metric = 'error', objective = 'binary:logistic')
ADB = AdaBoostClassifier(n_estimators = 100, random_state = RANDOM_STATE)
ETC = ExtraTreesClassifier(n_estimators = 100, random_state = RANDOM_STATE)
CBC = CatBoostClassifier(iterations = 10, learning_rate = 0.5, loss_function='MultiClass')           # loss_function='MultiClass' for Multiclass Classification

print("")

# Fitting Machine Learning Models on Train & Test Data and Measuring Time-Taken at the Same Time

### Random Forest Classifier for Training
RFC_train_time_start = time.perf_counter()
RFC.fit(f_train, t_train)
RFC_train_time_end = time.perf_counter()

### Gradient Boosting Classifier for Training
GBC_train_time_start = time.perf_counter()
GBC.fit(f_train, t_train)
GBC_train_time_end = time.perf_counter()

### XGBoosting Classifier for Training
XGB_train_time_start = time.perf_counter()
XGB.fit(f_train, t_train)
XGB_train_time_end = time.perf_counter()

### ADA Boost Classifier for Training
ADB_train_time_start = time.perf_counter()
ADB.fit(f_train, t_train)
ADB_train_time_end = time.perf_counter()

### ExtraTrees Classifier for Training
ETC_train_time_start = time.perf_counter()
ETC.fit(f_train, t_train)
ETC_train_time_end = time.perf_counter()

### CatBoost Classifier for Training
CBC_train_time_start = time.perf_counter()
CBC.fit(f_train, t_train)
CBC_train_time_end = time.perf_counter()

# Prediction of Test Data by Machine Learning (ML) Models and Measuring Time-Taken at the Same Time

### Random Forest Classifier for Prediction
RFC_pred_time_start = time.perf_counter()
t_pred1 = RFC.predict(f_test)
RFC_pred_time_end = time.perf_counter()

### Gradient Boosting Classifier for Prediction
GBC_pred_time_start = time.perf_counter()
t_pred2 = GBC.predict(f_test)
GBC_pred_time_end = time.perf_counter()

### XGBoosting Classifier for Prediction
XGB_pred_time_start = time.perf_counter()
t_pred3 = XGB.predict(f_test)
XGB_pred_time_end = time.perf_counter()

### ADA Boost Classifier for Prediction
ADB_pred_time_start = time.perf_counter()
t_pred4 = ADB.predict(f_test)
ADB_pred_time_end = time.perf_counter()

### ExtraTrees Classifier for Prediction
ETC_pred_time_start = time.perf_counter()
t_pred5 = ETC.predict(f_test)
ETC_pred_time_end = time.perf_counter()

### CatBoost Classifier for Prediction
CBC_pred_time_start = time.perf_counter()
t_pred6 = CBC.predict(f_test)
CBC_pred_time_end = time.perf_counter()

print("------------------------------------------------------")
print("----------------- Accessed Results -------------------")
print("------------------------------------------------------")
print("")

# Computing ML Training Models Accuracy
print("RFC Train Accuracy:", "{:.3f}".format(RFC.score(f_train, t_train)))
print("GBC Train Accuracy:", "{:.3f}".format(GBC.score(f_train, t_train)))
print("XGB Train Accuracy:", "{:.3f}".format(XGB.score(f_train, t_train)))
print("ADB Train Accuracy:", "{:.3f}".format(ADB.score(f_train, t_train)))
print("ETC Train Accuracy:", "{:.3f}".format(ETC.score(f_train, t_train)))
print("CBC Train Accuracy:", "{:.3f}".format(CBC.score(f_train, t_train)))

print("")

# Computing ML Testing Models Accuracy
print("RFC Test Accuracy:", "{:.3f}".format(RFC.score(f_test, t_test)))
print("GBC Test Accuracy:", "{:.3f}".format(GBC.score(f_test, t_test)))
print("XGB Test Accuracy:", "{:.3f}".format(XGB.score(f_test, t_test)))
print("ADB Test Accuracy:", "{:.3f}".format(ADB.score(f_test, t_test)))
print("ETC Test Accuracy:", "{:.3f}".format(ETC.score(f_test, t_test)))
print("CBC Test Accuracy:", "{:.3f}".format(CBC.score(f_test, t_test)))

print("")

print("RFC Overfitting-Underfitting Value:", "{:.3f}".format(((RFC.score(f_train, t_train)) - (RFC.score(f_test, t_test)))))
print("GBC Overfitting-Underfitting Value:", "{:.3f}".format(((GBC.score(f_train, t_train)) - (GBC.score(f_test, t_test)))))
print("XGB Overfitting-Underfitting Value:", "{:.3f}".format(((XGB.score(f_train, t_train)) - (XGB.score(f_test, t_test)))))
print("ADB Overfitting-Underfitting Value:", "{:.3f}".format(((ADB.score(f_train, t_train)) - (ADB.score(f_test, t_test)))))
print("ETC Overfitting-Underfitting Value:", "{:.3f}".format(((ETC.score(f_train, t_train)) - (ETC.score(f_test, t_test)))))
print("CBC Overfitting-Underfitting Value:", "{:.3f}".format(((CBC.score(f_train, t_train)) - (CBC.score(f_test, t_test)))))

print("")

print("Baseline ML Models Accuracy:")
print("****************************")

# Computing ML Models Accuracy 
print("RFC Baseline Accuracy:", "{:.3f}".format(accuracy_score(t_test, t_pred1)))
print("GBC Baseline Accuracy:", "{:.3f}".format(accuracy_score(t_test, t_pred2)))
print("XGB Baseline Accuracy:", "{:.3f}".format(accuracy_score(t_test, t_pred3)))
print("ADB Baseline Accuracy:", "{:.3f}".format(accuracy_score(t_test, t_pred4)))
print("ETC Baseline Accuracy:", "{:.3f}".format(accuracy_score(t_test, t_pred5)))
print("CBC Baseline Accuracy:", "{:.3f}".format(accuracy_score(t_test, t_pred6)))

print("")

print("K-Fold Cross Validation Average Accuracy:")
print("*****************************************")

# Cross validation for boosting accuracy of ML Models
cross_val_score(RFC, f, t, cv = 10)
cross_val_score(GBC, f, t, cv = 10)
cross_val_score(XGB, f, t, cv = 10)
cross_val_score(ADB, f, t, cv = 10)
cross_val_score(ETC, f, t, cv = 10)

print("")

cross_val_score(CBC, f, t, cv = 10)

print("")

print("RFC K-Fold C.V. Avg. Accuracy:", "{:.3f}".format(np.mean(cross_val_score(RFC, f, t, cv = 10))))
print("GBC K-Fold C.V. Avg. Accuracy:", "{:.3f}".format(np.mean(cross_val_score(GBC, f, t, cv = 10))))
print("XGB K-Fold C.V. Avg. Accuracy:", "{:.3f}".format(np.mean(cross_val_score(XGB, f, t, cv = 10))))
print("ADB K-Fold C.V. Avg. Accuracy:", "{:.3f}".format(np.mean(cross_val_score(ADB, f, t, cv = 10))))
print("ETC K-Fold C.V. Avg. Accuracy:", "{:.3f}".format(np.mean(cross_val_score(ETC, f, t, cv = 10))))

print("")

print("CBC K-Fold C.V. Avg. Accuracy:", "{:.3f}".format(np.mean(cross_val_score(CBC, f, t, cv = 10))))

print("")

print("ML Models ROC-AUC Curve:")
print("************************")

# Prediction of Test Data by ML Models for ROC_AUC_Score
t_pred1_prob = RFC.predict_proba(f_test)
t_pred2_prob = GBC.predict_proba(f_test)
t_pred3_prob = XGB.predict_proba(f_test)
t_pred4_prob = ADB.predict_proba(f_test)
t_pred5_prob = ETC.predict_proba(f_test)
t_pred6_prob = CBC.predict_proba(f_test)

print("RFC ROC-AUC Score:", "{:.3f}".format(roc_auc_score(t_test, t_pred1_prob, multi_class = 'ovo', average = 'weighted')))
print("GBC ROC-AUC Score:", "{:.3f}".format(roc_auc_score(t_test, t_pred2_prob, multi_class = 'ovo', average = 'weighted')))
print("XGB ROC-AUC Score:", "{:.3f}".format(roc_auc_score(t_test, t_pred3_prob, multi_class = 'ovo', average = 'weighted')))
print("ADB ROC-AUC Score:", "{:.3f}".format(roc_auc_score(t_test, t_pred4_prob, multi_class = 'ovo', average = 'weighted')))
print("ETC ROC-AUC Score:", "{:.3f}".format(roc_auc_score(t_test, t_pred5_prob, multi_class = 'ovo', average = 'weighted')))
print("CBC ROC-AUC Score:", "{:.3f}".format(roc_auc_score(t_test, t_pred6_prob, multi_class = 'ovo', average = 'weighted')))

print("")

print("ML Models Classification Report:")
print("********************************")

# Computing Percision, f1-Score, and Recall of ML Models
print ("RFC CR:")
print (classification_report(t_test, t_pred1, zero_division = 0, digits = 3))
print ("GBC CR:")
print (classification_report(t_test, t_pred2, zero_division = 0, digits = 3))
print ("XGB CR:")
print (classification_report(t_test, t_pred3, zero_division = 0, digits = 3))
print ("ADB CR:")
print (classification_report(t_test, t_pred4, zero_division = 0, digits = 3))
print ("ETC CR:")
print (classification_report(t_test, t_pred5, zero_division = 0, digits = 3))
print ("CBC CR:")
print (classification_report(t_test, t_pred6, zero_division = 0, digits = 3))

print("")

print("ML Models Confusion Matrix:")
print("***************************")

# Calculating Confusion Matrix for ML Models
print("RFC CM:")
print(confusion_matrix(t_test, t_pred1))
print("GBC CM:")
print(confusion_matrix(t_test, t_pred2))
print("XGB CM:")
print(confusion_matrix(t_test, t_pred3))
print("ADB CM:")
print(confusion_matrix(t_test, t_pred4))
print("ETC CM:")
print(confusion_matrix(t_test, t_pred5))
print("CBC CM:")
print(confusion_matrix(t_test, t_pred6))

print("")

print("ML Models Time-Taken (in Second):")
print("*********************************")

# Measuring Time-Taken for Training by ML Models
print("RFC TT for Training (in Second):", "{:.3f}".format((RFC_train_time_end - RFC_train_time_start)))
print("GBC TT for Training (in Second):", "{:.3f}".format((GBC_train_time_end - GBC_train_time_start)))
print("XGB TT for Training (in Second):", "{:.3f}".format((XGB_train_time_end - XGB_train_time_start)))
print("ADB TT for Training (in Second):", "{:.3f}".format((ADB_train_time_end - ADB_train_time_start)))
print("ETC TT for Training (in Second):", "{:.3f}".format((ETC_train_time_end - ETC_train_time_start)))
print("CBC TT for Training (in Second):", "{:.3f}".format((CBC_train_time_end - CBC_train_time_start)))

print("")

# Measuring Time-Taken for Prediction by ML Models
print("RFC TT for Prediction (in Second):", "{:.3f}".format((RFC_pred_time_end - RFC_pred_time_start)))
print("GBC TT for Prediction (in Second):", "{:.3f}".format((GBC_pred_time_end - GBC_pred_time_start)))
print("XGB TT for Prediction (in Second):", "{:.3f}".format((XGB_pred_time_end - XGB_pred_time_start)))
print("ADB TT for Prediction (in Second):", "{:.3f}".format((ADB_pred_time_end - ADB_pred_time_start)))
print("ETC TT for Prediction (in Second):", "{:.3f}".format((ETC_pred_time_end - ETC_pred_time_start)))
print("CBC TT for Prediction (in Second):", "{:.3f}".format((CBC_pred_time_end - CBC_pred_time_start)))

print("")

print("------------------------------------------------------")
print("----------------- Plotting Results -------------------")
print("------------------------------------------------------")
print("")

# Plotting Machine Learning Metrics and Performance by Scikit-Plot

### Plotting ROC-AUC Curve 
layout = plt.figure(figsize = (10, 15))
fig = layout.add_subplot(321)
skplt.metrics.plot_roc_curve(t_test, t_pred1_prob, title = "RFC Model ROC-AUC Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(322)
skplt.metrics.plot_roc_curve(t_test, t_pred2_prob, title = "GBC Model ROC-AUC Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(323)
skplt.metrics.plot_roc_curve(t_test, t_pred3_prob, title = "XGB Model ROC-AUC Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(324)
skplt.metrics.plot_roc_curve(t_test, t_pred4_prob, title = "ADB Model ROC-AUC Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(325)
skplt.metrics.plot_roc_curve(t_test, t_pred5_prob, title = "ETC Model ROC-AUC Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(326)
skplt.metrics.plot_roc_curve(t_test, t_pred6_prob, title = "CBC Model ROC-AUC Curve", ax = fig, figsize = (6, 4))

### Plotting Confusion Matrix
layout = plt.figure(figsize = (10, 15))
fig = layout.add_subplot(321)
skplt.metrics.plot_confusion_matrix(t_test, t_pred1, normalize = True, title = "RFC Model Confusion Matrix", cmap = "Greens", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(322)
skplt.metrics.plot_confusion_matrix(t_test, t_pred2, normalize = True, title = "GBC Model Confusion Matrix", cmap = "Purples", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(323)
skplt.metrics.plot_confusion_matrix(t_test, t_pred3, normalize = True, title = "XGB Model Confusion Matrix", cmap = "RdPu", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(324)
skplt.metrics.plot_confusion_matrix(t_test, t_pred4, normalize = True, title = "ADB Model Confusion Matrix", cmap = "OrRd", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(325)
skplt.metrics.plot_confusion_matrix(t_test, t_pred5, normalize = True, title = "ETC Model Confusion Matrix", cmap = "OrRd", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(326)
skplt.metrics.plot_confusion_matrix(t_test, t_pred6, normalize = True, title = "CBC Model Confusion Matrix", cmap = "OrRd", ax = fig, figsize = (6, 4))

### Plotting Learning Curve
layout = plt.figure(figsize = (10, 15))
fig = layout.add_subplot(321)
skplt.estimators.plot_learning_curve(RFC, f, t, cv = 10, title = "RFC Model Learning Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(322)
skplt.estimators.plot_learning_curve(GBC, f, t, cv = 10, title = "GBC Model Learning Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(323)
skplt.estimators.plot_learning_curve(XGB, f, t, cv = 10, title = "XGB Model Learning Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(324)
skplt.estimators.plot_learning_curve(ADB, f, t, cv = 10, title = "ADB Model Learning Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(325)
skplt.estimators.plot_learning_curve(ETC, f, t, cv = 10, title = "ETC Model Learning Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(326)
skplt.estimators.plot_learning_curve(CBC, f, t, cv = 10, title = "CBC Model Learning Curve", ax = fig, figsize = (6, 4))


print("------------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ----------")
print("---------- Signature: Mohammad Reza Saraei -----------")
print("------------------------------------------------------")

