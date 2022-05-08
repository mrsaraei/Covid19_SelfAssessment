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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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
t = df.iloc[:, -1].values                                        # [0 = 'Unsuspecting/Normal', 1 = 'Suspected', 2 = 'Probable']

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
print("----------------- ML Models Building -----------------")
print("------------------------------------------------------")
print("")

print("KNN = KNeighborsClassifier")
print("DTC = DecisionTreeClassifier")
print("GNB = GaussianNBClassifier")
print("SVM = SupportVectorMachineClassifier")
print("LRG = LogisticRegressionClassifier")
print("")

# Creating Machine Learning Models
KNN = KNeighborsClassifier(n_neighbors = 6, p = 2)
DTC = DecisionTreeClassifier(random_state = RANDOM_STATE)
GNB = GaussianNB()
SVM = SVC(decision_function_shape = "ovo", probability = True, random_state = RANDOM_STATE)
LRG = LogisticRegression(multi_class ='multinomial', solver ='lbfgs', random_state = RANDOM_STATE)

# Fitting Machine Learning Models on Train & Test Data and Measuring Time-Taken at the Same Time

### KNeighbors Classifier for Training
KNN_train_time_start = time.perf_counter()
KNN.fit(f_train, t_train)
KNN_train_time_end = time.perf_counter()

### Decision Tree Classifier for Training
DTC_train_time_start = time.perf_counter()
DTC.fit(f_train, t_train)
DTC_train_time_end = time.perf_counter()

### Gaussian Native Bayes Classifier for Training
GNB_train_time_start = time.perf_counter()
GNB.fit(f_train, t_train)
GNB_train_time_end = time.perf_counter()

### Support Vector Machine Classifier for Training
SVM_train_time_start = time.perf_counter()
SVM.fit(f_train, t_train)
SVM_train_time_end = time.perf_counter()

### Logistic Regression Classifier for Training
LRG_train_time_start = time.perf_counter()
LRG.fit(f_train, t_train)
LRG_train_time_end = time.perf_counter()

# Prediction of Test Data by Machine Learning (ML) Models and Measuring Time-Taken at the Same Time

### KNeighbors Classifier for Prediction
KNN_pred_time_start = time.perf_counter()
t_pred0 = KNN.predict(f_test)
KNN_pred_time_end = time.perf_counter()

### Decision Tree Classifier for Prediction
DTC_pred_time_start = time.perf_counter()
t_pred1 = DTC.predict(f_test)
DTC_pred_time_end = time.perf_counter()

### Gaussian Native Bayes Classifier for Prediction
GNB_pred_time_start = time.perf_counter()
t_pred2 = GNB.predict(f_test)
GNB_pred_time_end = time.perf_counter()

### Support Vector Machine Classifier for Prediction
SVM_pred_time_start = time.perf_counter()
t_pred3 = SVM.predict(f_test)
SVM_pred_time_end = time.perf_counter()

### Logistic Regression Classifier for Prediction
LRG_pred_time_start = time.perf_counter()
t_pred4 = LRG.predict(f_test)
LRG_pred_time_end = time.perf_counter()

print("------------------------------------------------------")
print("----------------- Accessed Results -------------------")
print("------------------------------------------------------")
print("")

# Computing ML Training Models Accuracy
print("KNN Train Accuracy:", "{:.3f}".format(KNN.score(f_train, t_train)))
print("DTC Train Accuracy:", "{:.3f}".format(DTC.score(f_train, t_train)))
print("GNB Train Accuracy:", "{:.3f}".format(GNB.score(f_train, t_train)))
print("SVM Train Accuracy:", "{:.3f}".format(SVM.score(f_train, t_train)))
print("LRG Train Accuracy:", "{:.3f}".format(LRG.score(f_train, t_train)))
print("")

# Computing ML Testing Models Accuracy
print("KNN Test Accuracy:", "{:.3f}".format(KNN.score(f_test, t_test)))
print("DTC Test Accuracy:", "{:.3f}".format(DTC.score(f_test, t_test)))
print("GNB Test Accuracy:", "{:.3f}".format(GNB.score(f_test, t_test)))
print("SVM Test Accuracy:", "{:.3f}".format(SVM.score(f_test, t_test)))
print("LRG Test Accuracy:", "{:.3f}".format(LRG.score(f_test, t_test)))
print("")

print("KNN Overfitting-Underfitting Value:", "{:.3f}".format(((KNN.score(f_train, t_train)) - (KNN.score(f_test, t_test)))))
print("DTC Overfitting-Underfitting Value:", "{:.3f}".format(((DTC.score(f_train, t_train)) - (DTC.score(f_test, t_test)))))
print("GNB Overfitting-Underfitting Value:", "{:.3f}".format(((GNB.score(f_train, t_train)) - (GNB.score(f_test, t_test)))))
print("SVM Overfitting-Underfitting Value:", "{:.3f}".format(((SVM.score(f_train, t_train)) - (SVM.score(f_test, t_test)))))
print("LRG Overfitting-Underfitting Value:", "{:.3f}".format(((LRG.score(f_train, t_train)) - (LRG.score(f_test, t_test)))))
print("")

print("Baseline ML Models Accuracy:")
print("****************************")

# Computing ML Models Accuracy 
print("KNN Baseline Accuracy:", "{:.3f}".format(accuracy_score(t_test, t_pred0)))
print("DTC Baseline Accuracy:", "{:.3f}".format(accuracy_score(t_test, t_pred1)))
print("GNB Baseline Accuracy:", "{:.3f}".format(accuracy_score(t_test, t_pred2)))
print("SVM Baseline Accuracy:", "{:.3f}".format(accuracy_score(t_test, t_pred3)))
print("LRG Baseline Accuracy:", "{:.3f}".format(accuracy_score(t_test, t_pred4)))
print("")

print("K-Fold Cross Validation Average Accuracy:")
print("*****************************************")

# Cross validation for boosting accuracy of ML Models
cross_val_score(KNN, f, t, cv = 10)
cross_val_score(DTC, f, t, cv = 10)
cross_val_score(GNB, f, t, cv = 10)
cross_val_score(SVM, f, t, cv = 10)
cross_val_score(LRG, f, t, cv = 10)

print("KNN K-Fold C.V. Avg. Accuracy:", "{:.3f}".format(np.mean(cross_val_score(KNN, f, t, cv = 10))))
print("DTC K-Fold C.V. Avg. Accuracy:", "{:.3f}".format(np.mean(cross_val_score(DTC, f, t, cv = 10))))
print("GNB K-Fold C.V. Avg. Accuracy:", "{:.3f}".format(np.mean(cross_val_score(GNB, f, t, cv = 10))))
print("SVM K-Fold C.V. Avg. Accuracy:", "{:.3f}".format(np.mean(cross_val_score(SVM, f, t, cv = 10))))
print("LRG K-Fold C.V. Avg. Accuracy:", "{:.3f}".format(np.mean(cross_val_score(LRG, f, t, cv = 10))))
print("")

print("ML Models ROC-AUC Curve:")
print("************************")

# Prediction of Test Data by ML Models for ROC_AUC_Score
t_pred0_prob = KNN.predict_proba(f_test)
t_pred1_prob = DTC.predict_proba(f_test)
t_pred2_prob = GNB.predict_proba(f_test)
t_pred3_prob = SVM.predict_proba(f_test)
t_pred4_prob = LRG.predict_proba(f_test)

print("KNN ROC-AUC Score:", "{:.3f}".format(roc_auc_score(t_test, t_pred0_prob, multi_class = 'ovo', average = 'weighted')))
print("DTC ROC-AUC Score:", "{:.3f}".format(roc_auc_score(t_test, t_pred1_prob, multi_class = 'ovo', average = 'weighted')))
print("GNB ROC-AUC Score:", "{:.3f}".format(roc_auc_score(t_test, t_pred2_prob, multi_class = 'ovo', average = 'weighted')))
print("SVM ROC-AUC Score:", "{:.3f}".format(roc_auc_score(t_test, t_pred3_prob, multi_class = 'ovo', average = 'weighted')))
print("LRG ROC-AUC Score:", "{:.3f}".format(roc_auc_score(t_test, t_pred4_prob, multi_class = 'ovo', average = 'weighted')))
print("")

print("ML Models Classification Report:")
print("********************************")

# Computing Percision, f1-Score, and Recall of ML Models
print ("KNN CR:")
print (classification_report(t_test, t_pred0, zero_division = 0, digits = 3))
print ("DTC CR:")
print (classification_report(t_test, t_pred1, zero_division = 0, digits = 3))
print ("GNB CR:")
print (classification_report(t_test, t_pred2, zero_division = 0, digits = 3))
print ("SVM CR:")
print (classification_report(t_test, t_pred3, zero_division = 0, digits = 3))
print ("LRG CR:")
print (classification_report(t_test, t_pred4, zero_division = 0, digits = 3))
print("")

print("ML Models Confusion Matrix:")
print("***************************")

# Calculating Confusion Matrix for ML Models
print("KNN CM:")
print(confusion_matrix(t_test, t_pred0))
print("DTC CM:")
print(confusion_matrix(t_test, t_pred1))
print("GNB CM:")
print(confusion_matrix(t_test, t_pred2))
print("SVM CM:")
print(confusion_matrix(t_test, t_pred3))
print("LRG CM:")
print(confusion_matrix(t_test, t_pred4))
print("")

print("ML Models Time-Taken (in Second):")
print("*********************************")

# Measuring Time-Taken for Training by ML Models
print("KNN TT for Training:", "{:.3f}".format((KNN_train_time_end - KNN_train_time_start)))
print("DTC TT for Training:", "{:.3f}".format((DTC_train_time_end - DTC_train_time_start)))
print("GNB TT for Training:", "{:.3f}".format((GNB_train_time_end - GNB_train_time_start)))
print("SVM TT for Training:", "{:.3f}".format((SVM_train_time_end - SVM_train_time_start)))
print("LRG TT for Training:", "{:.3f}".format((LRG_train_time_end - LRG_train_time_start)))
print("")

# Measuring Time-Taken for Prediction by ML Models
print("KNN TT for Prediction:", "{:.3f}".format((KNN_pred_time_end - KNN_pred_time_start)))
print("DTC TT for Prediction:", "{:.3f}".format((DTC_pred_time_end - DTC_pred_time_start)))
print("GNB TT for Prediction:", "{:.3f}".format((GNB_pred_time_end - GNB_pred_time_start)))
print("SVM TT for Prediction:", "{:.3f}".format((SVM_pred_time_end - SVM_pred_time_start)))
print("LRG TT for Prediction:", "{:.3f}".format((LRG_pred_time_end - LRG_pred_time_start)))
print("")

print("------------------------------------------------------")
print("----------------- Plotting Results -------------------")
print("------------------------------------------------------")
print("")

# Plotting Machine Learning Metrics and Performance by Scikit-Plot

### Plotting ROC-AUC Curve 
layout = plt.figure(figsize = (10, 15))
fig = layout.add_subplot(321)
skplt.metrics.plot_roc_curve(t_test, t_pred0_prob, title = "KNN Model ROC-AUC Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(322)
skplt.metrics.plot_roc_curve(t_test, t_pred1_prob, title = "DTC Model ROC-AUC Curve", ax = fig, figsize = (4, 6))
fig = layout.add_subplot(323)
skplt.metrics.plot_roc_curve(t_test, t_pred2_prob, title = "GNB Model ROC-AUC Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(324)
skplt.metrics.plot_roc_curve(t_test, t_pred3_prob, title = "SVM Model ROC-AUC Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(325)
skplt.metrics.plot_roc_curve(t_test, t_pred4_prob, title = "LRG Model ROC-AUC Curve", ax = fig, figsize = (6, 4))

### Plotting Confusion Matrix
layout = plt.figure(figsize = (10, 15))
fig = layout.add_subplot(321)
skplt.metrics.plot_confusion_matrix(t_test, t_pred0, normalize = True, title = "KNN Model Confusion Matrix", cmap = "Oranges", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(322)
skplt.metrics.plot_confusion_matrix(t_test, t_pred1, normalize = True, title = "DTC Model Confusion Matrix", cmap = "Reds", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(323)
skplt.metrics.plot_confusion_matrix(t_test, t_pred2, normalize = True, title = "GNB Model Confusion Matrix", cmap = "Blues", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(324)
skplt.metrics.plot_confusion_matrix(t_test, t_pred3, normalize = True, title = "SVM Model Confusion Matrix", cmap = "Greys", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(325)
skplt.metrics.plot_confusion_matrix(t_test, t_pred4, normalize = True, title = "LRG Model Confusion Matrix", cmap = "YlOrBr", ax = fig, figsize = (6, 4))

### Plotting Learning Curve
layout = plt.figure(figsize = (10, 15))
fig = layout.add_subplot(321)
skplt.estimators.plot_learning_curve(KNN, f, t, cv = 10, title = "KNN Model Learning Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(322)
skplt.estimators.plot_learning_curve(DTC, f, t, cv = 10, title = "DTC Model Learning Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(323)
skplt.estimators.plot_learning_curve(GNB, f, t, cv = 10, title = "GNB Model Learning Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(324)
skplt.estimators.plot_learning_curve(SVM, f, t, cv = 10, title = "SVM Model Learning Curve", ax = fig, figsize = (6, 4))
fig = layout.add_subplot(325)
skplt.estimators.plot_learning_curve(LRG, f, t, cv = 10, title = "LRG Model Learning Curve", ax = fig, figsize = (6, 4))


print("------------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ----------")
print("---------- Signature: Mohammad Reza Saraei -----------")
print("------------------------------------------------------")

