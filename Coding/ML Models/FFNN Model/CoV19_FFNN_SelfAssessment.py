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
from sklearn.neural_network import MLPClassifier
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
# print("-------------- Tune-up Seed for ANN Models -----------")
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
print("----------------- ANN Models Building -----------------")
print("------------------------------------------------------")
print("")

print("MLP = MLPClassifier")
print("")

# Creating Artificial Neural Network Models
MLP = MLPClassifier(max_iter = 500, solver = 'lbfgs', random_state = RANDOM_STATE)

# Fitting Artificial Neural Network Models on Train & Test Data and Measuring Time-Taken at the Same Time

### MLP Classifier for Training
MLP_train_time_start = time.perf_counter()
MLP.fit(f_train, t_train)
MLP_train_time_end = time.perf_counter()

# Prediction of Test Data by Artificial Neural Network (ANN) Models and Measuring Time-Taken at the Same Time

### MLP Classifier for Prediction
MLP_pred_time_start = time.perf_counter()
t_pred = MLP.predict(f_test)
MLP_pred_time_end = time.perf_counter()

print("------------------------------------------------------")
print("----------------- Accessed Results -------------------")
print("------------------------------------------------------")
print("")

# Computing ANN Training Models Accuracy
print("MLP Train Accuracy:", "{:.3f}".format(MLP.score(f_train, t_train)))
print("")

# Computing ANN Testing Models Accuracy
print("MLP Test Accuracy:", "{:.3f}".format(MLP.score(f_test, t_test)))
print("")

print("KNN Overfitting-Underfitting Value:", "{:.3f}".format(((MLP.score(f_train, t_train)) - (MLP.score(f_test, t_test)))))
print("")

print("Baseline ANN Models Accuracy:")
print("****************************")

# Computing ANN Models Accuracy 
print("MLP Baseline Accuracy:", "{:.3f}".format(accuracy_score(t_test, t_pred)))
print("")

print("K-Fold Cross Validation Average Accuracy:")
print("*****************************************")

# Cross validation for boosting accuracy of ANN Models
cross_val_score(MLP, f, t, cv = 10)

print("MLP K-Fold C.V. Avg. Accuracy:", "{:.3f}".format(np.mean(cross_val_score(MLP, f, t, cv = 10))))
print("")

print("ANN Models ROC-AUC Curve:")
print("************************")

# Prediction of Test Data by ANN Models for ROC_AUC_Score
t_pred_prob = MLP.predict_proba(f_test)

print("MLP ROC-AUC Score:", "{:.3f}".format(roc_auc_score(t_test, t_pred_prob, multi_class = 'ovo', average = 'weighted')))
print("")

print("ANN Models Classification Report:")
print("********************************")

# Computing Percision, f1-Score, and Recall of ANN Models
print ("MLP CR:")
print (classification_report(t_test, t_pred, zero_division = 0, digits = 3))
print("")

print("ANN Models Confusion Matrix:")
print("***************************")

# Calculating Confusion Matrix for ANN Models
print("MLP CM:")
print(confusion_matrix(t_test, t_pred))
print("")

print("ANN Models Time-Taken (in Second):")
print("*********************************")

# Measuring Time-Taken for Training by ANN Models
print("MLP TT for Training:", "{:.3f}".format((MLP_train_time_end - MLP_train_time_start)))
print("")

# Measuring Time-Taken for Prediction by ANN Models
print("MLP TT for Prediction:", "{:.3f}".format((MLP_pred_time_end - MLP_pred_time_start)))
print("")

print("------------------------------------------------------")
print("----------------- Plotting Results -------------------")
print("------------------------------------------------------")
print("")

# Plotting Artificial Neural Network Metrics and Performance by Scikit-Plot

### Plotting ROC-AUC Curve 
layout = plt.figure(figsize = (6, 4))
fig = layout.add_subplot(111)
skplt.metrics.plot_roc_curve(t_test, t_pred_prob, title = "MLP Model ROC-AUC Curve", ax = fig, figsize = (6, 4))

### Plotting Confusion Matrix
layout = plt.figure(figsize = (6, 4))
fig = layout.add_subplot(111)
skplt.metrics.plot_confusion_matrix(t_test, t_pred, normalize = True, title = "MLP Model Confusion Matrix", cmap = "Oranges", ax = fig, figsize = (6, 4))

### Plotting Learning Curve
layout = plt.figure(figsize = (6, 4))
fig = layout.add_subplot(111)
skplt.estimators.plot_learning_curve(MLP, f, t, cv = 10, title = "MLP Model Learning Curve", ax = fig, figsize = (6, 4))

print("------------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ----------")
print("---------- Signature: Mohammad Reza Saraei -----------")
print("------------------------------------------------------")

