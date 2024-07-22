# -*- coding: utf-8 -*-
"""Academic_Success_Dataset.ipynb

Original file is located at
    https://colab.research.google.com/drive/1Bs2eyMoPpwdLjEyVVkxToQs5AefhcDc7
"""

#!pip install catboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
#from sklearn.ensemble import AdaBoostClassifier, RandomFroestClassifier, GradientBosstingClassifier
from xgboost import XGBClassifier
from xgboost import XGBClassifier
#from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

train= pd.read_csv('train.csv', index_col='id')
test= pd.read_csv('test.csv', index_col='id')
train

train.isna().sum()

print(train.duplicated().sum())
print(train.info())

train.describe().T

unique_counts = {}
for col in train.columns:
    unique_counts[col] = train[col].nunique()

col_unique = pd.DataFrame(list(unique_counts.items()), columns=['Column', 'Unique Count'])
col_unique.set_index('Column', inplace=True)
print(col_unique)

cat_cols= [col for col in train.columns if train[col].nunique()<= 8]
num_cols= [col for col in train.columns if train[col].nunique()>= 9]

print(len(cat_cols), len(num_cols))

plt.figure(figsize=(9, 7))
ax= sns.countplot(x= 'Target', data= train, palette= 'flare_r')
for p in ax.patches:
  ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()*1.02))

plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target Distribution')
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 20))
plotnumber = 1

for col in cat_cols:
    if plotnumber <= len(cat_cols):
        ax = plt.subplot(6, 2, plotnumber)
        sns.countplot(x=train[col], data=train, palette='pastel')

        # Add labels to each bar in the plot
        for p in ax.patches:
          ax.annotate(format(p.get_height(),'.0f'), (p.get_x(), p.get_height()*1.006))

    plotnumber += 1

plt.suptitle('Distribution of Categorical Variables', fontsize=20, y=1)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 20))
plotnumber = 1

for col in cat_cols:
    if plotnumber <= len(cat_cols):
        ax = plt.subplot(6, 2, plotnumber)
        sns.countplot(x=train[col], data=train, hue= train['Target'], palette='bright')

    plotnumber += 1

plt.suptitle('Distribution of Categorical Variables', fontsize=20, y=1)
plt.tight_layout()
plt.show()

catagories= ['dropedout', 'enrolled', 'graduated']
label_encoder= LabelEncoder()
train['Target']= label_encoder.fit_transform(train['Target'])
train['Target']

plt.figure(figsize=(16, 14))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt='.1f', linewidths=2, linecolor='gray')
plt.suptitle('Correlation Matrix', fontsize=20, fontweight='bold', y=1)
plt.show()

# Split the features and target variable
X_train = train[initial_features]
y_train = train['Target']
X_test = test[initial_features]

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def cross_validate_model(model, X_train, y_train, params, n_splits=10):
    """
    Performs K-Fold cross-validation for a given model, returns the last model and average validation accuracy.

    Parameters:
        model: Machine learning model class (e.g., RandomForestClassifier)
        X_train: Training feature dataset
        y_train: Training target dataset
        params: Dictionary of parameters to initialize the model
        n_splits: Number of folds for cross-validation (default: 10)

    Returns:
        last_model: The last trained model instance
        average_val_accuracy: Average validation accuracy over all folds
    """
    # Initialize variables
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    val_scores = []

    # Cross-validation loop
    for fold, (train_ind, valid_ind) in enumerate(cv.split(X_train)):
        # Data splitting
        X_fold_train = X_train.iloc[train_ind]
        y_fold_train = y_train.iloc[train_ind]
        X_val = X_train.iloc[valid_ind]
        y_val = y_train.iloc[valid_ind]
        
        # Model initialization and training
        clf = model(**params)
        clf.fit(X_fold_train, y_fold_train)
        
        # Predict and evaluate
        y_pred_trn = clf.predict(X_fold_train)
        y_pred_val = clf.predict(X_val)
        train_acc = accuracy_score(y_fold_train, y_pred_trn)
        val_acc = accuracy_score(y_val, y_pred_val)
        print(f"Fold: {fold}, Train Accuracy: {train_acc:.5f}, Val Accuracy: {val_acc:.5f}")
        print("-" * 50)
        
        # Accumulate validation scores
        val_scores.append(val_acc)

    # Calculate the average validation score
    average_val_accuracy = np.mean(val_scores)
    print("Average Validation Accuracy:", average_val_accuracy)

    return clf, average_val_accuracy

print('Random Forest Cross-Validation Results:\n')
rf_model, rf_mean_accuracy = cross_validate_model(RandomForestClassifier, X_train, y_train, params={} verbose=0

# Predict the test set and reverse the label encoding
rf_preds = rf_model.predict(X_test)
rf_preds_labels = label_encoder.inverse_transform(rf_preds)

# Save the predictions to a CSV file
rf_result = pd.DataFrame(X_test.index)
rf_result['Target'] = rf_preds_labels
rf_result.to_csv('result_rf.csv', index=False)
rf_result

print('AdaBoost Cross-Validation Results:\n')
ada_model, ada_mean_accuracy = cross_validate_model(AdaBoostClassifier, X_train, y_train, params={})

# Predict the test set and reverse the label encoding
ada_preds = ada_model.predict(X_test)
ada_preds_labels = label_encoder.inverse_transform(ada_preds)

# Save the predictions to a CSV file
ada_result = pd.DataFrame(X_test.index)
ada_result['Target'] = ada_preds_labels
ada_result.to_csv('result_ada.csv', index=False)
ada_result

print('Gradient Boosting Cross-Validation Results:\n')
gb_model, gb_mean_accuracy = cross_validate_model(GradientBoostingClassifier, X_train, y_train, params={})

# Predict the test set and reverse the label encoding
gb_preds = gb_model.predict(X_test)
gb_preds_labels = label_encoder.inverse_transform(gb_preds)

# Save the predictions to a CSV file
gb_result = pd.DataFrame(X_test.index)
gb_result['Target'] = gb_preds_labels
gb_result.to_csv('result_gb.csv', index=False)
gb_result

print('XGBoost Cross-Validation Results:\n')
xgb_model, xgb_mean_accuracy = cross_validate_model(XGBClassifier, X_train, y_train, params={})

# Predict the test set and reverse the label encoding
xgb_preds = xgb_model.predict(X_test)
xgb_preds_labels = label_encoder.inverse_transform(xgb_preds)

# Save the predictions to a CSV file
xgb_result = pd.DataFrame(X_test.index)
xgb_result['Target'] = xgb_preds_labels
xgb_result.to_csv('result_xgb.csv', index=False)
xgb_result
