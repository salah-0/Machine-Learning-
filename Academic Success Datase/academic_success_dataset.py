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