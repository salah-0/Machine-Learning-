# -*- coding: utf-8 -*-
"""Binary_C_Bank_Dataset.ipynb


Original file is located at
    https://colab.research.google.com/drive/1lsprgx2q1Zarh3XTzbtbZjZ_wlGG6pNm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

train= pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
sample= pd.read_csv('sample_submission.csv')

sample.head()

train

test

df= train.copy()

df.info()

df.isnull().sum()

df.duplicated().sum()

df.shape

df.describe()

df.hist(figsize=(15,15))

df.drop(columns=['id', 'CustomerId', 'Surname'], inplace=True)
test.drop(columns=['id', 'CustomerId', 'Surname'], inplace=True)

df.head(10)

print(df['Geography'].unique())
print(df['Gender'].unique())

df['Exited'].value_counts()

#df['Exited'].sum()

plt.figure(figsize=(12, 4))
sns.boxplot(data=df.select_dtypes(include='number'))

sns.histplot(data=df, x='Age', hue='Exited', binwidth=2, alpha=0.7, kde=True)
plt.title('Age distribution')

test.info()

encoder= OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value= 10)
df[['Gender', 'Geography']]= encoder.fit_transform(df[['Gender', 'Geography']])
test[['Gender', 'Geography']]= encoder.transform(test[['Gender', 'Geography']])

df.corr()

fig, ax= plt.subplots(figsize=(7, 5))
sns.heatmap(df.corr(), fmt='.2f', annot=True)

x= df.drop('Exited', axis= 1)
y= df['Exited']

X_train, X_test, y_train, y_test= train_test_split(x,y, test_size= 0.2, random_state= 42,
                                                   stratify=y)

scaler= MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)
x_test= scaler.transform(test)

print(y_pred.shape)
print(y_test.shape)

d_t= DecisionTreeClassifier(random_state=42)
d_t.fit(X_train, y_train)
y_pred= d_t.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

l_r = LogisticRegression(max_iter=1000)
l_r= l_r.fit(X_train, y_train)
y_pred= l_r.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

knn= KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred= knn.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

svm= SVC()
svm.fit(X_train, y_train)
y_pred= svm.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

r_f= RandomForestClassifier(random_state=42)
r_f.fit(X_train, y_train)
y_pred= r_f.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

rg= GradientBoostingClassifier()
rg.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def evaluate_classifiers(X_train, y_train, X_test, y_test):
    # Model isimlerini ve nesnelerini bir listeye koyun
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    # Sonuçları saklamak için bir liste oluşturun
    results = []

    # Her bir sınıflandırıcıyı çalıştırın
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Confusion Matrix": confusion
        })

    return pd.DataFrame(results)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

results = evaluate_classifiers(X_train, y_train, X_test, y_test)
print(results)

rg= GradientBoostingClassifier()
rg.fit(X_train, y_train)
y_pred= rg.predict(x_test)

print(y_pred[:851])

sample.drop('Exited', axis= 1, inplace=True)
sample.head()

sample['Exited']= y_pred

