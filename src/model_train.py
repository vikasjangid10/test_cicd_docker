import numpy as np
import pandas as pd

df = pd.read_csv("data/salary.csv")
print(df.head())

print(df.isnull().sum())

from sklearn.model_selection import train_test_split
x = df.drop(columns=['Salary'])
y=df['YearsExperience']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import r2_score
score = r2_score(y_pred,y_test)
print(score)

import joblib

# Save the model
joblib.dump(lr, 'salary_model.pkl')
