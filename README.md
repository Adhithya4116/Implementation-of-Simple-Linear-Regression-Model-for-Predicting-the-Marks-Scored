# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Upload the file to your cell.

2.Type the required program. 

3.Print the program.

4.End the program.

## Program:
```
Developed by: Adhithya Perumal.D
RegisterNumber:  212222230007

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()

#segregating data to variables
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data

plt.scatter(x_train,y_train,color="black") 
plt.plot(x_train,regressor.predict(x_train),color="red") 
plt.title("Hours VS scores (learning set)") 
plt.xlabel("Hours") 
plt.ylabel("Scores") 
plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color="cyan")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

import numpy as np
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```
## Output:

df.head():

![263445760-4a10342a-4fcc-47db-a298-4d87d6485991](https://github.com/Adhithya4116/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707079/e66a3d9b-e4cb-4916-9abb-0eb48b0af07a)

df.tail():

![263445796-5b1b966e-600d-4aec-821c-0df2d9bbc311](https://github.com/Adhithya4116/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707079/89e11f29-7182-45ea-96a7-cd89d0b585a6)

Array value of X:

![263445955-bfc57abc-2843-49c2-a296-0ea9c2a26bba](https://github.com/Adhithya4116/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707079/551bf75d-5453-48a6-a1c4-d41234d16c2b)

Array value of Y:

![263445972-aadb93a4-2245-4963-9b6a-a1e83a4feaea](https://github.com/Adhithya4116/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707079/28c096f9-a216-43be-9d6a-94edba26ac8b)

Values of Y prediction:

![263445996-f5e5cf9a-c40c-40c7-bff6-21e7ac15f965](https://github.com/Adhithya4116/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707079/b87b3ea4-0dcb-45dd-aa8d-515acf8bb608)

Values of Y test:

![263446021-37b5fb12-d11a-48bc-8792-1798f55b3876](https://github.com/Adhithya4116/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707079/b9109d5b-6965-401b-b3e3-77a5bb748437)

Training Set Graph:

![263446040-60125d6d-4c88-4724-9924-a3f66bab0699](https://github.com/Adhithya4116/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707079/123498e5-2131-45a5-8c36-6f70f65a73ba)

Test Set Graph:

![263446057-c3e2fb4c-0f13-47a4-af45-b929f1ca90d3](https://github.com/Adhithya4116/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707079/1de58726-7ffa-43ed-9666-8f4416c71c29)

Values of MSE, MAE and RMSE:

![263446098-513a0073-7dd8-427e-b250-4415a60ea7a1](https://github.com/Adhithya4116/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707079/4e59d4fa-da91-4b7e-b172-9220fd9fefa7)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
