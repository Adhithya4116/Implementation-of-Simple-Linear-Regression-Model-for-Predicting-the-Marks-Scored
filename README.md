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
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Adhithya Perumal.D
RegisterNumber:  212222230007
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
X = np.array(eval(input()))
Y = np.array(eval(input()))

plt.scatter(X,Y)
plt.show()
```
```
import numpy as np
import matplotlib.pyplot as plt
X=np.array(eval(input()))
Y=np.array(eval(input()))
Xmean=np.mean(X)
Ymean=np.mean(Y)
num,den=0,0 # num = numerator, den = denomenator
for i in range(len(X)):
  num+=(X[i]-Xmean)*(Y[i]-Ymean)
  den+=(X[i]-Xmean)**2
m=num/den
c=Ymean-m*Xmean
print(m,c)
Y_pred=m*X+c
print(Y_pred)
plt.scatter(X,Y)
plt.plot(X,Y_pred,color="red")
```
## Output:

![Screenshot (42)](https://github.com/Adhithya4116/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707079/06e644d1-6677-4b22-a51f-b0f4d3cba445)

![Screenshot (43)](https://github.com/Adhithya4116/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707079/2616a0eb-a1fb-4256-a09b-9115bfa117b6)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
