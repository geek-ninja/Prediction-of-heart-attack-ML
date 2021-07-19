# Prediction-of-heart-attack-ML
<h2>Prediction Website made using Django and machine learning</h2><br>
<p>hope you have installed django or else you can follow the command</p><br>
python -m pip install Django
<br>
<h3>Create the project</h3>
django-admin startproject heartattack
<br>
<p>change the directory to the new project folder created</p>
<p>Then create the app for the project</p>
python manage.py startapp heartattackML
<br>
<p>For quick overview of the project you can copy and overwrite the files and folder from the repo to the respective files of django that you have created</p>
<h3>How to run the django server</h3>
python manage.py runserver

<h2>Machine Learning Model</h2>
<p>The model used here is Logistic Regression with an accuracy of 88 %</p>
<p>The dataset taken from the link below</p>
https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

<h2>Integration of model with django</h2>

<h2>Machine Learning model used</h2>
<p>Logistic Regression (model.py)</p>
<iframe src="https://www.kaggle.com/embed/geekninja/heart-attack-classificaion?cellId=2&cellIds=2&kernelSessionId=67487677" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="heart-attack-classificaion"></iframe>
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
import joblib

dataset = pd.read_csv('glass.csv')

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,9]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.ensemble import RandomForestClassifier

clss = RandomForestClassifier(criterion='entropy',n_estimators=300,random_state=42)
clss.fit(x_train,y_train)

print('accuracy is ', clss.score(x_test,y_test)*100,'%')

#saving the model
filename = 'finalized_model.sav'
joblib.dump(clss,filename)
```

  
