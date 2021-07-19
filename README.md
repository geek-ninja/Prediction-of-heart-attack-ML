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

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
import joblib



dataset = pd.read_csv('/kaggle/input/heart-attack-analysis-prediction-dataset/heart.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 102)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

print('accuracy is ', classifier.score(x_test,y_test)*100,'%')

#saving the model
filename = 'ml_model.sav'
joblib.dump(classifier,filename)
```
<p>joblib module of python is used to create an instance of the model so that we can use it in django as a .sav file</p>

<p>django (views.py)</p>

```python

from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
import joblib
from sklearn import preprocessing
import pandas as pd
import numpy as np
# Create your views here.
def input_view(request):
    return render(request,"ml/input.html")

def res_view(request):
    clss = joblib.load('model/ml_model.sav')
    lis = []
    lis.append(request.GET['age'])
    lis.append(request.GET['sex'])
    lis.append(request.GET['cp'])
    lis.append(request.GET['trtbps'])
    lis.append(request.GET['chol'])
    lis.append(request.GET['fbs'])
    lis.append(request.GET['restecg'])
    lis.append(request.GET['thalachh'])
    lis.append(request.GET['exng'])
    lis.append(request.GET['oldpeak'])
    lis.append(request.GET['slp'])
    lis.append(request.GET['caa'])
    lis.append(request.GET['thall'])
    
    lis = list(map(float,lis))
    lis = np.resize(lis,(1,13))
    
    dataset = pd.read_csv('model/heart.csv')
    df = pd.DataFrame(dataset)
    x = df.iloc[:,:-1].values
    x = np.append(x,lis,axis=0)
    
    sc_x = preprocessing.StandardScaler()
    sc_input = sc_x.fit_transform(x)
    sc_input = sc_input[-1]
   
    ans = clss.predict([sc_input])
    
    result = ['Less Chance of heart attack','More chance of heart attack']
    i = ans[0]
    res = result[i]
    return render(request,"ml/output.html",{'pred': res,'i':i})
```
<h3>res_view() function in views.py gets the input from the website(user) and predict the output and pass the result as a context variable </h3>

<p>django (urls.py)</p>

```python
from django.contrib import admin
from django.urls import path
from heartattackML.views import input_view,res_view
urlpatterns = [
    path('admin/', admin.site.urls),
    path('input/',input_view,name = 'input'),
    path('output/',res_view,name = 'output')
]
```
<p>These paths in urls.py will connect different pages of your website</p>

<p>django (settings.py)</p>

```python
import os

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'heartattackML',
]

STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "static"),
]

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR,"templates")],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

MEDIA_ROOT = os.path.join(BASE_DIR,'model')
MEDIA_URL = '/model/'
```
<p>Add these additional segments in settings.py of django will link your static folders and model model folder where you kept all your css & html files , sav files , csv files etc. It make your template folder global </p> 
