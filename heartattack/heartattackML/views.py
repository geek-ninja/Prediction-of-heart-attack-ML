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