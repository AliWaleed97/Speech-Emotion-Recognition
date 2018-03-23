import pandas as pd
import numpy as np
import plotly
import os
import time
import shutil
import plotly
import plotly.plotly as py
from django.conf import settings
import speech_recognition as sr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import model_selection
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis.audioFeatureExtraction import stFeatureExtraction
from pydub import AudioSegment
from pydub.utils import make_chunks
from plotly.graph_objs import *
from IPython.html.widgets import interact
from plotly.tools import FigureFactory as FF 
from plotly.offline import init_notebook_mode, iplot
from IPython.display import display, HTML
from django.shortcuts import render,HttpResponse,redirect
import soundfile as sf

D= pd.read_csv('FinalData.csv')
D.drop(['Unnamed: 0'],axis=1,inplace=True)
X = D.drop(['Emotion'],axis=1)
Y = D['Emotion']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=101)
LDA = LinearDiscriminantAnalysis()
LDA.fit(X_train,Y_train)
predictions = LDA.predict(X_test)
print classification_report(Y_test,predictions)


# Create your views here.
def index (request):
	return render(request,'index.html')
def details(request):
    return render(request,'details.html')
def liveRecord(request):
    return render(request,'liverecord.html')

def emptydir(top):
    if(top == '/' or top == "\\"): return
    else:
        for root, dirs, files in os.walk(top, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

def livepredict(request):
    path = settings.FILES_DIR
    os.chdir(path)
    files = sorted(os.listdir(os.getcwd()), key=os.path.getmtime ,reverse=True)
    file = os.path.expanduser('~/Downloads/'+files[0])
    return predictEmotion(request, file)


def predictEmotion(request,string):
    s = sf.SoundFile(string)
    duration = len(s) / s.samplerate
    if duration < 10:
        [Fs,x] = audioBasicIO.readAudioFile(string)
        x = audioBasicIO.stereo2mono(x)
        F = stFeatureExtraction(x,Fs,Fs*0.050,Fs*0.025)
        F = np.mean(F,axis=1)
        F = np.reshape(F,(1,34))
        predict1 = LDA.predict(F)
        predict = LDA.predict_proba(F)
        Map = {'Fear':predict[0][0],'disgust':predict[0][1],'Happiness':predict[0][2],'Boredom':predict[0][3],
        'Neutral':predict[0][4],'sadness':predict[0][5],'Anger':predict[0][6]}
        data = [Bar(x=Map.keys(),y=Map.values())]
        plotly.offline.plot(data, filename='basic-bar')
        return redirect('/')
    else:
        myaudio = AudioSegment.from_file(string)
        chunk_length_ms = 3000 # pydub calculates in millisec
        chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
        chunkarr = []
        if os.listdir('./test') != []:
            emptydir('./test')
        else:
            print 'Test is Empty'
            
        for i, chunk in enumerate(chunks):
            chunk_name = "chunk{0}.wav".format(i)
            print "exporting", chunk_name
            chunk.export('test/'+chunk_name, format="wav")
            chunkarr.append(chunk_name)

        files = os.listdir('./test')
        files.sort()
        to_predict=[]
        percent = []
        for f in files:
            [Fs,x] = audioBasicIO.readAudioFile("test/"+f)
            x = audioBasicIO.stereo2mono(x)
            F = stFeatureExtraction(x,Fs,800,500)
            F = np.mean(F,axis=1)
            to_predict.append(F)
            predict = LDA.predict_proba(F.reshape(1,-1))
            Map = {'Fear':predict[0][0],'disgust':predict[0][1],'Happiness':predict[0][2],'Boredom':predict[0][3],
            'Neutral':predict[0][4],'sadness':predict[0][5],'Anger':predict[0][6]}
            percent.append(Map)

        table = []

        for i in range(0,len(percent)):
            table.append(percent[i].values())
            columns = ['Anger','Boredom','Fear','Happiness','Neutral','Disgust','Sadness']
            columns.reverse()
            
        init_notebook_mode(connected=True)

        figure = {
            'data': [],
            'layout': {},
            'frames': []
        }
        config = {'scrollzoom': True}

        figure['data'] = [{'x': columns, 'y': table}]
        figure['layout']['xaxis'] = {'autorange': True}
        figure['layout']['yaxis'] =  {'range': [0, 1],'autorange':False}
        figure['layout']['title'] = 'Speech Emotion Recognition'
        figure['layout']['updatemenus']=[
            {
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 500, 'redraw': False},
                                 'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                        'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0.1,
                'yanchor': 'top'
            }
        ]
        sliders_dict = {
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 20},
                    'prefix': 'sample:',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': -0.1,
                'steps': []
            }

        for i in range(0,len(table)):
            frame = {'data': [], 'name': chunkarr[i]}
            data_dict={
                'x':columns,
                'y':table[i]
            }
            frame['data'].append(data_dict)
            figure['frames'].append(frame)
            slider_step={
                'args':[
                    [chunkarr[i]],
                    {'frame': {'duration': 300, 'redraw': False},
                 'mode': 'immediate',
               'transition': {'duration': 300}}
                ],
                'label': chunkarr[i],
                'method': 'animate'   
            }
            sliders_dict['steps'].append(slider_step)
        figure['layout']['sliders'] = [sliders_dict]
        plotly.offline.plot(figure)




