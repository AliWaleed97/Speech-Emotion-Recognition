import pandas as pd
import numpy as np
import plotly
import sys, os, alsaaudio, time, audioop, numpy, glob,  scipy, subprocess, wave, cPickle, threading, shutil
import time
import thread
import shutil
import plotly
import plotly.plotly as py
from django.conf import settings
import speech_recognition as sr
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.fftpack import rfft
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import model_selection
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# from pyAudioAnalysis.audioAnalysisRecordAlsa import recordAudioSegments
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
from django.http import JsonResponse
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
F = []


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

# def Emo(request):
#     try:
#         thread.start_new_thread( recordAudioSegments, (request,'./rec/',1) )
#     except Exception as e:
#         print e
#     else:
#         pass
#     finally:
#         pass
#     return display(request)


# def display(request):
#     return render(request,'liverecord.html',{'F':F})
def feat(request):
    coef = LDA.coef_
    classes =['Fear','Disgust','Happiness','Boredom','Neutral','Sadness','Anger']
    columns = ['ZRC','Energy','Entropy of Energy','Spectral Centroid','Spectral Spread','Spectral Entropy','Spectral Flux','Spectral Rolloff',
          'MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9','MFCC10','MFCC11','MFCC12','MFCC13',
          'CVECTOR1','CVECTOR2','CVECTOR3','CVECTOR4','CVECTOR5','CVECTOR6','CVECTOR7','CVECTOR8','CVECTOR9','CVECTOR10','CVECTOR11','CVECTOR12',
          'Chroma Deviation']
    trace = Heatmap(z=coef,
                   x=columns,
                   y=classes)
    data=[trace]
    return plotly.offline.plot(data, filename='labelled-heatmap')





def recordAudioSegments(request,RecordPath, BLOCKSIZE): 
    print "Press Ctr+C to stop recording"
    RecordPath += os.sep
    d = os.path.dirname(RecordPath)
    if os.path.exists(d) and RecordPath!=".":
        shutil.rmtree(RecordPath)   
    os.makedirs(RecordPath) 

    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,alsaaudio.PCM_NONBLOCK)
    inp.setchannels(1)
    inp.setrate(16000)
    inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    inp.setperiodsize(512)
    midTermBufferSize = int(16000*BLOCKSIZE)
    midTermBuffer = []
    curWindow = []
    elapsedTime = "%08.3f" % (time.time())
    while 1:
            l,data = inp.read()        
            if l:
                for i in range(len(data)/2):
                    curWindow.append(audioop.getsample(data, 2, i))
        
                if (len(curWindow)+len(midTermBuffer)>midTermBufferSize):
                    samplesToCopyToMidBuffer = midTermBufferSize - len(midTermBuffer)
                else:
                    samplesToCopyToMidBuffer = len(curWindow)

                midTermBuffer = midTermBuffer + curWindow[0:samplesToCopyToMidBuffer];
                del(curWindow[0:samplesToCopyToMidBuffer])
            

            if len(midTermBuffer) == midTermBufferSize:
                global F
                # allData = allData + midTermBuffer             
                curWavFileName = RecordPath + os.sep + str(elapsedTime) + ".wav"                
                midTermBufferArray = numpy.int16(midTermBuffer)
                wavfile.write(curWavFileName, 16000, midTermBufferArray)
                # print "AUDIO  OUTPUT: Saved " + curWavFileName
                [Fs2,x] = audioBasicIO.readAudioFile('./rec/'+curWavFileName[5:])
                x = audioBasicIO.stereo2mono(x)
                F = stFeatureExtraction(x,Fs2,Fs2*0.050,Fs2*0.025)
                F = np.mean(F,axis=1)
                print F
                midTermBuffer = []
                elapsedTime = "%08.3f" % (time.time())
                
def livepredict(request):
    path = settings.FILES_DIR
    os.chdir(path)
    files = sorted(glob.glob("*.wav"),key=os.path.getmtime,reverse=True)
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




