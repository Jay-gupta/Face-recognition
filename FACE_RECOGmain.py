import numpy as np
import pandas as pd
from sklearn import svm
import cv2
from sklearn import metrics
import os
from sklearn.externals import joblib
import tkinter as tk
import time

e2=0
e3=0
def new_user():
    
    global e2
    global e3

    def submit():
        global e3
        global e2
        e2 = str(e2.get())
        e3 = str(e3.get())
        print(e2)
        print(e3)

    root=tk.Tk()
    root.title("New User?")

    f1=tk.Frame(root,bg='blue')
    f1.pack(side="top",fill="both")
    l1=tk.Label(f1,text="Enter Details",fg="black",bg='yellow',font=('bold'))
    l1.pack(side="top",fill="both")

    f11=tk.Frame(root,bg='white')
    f11.pack(side="top",fill="both")
    l11=tk.Label(f11,text="",fg="white",bg='white')
    l11.pack(side="top",fill="both")

    f2=tk.Frame(root,bg='white')
    f2.pack(side="top",fill="both")
    l2=tk.Label(f2,text="Folder name",fg="black",bg='white',font=('bold'))
    l2.pack(side="left",fill="both",padx=20)
    e2=tk.Entry(f2,fg='black',bg='white')
    e2.pack(side="right", fill="both",padx=50)

    f12=tk.Frame(root,bg='white')
    f12.pack(side="top",fill="both")
    l12=tk.Label(f12,text="",fg="white",bg='white')
    l12.pack(side="top",fill="both")

    f3=tk.Frame(root,bg='white')
    f3.pack(side="top",fill="both")
    l3=tk.Label(f3,text="User Name",fg="black",bg='white',font=('bold'))
    l3.pack(side="left",fill="both",padx=20)
    e3=tk.Entry(f3,fg='black',bg='white')
    e3.pack(side="right", fill="both",padx=50)

    f13=tk.Frame(root,bg='white')
    f13.pack(side="top",fill="both")
    l13=tk.Label(f13,text="",fg="white",bg='white')
    l13.pack(side="top",fill="both")

    f4=tk.Frame(root,bg='white')
    f4.pack(side="top",fill="both")
    b4=tk.Button(f4,text="submit",fg='black',bg='white',command=submit)
    b4.pack(fill="both",padx=50)

    f14=tk.Frame(root,bg='white')
    f14.pack(side="top",fill="both")
    l14=tk.Label(f14,text="",fg="white",bg='white')
    l14.pack(side="top",fill="both")

def load_images():

    global e2
    global e3

    #CREATING FOLDER TO STORE IMAGES

    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. '+  directory)


    f_name = e2
    #f_name = str(input("Enter folder name :"))
    createFolder('C:/Python35/FACE_RECOG/face/'+f_name)
    createFolder('C:/Python35/FACE_RECOG/face_crop/'+f_name)

    name = e3
    #name = input("Enter the name of user : ")
    f = open("C:/Python35/FACE_RECOG/face_name.txt", "a")
    f.write(name+",")
    f.close()
    createFolder('C:/Python35/FACE_RECOG/MASTER/'+name)

    #IMAGE CAPTURING

    cam  = cv2.VideoCapture(0)
    i = 1
    while(1):
        #if(cv2.waitKey(1000) and 0xff==ord('s')):
        ret, img = cam.read()
        cv2.imshow('Live', img)
        cv2.imwrite('C:/Python35/FACE_RECOG/face/'+f_name+'/'+str(i)+'.png',img)
        i=i+1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows() 
    cam.release()

    #FACE DETECTION AND CROPPING

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    r  = np.random.randint(1,300,200)

    for i in range(1,201):
    
        gray = cv2.imread('C:/Python35/FACE_RECOG/face/'+f_name+'/'+str(r[i-1])+'.png', 0)  

        faces = face_cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            gray = cv2.rectangle(gray,(x,y),(x+180,y+180),(255,0,0),2)
            roi_gray = gray[y:y+180, x:x+180]


            cv2.imwrite('C:/Python35/FACE_RECOG/face_crop/'+f_name+'/'+str(i)+'.png',roi_gray)
            cv2.destroyAllWindows()

def train_model():
    Dir = next(os.walk('C:/Python35/FACE_RECOG/face_crop'))[1]
    n = [[i][0] for i in Dir]

    #TESTING AND TRAINING

    x_train = []
    x_test = []
    ##y_train = []
    ##y_test = []

    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    useSignedGradients = True
 
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
                            derivAperture,winSigma,histogramNormType,L2HysThreshold,
                            gammaCorrection,nlevels, useSignedGradients)

    for i in range(0,len(n)):
        for j in range(1,201):
            if(j<141):    #x_train
                img = cv2.imread('C:/Python35/FACE_RECOG/face_crop/'+n[i]+'/'+str(j)+'.png', 0)
                #img = np.reshape(img, (180, 180))
                descriptor = hog.compute(img)
                descriptor = np.ravel(descriptor)
                x_train.append(descriptor)
                #y_train.append(i+1)
            elif(j>140):  #x_test
                img = cv2.imread('C:/Python35/FACE_RECOG/face_crop/'+n[i]+'/'+str(j)+'.png', 0)
                #img = np.reshape(img, (180, 180))
                descriptor = hog.compute(img) 
                descriptor = np.ravel(descriptor)
                x_test.append(descriptor)
                #y_test.append(i+1)

    ##y_train = np.array(y_train)
    ##y_test = np.array(y_test)

    y_train = np.zeros((len(n),140), dtype='int')
    for i in range(0,len(n)):
        y_train[i]  = i+1
    y_train = np.ravel(y_train)

    y_test = np.zeros((len(n),60), dtype='int')
    for i in range(0,len(n)):
        y_test[i]  = i+1
    y_test = np.ravel(y_test)

    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=True, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

    clf = clf.fit(x_train,y_train)

    filename =('C://Python35//FACE_RECOG//face_model//svcmodel.sav')
    joblib.dump(clf, filename)
    #filename.close()

    y_pred = clf.predict(x_test)
    print(y_pred)
    print("SVC_Recognition_accuracy :",metrics.accuracy_score(y_test,y_pred))

y_rtp = 0
def realtime_predict():
    #PREDICTION

    global y_rtp
    
    load_model = joblib.load('C://Python35//FACE_RECOG//face_model//svcmodel.sav')
    #names = ['Jay', 'Salman', 'Vibhor sir', 'Sumer', 'Abhyday', 'Parth', 'Joy']

    f = open("C:/Python35/FACE_RECOG/face_name.txt", "r")
    for i in f:
        names= i.split(",")
    names.pop(-1)
    f.close()
    
    x_rtp = [0]

    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    useSignedGradients = True
 
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
                            derivAperture,winSigma,histogramNormType,L2HysThreshold,
                            gammaCorrection,nlevels, useSignedGradients)

    cam  = cv2.VideoCapture(0)
    while(1):
        ret, img = cam.read()
        gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
       #cv2.imshow('Live_grayscale', gray)

        faces = face_cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+h,y+h),(255,255,255),2)
            roi_g = gray[y:y+h, x:x+h]
            roi_g=np.resize(roi_g,(180,180))

            #cv2.imshow('ROI', roi_g)

            descr = hog.compute(roi_g)
            descr = np.ravel(descr)

            x_rtp[0] = descr
            y_rtp = load_model.predict(x_rtp)
            y_prob = load_model.predict_proba(x_rtp)
            print(y_prob)
            y_prob = np.ravel(y_prob)
            #print(y_prob[y_rtp[0]-1])

            if((y_prob[y_rtp[0]-1])>0.5):

                cv2.putText(img, names[y_rtp[0]-1], (x+h,y), 1, 4, (255,255,255),
                        2, cv2.LINE_AA)

                cv2.imshow('Live', img)

            else:
                
                cv2.putText(img, "unknown", (x+h,y), 1, 4, (255,255,255),
                        2, cv2.LINE_AA)

                cv2.imshow('Live', img)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
                break       
        
    cv2.destroyAllWindows() 
    cam.release()
    
    file_system()

def file_system():

    f = open("C:/Python35/FACE_RECOG/face_name.txt", "r")
    for i in f:
        names= i.split(",")
    names.pop(-1)
    f.close()

    Master_dir = next(os.walk('C:/Python35/FACE_RECOG/MASTER'))[1]

    def open_folder():
        
        os.startfile('C:/Python35/FACE_RECOG/MASTER/'+names[y_rtp[0]-1])

    root=tk.Tk()
    root.title("File System")

    f11=tk.Frame(root,bg='white')
    f11.pack(side="top",fill="both")
    l11=tk.Label(f11,text="",fg="white",bg='white')
    l11.pack(side="top",fill="both")

    f1=tk.Frame(root,bg='blue')
    f1.pack(side="top",fill="both")
    l1=tk.Label(f1,text="WELCOME!",fg="black",bg='white',font=('bold'))
    l1.pack(side="top",fill="both")

    f12=tk.Frame(root,bg='white')
    f12.pack(side="top",fill="both")
    l12=tk.Label(f12,text="",fg="white",bg='white')
    l12.pack(side="top",fill="both")

    f2=tk.Frame(root,bg='blue')
    f2.pack(side="top",fill="both")
    l2=tk.Label(f2,text=names[y_rtp[0]-1],fg="black",bg='cyan',font=('bold'))
    l2.pack(side="top",fill="both")

    f13=tk.Frame(root,bg='white')
    f13.pack(side="top",fill="both")
    l13=tk.Label(f13,text="",fg="white",bg='white')
    l13.pack(side="top",fill="both")

    f14=tk.Frame(root,bg='white')
    f14.pack(side="top",fill="both")
    l14=tk.Label(f14,text="",fg="white",bg='white')
    l14.pack(side="top",fill="both")

    f3=tk.Frame(root,bg='white')
    f3.pack(side="top",fill="both")
    b3=tk.Button(f3,text="Access Folder",fg='black',bg='white',command=open_folder)
    b3.pack(fill="both",padx=50)

    f15=tk.Frame(root,bg='white')
    f15.pack(side="top",fill="both")
    l15=tk.Label(f15,text="",fg="white",bg='white')
    l15.pack(side="top",fill="both")
    
    #print(names[y_rtp[0]-1])
    


root=tk.Tk()
root.title("FACE VAULT")

f1=tk.Frame(root,bg='blue')
f1.pack(side="top",fill="both")
l1=tk.Label(f1,text="Real Time Face Recognition",fg="black",bg='cyan',font=('bold'))
l1.pack(side="top",fill="both")

f11=tk.Frame(root,bg='white')
f11.pack(side="top",fill="both")
l11=tk.Label(f11,text="",fg="white",bg='white')
l11.pack(side="top",fill="both")

f5=tk.Frame(root,bg='white')
f5.pack(side="top",fill="both")
l5=tk.Label(f5,text="Train SVC Model",fg="black",bg='white',font=('bold'))
l5.pack(side="left",fill="both",padx=20)
b5=tk.Button(f5,text="Train Model",fg='black',bg='white',command=train_model)
b5.pack(side="right", fill="both",padx=50)

f15=tk.Frame(root,bg='white')
f15.pack(side="top",fill="both")
l15=tk.Label(f15,text="",fg="white",bg='white')
l15.pack(side="top",fill="both")

f2=tk.Frame(root,bg='white')
f2.pack(side="top",fill="both")
l2=tk.Label(f2,text="Start RTFR",fg="black",bg='white',font=('bold'))
l2.pack(side="left",fill="both",padx=20)
b2=tk.Button(f2,text="Open Camera",fg='black',bg='white',command=realtime_predict)
b2.pack(side="right", fill="both",padx=50)

f12=tk.Frame(root,bg='white')
f12.pack(side="top",fill="both")
l12=tk.Label(f12,text="",fg="white",bg='white')
l12.pack(side="top",fill="both")

f3=tk.Frame(root,bg='blue')
f3.pack(side="top",fill="both")
l3=tk.Label(f3,text="New User?",fg="black",bg='cyan',font=('bold'))
l3.pack(side="top",fill="both")

f13=tk.Frame(root,bg='white')
f13.pack(side="top",fill="both")
l13=tk.Label(f13,text="",fg="white",bg='white')
l13.pack(side="top",fill="both")

f4=tk.Frame(root,bg='white')
f4.pack(side="top",fill="both")
l4=tk.Label(f4,text="Name",fg="black",bg='white',font=('bold'))
l4.pack(side="left",fill="both",padx=20)
b4=tk.Button(f4,text="register",fg='black',bg='white',command=new_user)
b4.pack(side="right", fill="both",padx=50)

f14=tk.Frame(root,bg='white')
f14.pack(side="top",fill="both")
l14=tk.Label(f14,text="",fg="white",bg='white')
l14.pack(side="top",fill="both")

f5=tk.Frame(root,bg='white')
f5.pack(side="top",fill="both")
l5=tk.Label(f5,text="Load face images",fg="black",bg='white',font=('bold'))
l5.pack(side="left",fill="both",padx=20)
b5=tk.Button(f5,text="open camera",fg='black',bg='white',command=load_images)
b5.pack(side="right", fill="both",padx=50)

f15=tk.Frame(root,bg='white')
f15.pack(side="top",fill="both")
l15=tk.Label(f15,text="",fg="white",bg='white')
l15.pack(side="top",fill="both")


