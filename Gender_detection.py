import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import svm
import pickle

Form = tk.Tk()
Form.title("Gender and Facial Expression Recognition project")
Form.geometry("300x250")

# men average width = 194.69
# men average height = 260.7125

# women average width = 183.178
# women average height = 239.805


def Create_Model(kernel='rbf',c=200,gamma=0.0001):
    "This function create a model with the given hyper parameters \
    and return the model"
    Model=svm.SVC(kernel=kernel,C=c,gamma=gamma,probability=True)
    return Model
def Train_Model(Features,lbls,Model):
    "This function train the given model on the features data and its labels \
     and return the model after train."
    Model.fit(Features,lbls)
    return Model

def Calculate_Accuracy(Features_Data,lbls,Model):
    "This function calculate the accuracy and return it"

    Acc=Model.score(Features_Data,lbls)

    return Acc

def Predict_lbls(Features_Data,Model):
    "This function predict the classes of the given features by the given model and return the labels"

    lbls=Model.predict_proba(Features_Data)
    return lbls

def Calculate_HoG_Feature(Img):
    HoG = cv2.HOGDescriptor()
    return HoG.compute(Img)
def Train_Gender_detection():
    TrainPath_Men = "Project_Data_Set/genderdetectionface/dataset1/train/man"
    TrainPath_Women =  "Project_Data_Set/genderdetectionface/dataset1/train/woman"
    Men_Images = []
    Men_Labels = []

    Women_Images = []
    Women_Labels = []

    All_Train_Data = []
    All_Train_Labels = []



    All_Data_Features = []
    for i in tqdm(os.listdir(TrainPath_Men)):
        Man_Img = cv2.imread(TrainPath_Men + '/' + i , 0)[...,::-1]
        Man_Img = cv2.resize(Man_Img,(64,128))
        Men_Images.append(Man_Img)
        Men_Labels.append(1)

    for i in tqdm(os.listdir(TrainPath_Women)):
        Woman_Img = cv2.imread(TrainPath_Women  + '/' + i , 0)[...,::-1]
        Woman_Img = cv2.resize(Woman_Img,(64,128))
        Women_Images.append(Woman_Img)
        Women_Labels.append(0)

    All_Train_Data = Men_Images + Women_Images
    All_Train_Labels = Men_Labels + Women_Labels

    All_Train_Data = np.array(All_Train_Data)
    All_Train_Labels = np.array(All_Train_Labels)

    Number_of_Samples = All_Train_Data.shape[0]
    permutation = list(np.random.permutation(Number_of_Samples))
    All_Train_Data = All_Train_Data[permutation]
    All_Train_Labels = All_Train_Labels[permutation]
    for sample in All_Train_Data:
        All_Data_Features.append(Calculate_HoG_Feature(sample))

    All_Data_Features = np.array(All_Data_Features)
    All_Data_Features = All_Data_Features.reshape((All_Data_Features.shape[0] , All_Data_Features.shape[1]))
    print(All_Data_Features.shape)

    SVM_Model = Create_Model()
    Trained_SVM_Model = Train_Model(All_Data_Features,All_Train_Labels,SVM_Model)
    Accuracy = Calculate_Accuracy(All_Data_Features,All_Train_Labels,Trained_SVM_Model)
    # Sava model
    pickle.dump(Trained_SVM_Model, open('Model', 'wb'))

    print("The Train Accuracy is ",Accuracy)
    return Trained_SVM_Model

def Generate_Live_Classification():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    SVM_Classifier = pickle.load(open('Model','rb'))#Train_Gender_detection()
    while True:
        ret,img = cap.read()
        if ret:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3,minSize=(5,5))
            for (x,y,h,w) in faces:
                Gray_region = gray[y:y + h, x:x + w]
                Gray_region = cv2.resize(Gray_region,(64,128))
                Feature_Vector = Calculate_HoG_Feature(Gray_region)
                Feature_Vector = np.array(Feature_Vector)
                Feature_Vector = Feature_Vector.reshape((Feature_Vector.shape[1],Feature_Vector.shape[0]))
                prediction = Predict_lbls(Feature_Vector,SVM_Classifier)
                print(prediction)
                if prediction[0][1] > prediction[0][0]:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),3)
                    cv2.putText(img,"Male",(x,y-10) ,  cv2.FONT_HERSHEY_PLAIN ,2 ,color=(255, 255, 0),thickness=2)
                elif prediction[0][1] < prediction[0][0]:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)
                    cv2.putText(img, "Female", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color=(255, 255, 0),thickness=2)
                elif prediction[0][0] == prediction[0][1]:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color=(255, 255, 0),thickness=2)

                #cv2.imwrite("face-" + str(x) + ".jpg", roi_color) for writing faces
            cv2.imshow('img',img)
            k = cv2.waitKey(30)& 0xff
            if k == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

def Browse_for_video():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    SVM_Classifier = pickle.load(open('Model', 'rb'))
    Video_path = filedialog.askopenfilename(filetypes=(("File",".mp4"),("File" , ".mpg"),("File",".avi"),("All Files","*.*")))
    if Video_path != '':
        messagebox.showinfo(title="Success",message=str("Video successfully read"))
    cap = cv2.VideoCapture(Video_path)
    while True:
        ret, img = cap.read()
        if (ret == False):
            break
        else:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(5, 5))
            for (x,y,h,w) in faces:
                Gray_region = gray[y:y + h, x:x + w]
                Gray_region = cv2.resize(Gray_region,(64,128))
                Feature_Vector = Calculate_HoG_Feature(Gray_region)
                Feature_Vector = np.array(Feature_Vector)
                Feature_Vector = Feature_Vector.reshape((Feature_Vector.shape[1],Feature_Vector.shape[0]))
                prediction = Predict_lbls(Feature_Vector,SVM_Classifier)
                print(prediction)
                if prediction[0][1] > prediction[0][0]:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),3)
                    cv2.putText(img,"Male",(x,y-8) ,  cv2.FONT_HERSHEY_PLAIN ,2 ,color=(255, 255, 0),thickness=2)
                elif prediction[0][1] < prediction[0][0]:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cv2.putText(img, "Female", (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 1, color=(255, 255, 0),thickness=2)
                elif prediction[0][0] == prediction[0][1]:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)
                    cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color=(255, 255, 0),thickness=2)

                #cv2.imwrite("face-" + str(x) + ".jpg", roi_color) for writing faces
            cv2.imshow('img',img)
            k = cv2.waitKey(30)& 0xff
            if k == 27:
                break
    cap.release()
    cv2.destroyAllWindows()



Live_Video_Btn = tk.Button(text = "Classify with live video " , command = Generate_Live_Classification)
Live_Video_Btn.place(x=40,y=50)

Search_for_video_Btn = tk.Button(text = "Browse for video" , command = Browse_for_video)
Search_for_video_Btn.place(x=70,y=100)

Form.mainloop()


