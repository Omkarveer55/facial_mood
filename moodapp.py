# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:38:32 2022

@author: omkarveerpayal
"""

import pandas as pd
import numpy as np
import keras
from tensorflow.keras.models import model_from_json

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import webbrowser

import streamlit as st
import requests

from PIL import Image

json_file=open('modelface.json','r')
loaded_model=json_file.read()
json_file.close()
loaded_model1=model_from_json(loaded_model)
loaded_model1.load_weights('modelface.h5')

import cv2
emotion_dict={0:'angry',1:'disgust',2:'fear',3:'happy',4:'neutral',5:'sad',6:'surprise'}

def get_songs(emotion):
    sp=spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="4addbe9e9a10408b8d2a09ed1574abcc",
                                                           client_secret="dc1d47769e9b4a10b4dddfdffed37b47"))
    result=sp.search(q=emotion,limit=1,type='playlist')
    list1=[]
    for i in result['playlists']['items']:
        dict1={}
        dict1['playlist_name']=i['name']
        dict1['playlist_id']=i['id']
        dict1['playlist_href']=i['href']
        dict1['spotify_link']=i['external_urls']['spotify']
        list1.append(dict1)
    spotifylink=list1[0]['spotify_link']
    return spotifylink



st.title("Upload images")

file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

#img_cap=cv2.VideoCapture(0)
#img_cap=cv2.VideoCapture(image_file)
#while True:
 #   ret,frame=img_cap.read()
image=Image.open(file)
st.image(image)
image_array=np.array(image)
cv2.imwrite('out.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))



img_cap=cv2.imread('out.jpg')
while True:
    ret,frame=img_cap.read()
    try:
        frame=cv2.resize(img_cap,(1280,720))
    except:
        pass
    if not ret:
        break
    face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    num_faces=face_detector.detectMultiScale(gray_frame,scaleFactor=1.3,minNeighbors=5)

    for (x,y,w,h) in num_faces:
        cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(0,255,0),4)
        region_of_interest=gray_frame[y:y+h,x:x+w]
        cropped_image=np.expand_dims(np.expand_dims(cv2.resize(region_of_interest,(48,48)),-1),0)
        pred=loaded_model1.predict(cropped_image)
        index=int(np.argmax(pred))
        cv2.putText(frame,emotion_dict[index],(x+5,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2, cv2.LINE_AA)
    cv2.imshow('emotions',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
img_cap.release()
cv2.destroyAllWindows()

sp=spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="4addbe9e9a10408b8d2a09ed1574abcc",
                                                           client_secret="dc1d47769e9b4a10b4dddfdffed37b47"))


a_link=get_songs(emotion_dict[index])
text='[link]({link})'.format(link = a_link)
st.markdown(text,unsafe_allow_html=True)



