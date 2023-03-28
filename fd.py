import os
import cv2
import numpy as np
import face_recognition
import streamlit as st
from PIL import Image
# Initialize some variables
face_locations=[]
face_encodings = []
face_names = []

# Grab a single frame of video
uploaded_image=st.file_uploader("Upload image")
if(uploaded_image!=None):
  display_image=Image.open(uploaded_image)
  st.image(display_image)
  if st.button("Predict"):
    with open(os.path.join(uploaded_image.name),"wb") as f:
      f.write(uploaded_image.getbuffer())
    img=cv2.imread(os.path.join(uploaded_image.name))
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img,face_locations)
    import pickle
    known_face_names=pickle.load(open("known_face_names.pkl",'rb'))
    known_face_encodings=pickle.load(open("known_face_encodings.pkl",'rb'))
    face_names = []
    for face_encoding in face_encodings:
      matches = face_recognition.compare_faces(face_encoding,known_face_encodings)
      print(matches)
      name = "Unknown"
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      print(face_distances)
      best_match_index = np.argmin(face_distances)
      print(best_match_index)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]

        face_names.append(name)
      else:
        face_names.append("Unknown")
    if len(face_names)==1:
      st.text("Seems like "+face_names[0]+".")
    else:
      st.text("Seems like {} labelled from left to right.".format(face_names))
