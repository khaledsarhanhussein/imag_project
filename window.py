from tkinter import *
import tkinter as tk
import os
from PIL import Image
import numpy as np
import cv2
import pickle
window = Tk()
#window.configure(background='dodgerblue')
window.title("Image Project")
window.geometry('350x200')
#label
lbl = Label(window, text="Enter your Name")
lbl.grid(column=0, row=1)
#input
txt = Entry(window,width=10)
txt.grid(column=1, row=1)
#r=txt.get()
#button1-make folder
path=""
def mdir():
    r=txt.get()
    path='C:/Users/DELL/Downloads/Documents/image processing/imag_project-master/imag_project-master/images/'+r
    #newpath = r dir
    if not os.path.exists (path ):
        os.makedirs (path)
btn1 = Button(window, text="ENTER",command =mdir)
btn1.grid(column=4, row=1)
#button2
def clicked():
    face_cascade = cv2.CascadeClassifier ( 'cascades/data/haarcascade_frontalface_alt2.xml' )
    eye_cascade = cv2.CascadeClassifier ( 'cascades/data/haarcascade_eye.xml' )
    smile_cascade = cv2.CascadeClassifier ( 'cascades/data/haarcascade_smile.xml' )

    recognizer = cv2.face.LBPHFaceRecognizer_create ()
    recognizer.read ( "./recognizers/face-trainner.yml" )

    labels = {"person_name": 1}
    with open ( "pickles/face-labels.pickle", 'rb' ) as f:
        og_labels = pickle.load ( f )
        labels = {v: k for k, v in og_labels.items ()}

    cap = cv2.VideoCapture (0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read ()
        gray = cv2.cvtColor ( frame, cv2.COLOR_BGR2GRAY )
        faces = face_cascade.detectMultiScale ( gray, scaleFactor=1.5, minNeighbors=5 )
        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
            roi_color = frame[y:y + h, x:x + w]

            # recognize? deep learned model predict keras tensorflow pytorch scikit learn
            id_, conf = recognizer.predict ( roi_gray )
            if conf >= 4 and conf <= 85:
                # print(5: #id_)
                # print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText ( frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA )

            img_item = "9.png"
            cv2.imwrite (img_item, roi_color )
            os.rename(img_item,path+'9.png')
            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle ( frame, (x, y), (end_cord_x, end_cord_y), color, stroke )
        # subitems = smile_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in subitems:
        #	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        # Display the resulting frame
        cv2.imshow ( 'frame', frame )
        if cv2.waitKey ( 20 ) & 0xFF == ord ( 'q' ):
            break

    # When everything done, release the capture
    cap.release ()
    cv2.destroyAllWindows ()


btn2 = Button(window, text="Take a Snap",command =clicked)
btn2.grid(column=5, row=1)
#button3RUN
def train():
    BASE_DIR = os.path.dirname ( os.path.abspath ( __file__ ) )
    image_dir = os.path.join ( BASE_DIR, "images" )

    face_cascade = cv2.CascadeClassifier ( 'cascades/data/haarcascade_frontalface_alt2.xml' )
    recognizer = cv2.face.LBPHFaceRecognizer_create ()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk ( image_dir ):
        for file in files:
            if file.endswith ( "png" ) or file.endswith ( "jpg" ):
                path = os.path.join ( root, file )
                label = os.path.basename ( root ).replace ( " ", "-" ).lower ()
                # print(label, path)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                # print(label_ids)
                # y_labels.append(label) # some number
                # x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
                pil_image = Image.open ( path ).convert ( "L" )  # grayscale
                size = (550, 550)
                final_image = pil_image.resize ( size, Image.ANTIALIAS )
                image_array = np.array ( final_image, "uint8" )
                # print(image_array)
                faces = face_cascade.detectMultiScale ( image_array, scaleFactor=1.5, minNeighbors=5 )

                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append ( roi )
                    y_labels.append ( id_ )

    # print(y_labels)
    # print(x_train)

    with open ( "pickles/face-labels.pickle", 'wb' ) as f:
        pickle.dump ( label_ids, f )

    recognizer.train ( x_train, np.array ( y_labels ) )
    recognizer.save ( "recognizers/face-trainner.yml" )


btn3 = Button(window, text="Train",command =train)
btn3.grid(column=4, row=2)
window.mainloop()
