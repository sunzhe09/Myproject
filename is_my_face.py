
import tensorflow as tf
import cv2
import sys
import os
import dlib
from sklearn.model_selection import train_test_split
from train_faces import cnnLayer,size,x,keep_prob_5,keep_prob_75

output = cnnLayer()  
predict = tf.argmax(output, 1)  

saver = tf.train.Saver()  
sess = tf.Session()  
path=tf.train.latest_checkpoint('.')
path=path.replace('.\\','./',1)
saver.restore(sess,path)  

def is_my_face(image):  
    res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})  
    if res[0] == 1:  
        return True  
    else:  
        return False  

#ʹ��dlib�Դ���frontal_face_detector��Ϊ���ǵ�������ȡ��
detector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture(0)  
font=cv2.FONT_HERSHEY_COMPLEX

while True:  
    _, img = cam.read()  
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    
    if not len(dets):
        print('Can`t get face.')
        cv2.imshow('img', img)
        key = cv2.waitKey(30) & 0xff  
        if key == 27:
            sys.exit(0)
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1,x2:y2]
        # ����ͼƬ�ĳߴ�
        face = cv2.resize(face, (size,size))
        k=is_my_face(face)
        print('Is this my face? %s' % k)
        if k==True:
            cv2.putText(img,'Me',(x2,x1),cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0 ,255), thickness = 4, lineType = 1)
            #playsound('./hello.mp3')
            
        if k==False:
            cv2.putText(img,'Stranger',(x2,x1),cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0 ,255), thickness = 4, lineType = 1)

        cv2.rectangle(img, (x2,x1),(y2,y1), (255,0,0),3)
        cv2.imshow('img',img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)

sess.close() 
