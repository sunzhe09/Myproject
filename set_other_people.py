
import sys
import os
import cv2
import dlib

input_dir = 'F:/samples/lfw/lfw'
output_dir = 'F:/samples/other_faces'
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#ʹ��dlib�Դ���frontal_face_detector��Ϊ���ǵ�������ȡ��
detector = dlib.get_frontal_face_detector()

index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % index)
            img_path = path+'/'+filename
            # ���ļ���ȡͼƬ
            img = cv2.imread(img_path)
            # תΪ�Ҷ�ͼƬ
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # ʹ��detector����������� detsΪ���صĽ��
            dets = detector(gray_img, 1)

            #ʹ��enumerate �������������е�Ԫ���Լ����ǵ��±�
            #�±�i��Ϊ�������
            #left��������߾���ͼƬ��߽�ľ��� ��right�������ұ߾���ͼƬ��߽�ľ��� 
            #top�������ϱ߾���ͼƬ�ϱ߽�ľ��� ��bottom�������±߾���ͼƬ�ϱ߽�ľ���
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                # img[y:y+h,x:x+w]
                face = img[x1:y1,x2:y2]
                # ����ͼƬ�ĳߴ�
                face = cv2.resize(face, (size,size))
                cv2.imshow('image',face)
                # ����ͼƬ
                cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
                index += 1

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)