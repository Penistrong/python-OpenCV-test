#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   labelYou.py
@Time    :   2019/12/10 23:54:25
@Author  :   Penistrong
@Version :   1.0
@Contact :   770560618@qq.com
@License :   (C)Copyright 2019-2020
@Desc    :   Project of Digital Image Processing
'''

# here put the import lib
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
import os
import threading
import time

class Application():
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Label You")
        self.img = None
        self.img_open = None
        self.img_handler = None
        self.face_recognizer = None
        self.__init_GUI()

    def __init_GUI(self, width=1280, height=720):
        sw = self.window.winfo_screenwidth()
        sh = self.window.winfo_screenheight()
        align = '%dx%d+%d+%d' % (width, height, (sw-width)/2, (sh-height)/2)
        self.window.geometry(align)
        self.window.resizable(width=False,height=False)
        #创建框架
        self.frame = tk.Frame(self.window)
        #创建画布,10X10 Grid里占据左上8X8网格,并且居中显示
        self.canvas = tk.Canvas(self.window, width=width*8/10,height=height*8/10, bg='black')
        self.canvas.grid(row=0,column=0,rowspan=8,columnspan=8,sticky=tk.E+tk.W+tk.N+tk.S,padx=5,pady=5)
        self.canvas.config(highlightbackground='green')

        #创建直方图画布
        self.figure=Figure(figsize=(2.5,4),dpi=100)
        self.figure.add_axes([0,0,256,256], polar=True)
        self.f_plot=self.figure.add_subplot(211)
        #self.f_plot.title='直方图'
        self.c_plot=self.figure.add_subplot(212)
        self.figure.subplots_adjust(hspace=0.4)
        #self.c_plot.title='累积直方图'
        self.hist_canvas=FigureCanvasTkAgg(self.figure, master=self.window)
        self.hist_canvas.get_tk_widget().grid(row=0,column=8,columnspan=2,sticky=tk.E+tk.W+tk.N+tk.S)
        
        # 作者相关信息
        peom = '软工1703\nU201717014\n陈立伟\n数字图像大作业\n2019-12-20'
        self.author_info=tk.Label(self.window,text = peom,fg='gray',font=('微软雅黑',18,'bold'))
        self.author_info.grid(row=7,column=8,sticky=tk.E+tk.W+tk.N+tk.S)

        self.btnOpenPIC = tk.Button(self.window, text='选择图片',command=self.open_Pic,relief='groove', width=10, height = 2)
        self.btnOpenPIC.grid(row = 9, column = 0, columnspan = 1)
        self.btnSavePIC = tk.Button(self.window, text="另存为", command=self.save_Pic,relief='groove', width=10, height = 2)
        self.btnSavePIC.grid(row = 10, column = 0, columnspan = 1)
        self.btnFDimg = tk.Button(self.window, text='图片人脸检测',command=self.faceDetectInImg,relief='groove', width=20, height = 2)
        self.btnFDimg.grid(row = 9, column = 1, columnspan = 2)
        self.btnFaceDetect = tk.Button(self.window, text='摄像头人脸检测',command=self.facedetect,relief='groove', width=20, height = 2)
        self.btnFaceDetect.grid(row = 10, column = 1, columnspan = 2)
        self.btnTransfromGrayImg = tk.Button(self.window, text='灰度图像转换',command=self.transformGrayImg,relief='groove',width=20,height=2)
        self.btnTransfromGrayImg.grid(row = 9, column = 5, columnspan=2)
        self.btnEdgeDetect = tk.Button(self.window, text='边缘检测',command=self.edgeDetect,relief='groove',width=20,height=2)
        self.btnEdgeDetect.grid(row = 10, column = 5, columnspan=2)

        self.btnChooseTrainData = tk.Button(self.window, text='选择训练文件',command=self.chooseDataSet,relief='groove', width=20, height = 2)
        self.btnChooseTrainData.grid(row = 9, column = 3, columnspan = 2)
        self.btnRecognizeImg = tk.Button(self.window, text='预测人像',command=self.predict,relief='groove', width=20, height = 2)
        self.btnRecognizeImg.grid(row = 10, column = 3, columnspan = 2)

    def open_Pic(self):
        '''
        打开文件
        '''
        path = tk.filedialog.askopenfilename(title=u'选择图片', initialdir=(os.path.expanduser('./')))
        if path is not None:
            print('已选择的图片: ', path)
            self.img_open = Image.open(path)
            if self.img_handler is not None:
                self.canvas.delete(self.img_handler)
            i_w = self.img_open.width
            i_h = self.img_open.height
            c_w = self.canvas.winfo_width()
            c_h = self.canvas.winfo_height()
            #将超出画布显示范围的图片缩放至小于等于画布尺寸
            if i_w > c_w:
                k = c_w / i_w
                i_w = round(i_w * k)
                i_h = round(i_h * k)
            if i_h > c_h:
                k = c_h / i_h
                i_w = round(i_w * k)
                i_h = round(i_h * k)
            self.img_open = self.img_open.resize((i_w, i_h), Image.ANTIALIAS)
            self.img = ImageTk.PhotoImage(self.img_open)
            print("画布大小:", str(self.canvas.winfo_width()), "x", str(self.canvas.winfo_height()))
            print("图片大小:", str(self.img.width()),"x", str(self.img.height()))
            self.img_handler = self.canvas.create_image(c_w/2, c_h/2, anchor='center', image=self.img)
            self.showHist()

    #保存处理过的图片
    def save_Pic(self):
        filename = tk.filedialog.asksaveasfilename(title=u'另存为...', filetypes=[("PNG", ".png"),("JPG", ".jpg")], initialdir=(os.path.expanduser('./')))
        if filename is not None and self.img_open is not None:
            im_array = np.array(self.img_open)
            print(im_array.shape)
            save_img = Image.fromarray(np.uint8(im_array))
            save_img.save(filename+".png")
    
    def transformGrayImg(self):
        if self.img_open is None:
            tk.messagebox.showwarning('Warning','Please select a img first...')
            return
        cvImg = np.array(self.img_open)
        #检测是否已为灰度图像,彩色图像通道数为3
        if cvImg.ndim is 3:
            grayImg = cv.cvtColor(cvImg, cv.COLOR_BGR2GRAY)
        else:
            grayImg = cvImg
        self.img_open = Image.fromarray(np.uint8(grayImg))
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        self.img = ImageTk.PhotoImage(self.img_open)
        self.img_handler = self.canvas.create_image(c_w/2, c_h/2, anchor='center', image=self.img)

    def checkIfGray(self, img_arr):
        flag = 1
        for x in range(img_arr.shape[0]):
            for y in range(img_arr.shape[1]):
                r,g,b = img_arr[x, y]
                if (r==g) and (g==b):
                    pass
                else:
                    flag = 0
        return flag

    '''
    def tkImage(self):
        ret,frame=capture.read()
        cvimage = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        pilImage=Image.fromarray(cvimage)
        pilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
        tkImage =  ImageTk.PhotoImage(image=pilImage)
        return tkImage
    '''
    def facedetect(self):
        def cc():
            print("开启摄像头..")
            cap=cv.VideoCapture(0)
            # 加载分类器
            classifier=cv.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
            bColor=(0,225,0)
            c_w = self.canvas.winfo_width()
            c_h = self.canvas.winfo_height()
            while cap.isOpened():
                rtn,frame=cap.read()
                if not rtn:
                    break
                grayImg=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                faceRects=classifier.detectMultiScale(grayImg,scaleFactor=1.2,minNeighbors=3,minSize=(30,30))
                if len(faceRects)>0:
                    for faceRect in faceRects:#标出每一帧人脸
                        x,y,w,h=faceRect #人脸矩形框左上坐标及长宽
                        cv.rectangle(frame,(x-10,y-10),(x+w-10,y+h-10),bColor,2)
                frame_rgba = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
                capImg_open = Image.fromarray(frame_rgba)
                i_w = capImg_open.width
                i_h = capImg_open.height
                if i_w > c_w:
                    k = c_w / i_w
                    i_w = round(i_w * k)
                    i_h = round(i_h * k)
                if i_h > c_h:
                    k = c_h / i_h
                    i_w = round(i_w * k)
                    i_h = round(i_h * k)
                capImg_open = capImg_open.resize((i_w, i_h), Image.ANTIALIAS)
                capImg = ImageTk.PhotoImage(capImg_open)
                #清空canvas里的其他图像
                if self.img_handler is not None:
                    self.canvas.delete(self.img_handler)
                self.img_handler = self.canvas.create_image(c_w/2, c_h/2, anchor='center', image=capImg)
                if self.stopFlag == 1:
                    cap.release()
                    break

        if self.btnFaceDetect['text'] == '摄像头人脸检测':
            self.btnFaceDetect['text'] = '停止检测'
            self.stopFlag = 0
            self.t = threading.Thread(target=cc)
            self.t.start()
        elif self.btnFaceDetect['text'] == '停止检测':
            self.btnFaceDetect['text'] = '摄像头人脸检测'
            self.stopFlag = 1

    def faceDetectInImg(self):
        if self.img_open is None:
            tk.messagebox.showwarning('Warning','Please select a image to detect...')
            return
        bColor=(0,225,0)
        classifier=cv.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
        # opencv基础数据类型为np.ndarray,进行转换后交给cv2进行处理
        cvImg = np.array(self.img_open)
        if cvImg.ndim is 3:
            grayImg = cv.cvtColor(cvImg, cv.COLOR_BGR2GRAY)
        else:
            grayImg = cvImg
        faceRects=classifier.detectMultiScale(grayImg,scaleFactor=1.2,minNeighbors=3,minSize=(30,30))
        if len(faceRects)>0:
            for faceRect in faceRects:#标出每一个人脸
                x,y,w,h=faceRect #人脸矩形框左上坐标及长宽
                cv.rectangle(cvImg,(x-10,y-10),(x+w+10,y+h+10),bColor,2)
            self.img_open = Image.fromarray(np.uint8(cvImg))
        else:
            tk.messagebox.showwarning('Warning','There is no face detected in the image!')
            pass
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        self.img = ImageTk.PhotoImage(self.img_open)
        self.img_handler = self.canvas.create_image(c_w/2, c_h/2, anchor='center', image=self.img)

    def edgeDetect(self):
        if self.img_open is None:
            tk.messagebox.showwarning('Warning','Please select a image frist...')
            return
        cvImg = np.array(self.img_open)
        #首先使用高斯滤波平滑图像,去除噪声
        blur = cv.GaussianBlur(cvImg, (3,3), 0)
        if cvImg.ndim is 3:
            grayImg = cv.cvtColor(cvImg, cv.COLOR_BGR2GRAY)
        else:
            grayImg = cvImg
        x_grad = cv.Sobel(grayImg,cv.CV_16SC1,1,0)
        y_grad = cv.Sobel(grayImg,cv.CV_16SC1,1,0)
        edge_out = cv.Canny(grayImg, 50, 150)
        dst = cv.bitwise_and(cvImg, cvImg, mask=edge_out)
        self.img_open = Image.fromarray(np.uint8(dst))
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        self.img = ImageTk.PhotoImage(self.img_open)
        self.img_handler = self.canvas.create_image(c_w/2, c_h/2, anchor='center', image=self.img)

    def showHist(self):
        self.f_plot.clear()
        cvImg = np.array(self.img_open)
        color=('b','g','r')
        for i,col in enumerate(color):
            c_hist = cv.calcHist([cvImg],[i],None,[256],[0,256])
            self.f_plot.plot(c_hist, color=col)
            hist,bins=np.histogram(cvImg[i].flatten(),256,[0,256])
            cdf=hist.cumsum()
            cdf_normalized=cdf*hist.max()/cdf.max()
            self.c_plot.plot(cdf_normalized,color=col)
        self.f_plot.set_title('Hist',fontsize=10)
        self.f_plot.legend(('BGR'),loc='upper left')
        self.c_plot.set_title('CDF',fontsize=10)
        self.c_plot.legend(('BGR'),loc='upper left')
        self.hist_canvas.draw()

    def detect_face(self,img):
        gray_faces=[]
        face_rects=[]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
        if len(faces) is 0:
            return None, None
        #只取检测出的第一张脸
        for face in faces:
            (x, y, w, h) = face
            gray_faces.append(gray[y:y+w, x:x+h])
            face_rects.append(face)
        return gray_faces,face_rects

    def train_dataset(self, data_folder_path):
        dirs = os.listdir(data_folder_path)
        faces = []
        labels = []
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        for dir_name in dirs:
            label = int(dir_name)
            sub_dir_path = data_folder_path+"/"+dir_name
            sub_img_names = os.listdir(sub_dir_path)
            for img_name in sub_img_names:
                img_path = sub_dir_path+"/"+img_name
                img=cv.imread(img_path)
                gray_faces, rects = self.detect_face(img)
                if gray_faces is not None:
                    for i, face in enumerate(gray_faces):
                        faces.append(face)
                        labels.append(label)
                        self.draw_rect(img, rects[i])
                    #self.img = ImageTk.PhotoImage(Image.fromarray(np.uint8(cv.cvtColor(img, cv.COLOR_BGR2RGBA))))
                    #self.img_handler = self.canvas.create_image(c_w/2, c_h/2, anchor='center', image=self.img)
                    cv.imshow('Training image...',img)
                    cv.waitKey(100)
        cv.destroyAllWindows()
        return faces,labels

    def chooseDataSet(self):
        path = tk.filedialog.askdirectory(title=u'选择训练集', initialdir=(os.path.expanduser('./')))
        print('已选择训练集文件夹:'+path)
        faces,labels = self.train_dataset(path)
        self.face_recognizer = cv.face.LBPHFaceRecognizer_create()
        self.face_recognizer.train(faces, np.array(labels))
        self.subjects = ['Donald.J.Trump','Obama']
    
    def predict(self):
        if self.face_recognizer is None:
            tk.messagebox.showwarning('Warning','Please select the train data set first!')
            return
        elif self.img_open is None:
            tk.messagebox.showwarning('Warning','Please select a image to recognize')
            return
        img = np.array(self.img_open)
        faces, rects = self.detect_face(img)
        s_i = -1
        min_similar = 0 
        #选出偏离度最小的
        for i, face in enumerate(faces):
            label = self.face_recognizer.predict(face)
            print(label)
            if min_similar is 0:
                min_similar = label[1]
                s_i = 0
            elif min_similar > label[1]:
                min_similar = label[1]
                s_i = i
        rect = rects[s_i]
        label = self.face_recognizer.predict(faces[s_i])
        label_text = self.subjects[label[0]]
        print('已识别:'+label_text+" 相似偏离度:"+str(label[1]))
        self.draw_rect(img, rect)
        self.draw_text(img, label_text,rect[0],rect[1] - 5)
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        self.img_open=Image.fromarray(np.uint8(img))
        self.img = ImageTk.PhotoImage(self.img_open)
        self.img_handler = self.canvas.create_image(c_w/2, c_h/2, anchor='center', image=self.img)
    
    def draw_rect(self,img,rect):
        (x,y,w,h)=rect
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 127, 80), 2)

    def draw_text(self, img, text, x, y):
        cv.putText(img, text, (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

    def launch(self):
        self.window.mainloop()

if __name__ == '__main__':
    app = Application()
    app.launch()


    