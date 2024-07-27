import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QDialog
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

class EmotionDetector(QWidget):
    def __init__(self):
        super().__init__()
        
        # Initialize DataFrame
        self.df = pd.DataFrame(columns=['time', 'emotion']) 
        
        # Load the face detection model and the emotion classification model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = load_model('model.h5')
        
        # Define the emotion dictionary
        self.emotion_dict = {0: 'anger', 1: 'contempt', 2: 'disgust',
                             3: 'fear', 4: 'happiness',
                             5: 'sadness', 6: 'surprise'}
        
        # Initialize the user interface
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Emotion Detection')
        self.setStyleSheet("background-color: yellow;")
        
        # Header
        headerLayout = QHBoxLayout()
        profilesLabel = QLabel("Profiles")
        profilesLabel.setFont(QFont('Arial', 14, QFont.Bold))
        headerLayout.addWidget(profilesLabel)
        headerLayout.addStretch(1)
        
        navLinks = ["Home", "About", "Contact", "Tutorial"]
        for link in navLinks:
            navLabel = QLabel(link)
            navLabel.setFont(QFont('Arial', 12))
            headerLayout.addWidget(navLabel)
        
        # Main Content
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(headerLayout)
        
        # Description
        descriptionLayout = QHBoxLayout()
        descriptionLabel = QLabel("Advanced analytics to grow your business")
        descriptionLabel.setFont(QFont('Arial', 24, QFont.Bold))
        descriptionLayout.addWidget(descriptionLabel)
        
        getStartedButton = QPushButton("Get Started")
        getStartedButton.setFont(QFont('Arial', 14))
        getStartedButton.setStyleSheet("background-color: black; color: white;")
        descriptionLayout.addWidget(getStartedButton)
        
        mainLayout.addLayout(descriptionLayout)
        
        # Image
        imageLabel = QLabel()
        imagePath = r"C:\Users\darsh\Documents\WhatsApp Image 2024-06-06 at 23.32.19_a339d9c0.jpg"
        pixmap = QPixmap(imagePath)
        imageLabel.setPixmap(pixmap)
        imageLabel.setScaledContents(True)
        mainLayout.addWidget(imageLabel)
        
        # Functionality Buttons
        buttonLayout = QHBoxLayout()
        
        self.btnLiveFeed = QPushButton('LIVE View')
        self.btnLiveFeed.setFont(QFont('Arial', 14))
        self.btnLiveFeed.clicked.connect(self.startLiveFeed)
        buttonLayout.addWidget(self.btnLiveFeed)
        
        self.btnVideo = QPushButton('Upload Video')
        self.btnVideo.setFont(QFont('Arial', 14))
        self.btnVideo.clicked.connect(self.selectVideoFile)
        buttonLayout.addWidget(self.btnVideo)
        
        self.btnImage = QPushButton('Upload Photo')
        self.btnImage.setFont(QFont('Arial', 14))
        self.btnImage.clicked.connect(self.selectImageFile)
        buttonLayout.addWidget(self.btnImage)
        
        self.btnQuit = QPushButton('EXIT')
        self.btnQuit.setFont(QFont('Arial', 14))
        self.btnQuit.clicked.connect(self.quitApplication)
        buttonLayout.addWidget(self.btnQuit)
        
        mainLayout.addLayout(buttonLayout)
        
        self.label = QLabel()
        mainLayout.addWidget(self.label)
        
        self.setLayout(mainLayout)
        
    def startLiveFeed(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer = QTimer()
        self.timer.timeout.connect(self.detectEmotions)
        self.timer.start(30)  # 30 milliseconds
        
    def selectVideoFile(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Video File')
        if path:
            self.cap = cv2.VideoCapture(path)
            self.timer = QTimer()
            self.timer.timeout.connect(self.detectEmotions)
            self.timer.start(30)  # 30 milliseconds
            
    def selectImageFile(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Image File')
        if path:
            self.detectEmotionsFromFile(path)
            
    def quitApplication(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.showPieChartInNewWindow()
        self.close()
        
    def detectEmotions(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
                roi_color = roi_color.astype('float32') / 255.0
                roi_color = np.expand_dims(roi_color, axis=0)
                prediction = self.model.predict(roi_color)
                emotion = self.emotion_dict[np.argmax(prediction)]
                self.df = pd.concat([self.df, pd.DataFrame({'time': [datetime.now()], 'emotion': [emotion]})], ignore_index=True)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self.displayFrame(frame)
            
    def detectEmotionsFromFile(self, path):
        frame = cv2.imread(path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
            roi_color = roi_color.astype('float32') / 255.0
            roi_color = np.expand_dims(roi_color, axis=0)
            prediction = self.model.predict(roi_color)
            emotion = self.emotion_dict[np.argmax(prediction)]
            self.df = pd.concat([self.df, pd.DataFrame({'time': [datetime.now()], 'emotion': [emotion]})], ignore_index=True)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        self.displayFrame(frame)
        
    def displayFrame(self, frame):
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qImg.rgbSwapped())
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)
        
    def showPieChartInNewWindow(self):
        if not self.df.empty:
            emo_data = self.df.groupby('emotion').size()
            emo_count = [emo_data[emotion] for emotion in self.emotion_dict.values() if emotion in emo_data.index]
            emo_name = [emotion for emotion in self.emotion_dict.values() if emotion in emo_data.index]
            
            plt.figure()
            plt.pie(x=emo_count, labels=emo_name, autopct='%1.2f%%', startangle=90)
            plt.title("FEED BACK OF USER")
            plt.savefig('pie_chart.png')
            plt.close()
            
            self.showChartDialog('pie_chart.png')
    
    def showChartDialog(self, imagePath):
        dialog = QDialog(self)
        dialog.setWindowTitle("feedback OF USER")
        layout = QVBoxLayout()
        
        label = QLabel()
        pixmap = QPixmap(imagePath)
        label.setPixmap(pixmap)
        
        layout.addWidget(label)
        dialog.setLayout(layout)
        dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = EmotionDetector()
    mainWindow.show()
    sys.exit(app.exec_())
