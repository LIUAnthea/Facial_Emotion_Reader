# EmoView

## Overview
EmoView is an emotion, gender and age detection application designed to revolutionize the way influencers and marketers understand their audience. By analyzing anudience's real-time micro-expressions when they are expose to certain contents, EmoView provides detailed insights into the emotional responses of viewers and viewer's gender/age to the content creater or market researcher. 
The machine learning models used in this project includes: VGG19 and ResNet-18 for emotion analysis; MobileNetV3Small and EfficientNetB0 for age and gender predictions. The classified emotion categories include: anger, disgust, fear, happiness, sadness, surprise, and neutral.

## How it works
- Frontend: WebRTC for capturing video input.
- Backend: Flask for server-side operations.
- Machine Learning: VGG19, ResNet-18 for emotion detection; MobileNetV3Small, EfficientNetB0 for demographic analysis.
- Data Processing: NumPy, TensorFlow for image and video processing.
  
![image](https://github.com/LIUAnthea/Facial_Emotion_Reader/assets/130535253/4768f58b-03c6-461e-b7de-4c5f3ffe2ced)

![image](https://github.com/LIUAnthea/Facial_Emotion_Reader/assets/130535253/a611963f-866d-4095-9455-fcf035fd75bf)

## Key Features
- Real-time emotion detection from facial expressions.
- Analysis of seven universal emotions for comprehensive emotional insights.
- Age and gender prediction to tailor content to specific demographic groups.
