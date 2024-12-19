import torch
import torch.nn as nn  
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os

def run_model1() :
    def get_regnet(num_classes):
        model = models.regnet_y_400mf(pretrained=True)
        
        for name, param in model.named_parameters():
            if "trunk_output.block1" in name or "trunk_output.block2" in name:
                param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        return model

    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    emotion_to_label = {idx: emotion for idx, emotion in enumerate(emotions)}

    num_classes = len(emotions)
    model = get_regnet(num_classes=num_classes)
    # model.load_state_dict(torch.load('best_model_fold7.pth', map_location=torch.device('cpu')))
    model_path = os.path.join(os.path.dirname(__file__), 'best_model_fold7.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # image_path = 'test_data.jpg' # 替換為你要輸入照片的路徑，照片要與run_model.py在同一個目錄之下(或是你自己改也行!)
    image_path = 'static/predict_photo/test_data.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_tensor = transform_test(image)
    input_tensor = input_tensor.unsqueeze(0)  
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs.data, 1)

    # 輸出預測結果
    predicted_emotion = emotion_to_label[predicted.item()]
    print(f"Predicted Emotion: {predicted_emotion}")
