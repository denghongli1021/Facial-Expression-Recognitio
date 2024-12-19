# -*- coding: utf-8 -*-
import cv2
import os
import warnings
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18
from tqdm import tqdm
import mediapipe as mp
from resnet3D import process_image, augment_frames, video_to_frames, ResNet3DModel

warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, message=".*mediapipe.*")

facial_expression = {
    'Angry' :     0,
    'Happy' :     1,
    'Neutral' :   2,
    'Sad' :       3,
    'Surprised' : 4,
    'Fear' :      5,
    'Disgust' :   6
}

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    else:
        os.makedirs(folder_path)

# 定義把影片切成圖片後的儲存路徑
test_dir = "./extracted_frames"

# 在開始讀影片檔案並切成圖片之前，會清空存圖片的資料夾(若是空資料夾則沒有影響)
clear_folder(test_dir)

# 輸入想要預測影片的資料夾和檔名
video_folder = "./test_videos"
#video_file = "Neutral.mp4"

video_path = os.path.join(video_folder, video_file)

video_to_frames(video_path, test_dir)

frame_paths = [
    os.path.join(test_dir, f) for f in sorted(os.listdir(test_dir)) if f.endswith('.png')
]

if not frame_paths:
    raise ValueError(f"No frames have been extracted to {test_dir}")

frames = process_image(frame_paths, max_frames=16, resize=(112, 112))

if len(frames) == 0:
    raise ValueError(f"Failed to load video from {test_dir}")

frames = augment_frames(frames)
processed_frames = np.transpose(frames, (1, 0, 2, 3)).unsqueeze(0)

# 提供訓練好的model所在路徑
model_dir = "./saved_models"
model_name = "test_model.pth"
model_path = os.path.join(model_dir, model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet3DModel(num_classes=7).to(device)
#model.load_state_dict(torch.load(model_path, weights_only=True))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))


# print("Model weights loaded successfully")
model.eval()

with torch.no_grad():
    inputs = processed_frames.to(device)
    output = model(inputs)

predicted_label = output.argmax(dim=1).item()

facial_expression_reverse = {v: k for k, v in facial_expression.items()}
predicted_expression = facial_expression_reverse.get(predicted_label, "Unknown")

# print預測的表情類別
print(f"Predicted label: {predicted_expression}")