# https://devs.tw/post/448
import subprocess
import sys
from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import os
import random
from PIL import Image
import numpy as np
from mimetypes import guess_type 
import torch
import torch.nn as nn  
from torchvision import models, transforms
import cv2
import shutil
from pathlib import Path
import torch.nn.functional as F
from torchvision.models.video import r3d_18
import mediapipe as mp
import dlib 
import myGAN 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = ALL, 1 = INFO, 2 = WARNINGS, 3 = ERRORS


# Import the "generate happy" logic
from generate_happy import generate_emotion
import dataProcess

from resnet3D import ResNet3DModel, process_image, video_to_frames, augment_frames

facial_expression = {
    'Angry' :     0,
    'Happy' :     1,
    'Neutral' :   2,
    'Sad' :       3,
    'Surprise' : 4,
    'Fear' :      5,
    'Disgust' :   6
}

app = Flask(__name__, static_folder='static')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['GENERATED_FOLDER'] = 'generated'
app.config['IMAGE_FOLDER'] = 'expression_images'  
app.config['MODEL_FOLDER'] = app.static_folder  # 靜態文件夾 (static)
app.config['MODEL_FILE'] = 'best_model_fold7.pth'  # 模型文件名稱
app.config['EXTRACTED_FRAMES_FOLDER'] = './extracted_frames'


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXTRACTED_FRAMES_FOLDER'], exist_ok=True)

# sys.path.append(os.path.join(os.path.dirname(__file__), 'static', 'predict_photo'))
# from static.predict_photo.run_model1 import run_model1
# print(run_model1())  
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_to_label = {idx: emotion for idx, emotion in enumerate(emotions)}

def predict_expression(image_path):
    image = Image.open(image_path).convert("L").resize((48, 48))
    image_data = np.array(image) / 255.0
    result = random.choice(emotions)  
    # result = "sad"
    return result

def get_regnet(num_classes):
    model = models.regnet_y_400mf(pretrained=True)
    for name, param in model.named_parameters():
        if "trunk_output.block1" in name or "trunk_output.block2" in name:
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
model_path = os.path.join(app.config['MODEL_FOLDER'], app.config['MODEL_FILE'])
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
num_classes = len(emotions)
model = get_regnet(num_classes=num_classes)
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
# index.html
@app.route('/')
@app.route('/index.html')
def index():
    return send_from_directory('.', 'index.html')
# video.html
@app.route('/video.html')
def video():
    return send_from_directory('.', 'video.html')
@app.route('/generate.html')
def g():
    return send_from_directory('.', 'generate.html')
@app.route('/test.html')
def test():
    return send_from_directory('.', 'test.html')

# static folder
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# photo predict
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 使用 pathlib 清理上传文件夹
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    clear_folder(upload_folder)  # 清理旧文件

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    # 記錄當前上傳的圖片名稱
    

    try:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': f'Could not read image at {file_path}'}), 400

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = transform_test(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        predicted_emotion = emotion_to_label[predicted.item()]
        
        folder_path = os.path.join(app.config['IMAGE_FOLDER'], predicted_emotion)
        if not os.path.exists(folder_path) or not os.listdir(folder_path):
            return jsonify({'error': f'No images found for expression: {predicted_emotion}'}), 404
        image_list = [
            f"/get_image/{predicted_emotion}/{filename}" for filename in os.listdir(folder_path)
        ]

        return jsonify({
            'result': predicted_emotion,
            'images': image_list
        })
        # return jsonify({'result': predicted_emotion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/get_image/<expression>/<filename>', methods=['GET'])
def get_image(expression, filename):
    folder_path = os.path.join(app.config['IMAGE_FOLDER'], expression)
    file_path = os.path.join(folder_path, filename)
    if os.path.exists(file_path):
        mimetype, _ = guess_type(file_path)
        return send_file(file_path, mimetype=mimetype)
    return jsonify({'error': 'Image not found'}), 404

# video predict
@app.route('/predict_video', methods=['POST'])
def predict_video():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded video
    video_filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    file.save(video_path)

    # 初始化人脸检测模块
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    # Extract frames from the video
    test_dir = app.config['EXTRACTED_FRAMES_FOLDER']
    clear_folder(test_dir)
    video_to_frames(video_path, test_dir)

    # Load frames for prediction
    frame_paths = [
        os.path.join(test_dir, f) for f in sorted(os.listdir(test_dir)) if f.endswith('.png')
    ]

    if not frame_paths:
        return jsonify({'error': 'No frames extracted from the video'}), 500


    # Process frames for model input
    frames = process_image(frame_paths, max_frames=16, resize=(112, 112))
    if len(frames) == 0:
        return jsonify({'error': 'Failed to process frames'}), 500

    frames = augment_frames(frames)
    #processed_frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0)
    # 修正形狀順序
    processed_frames = torch.tensor(np.transpose(frames, (1, 0, 2, 3))).unsqueeze(0)

    # Load the trained model
    model_path = os.path.join('./saved_models', 'test_model.pth')
    model = ResNet3DModel(num_classes=7).to(torch.device('cpu'))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Predict the expression
    with torch.no_grad():
        output = model(processed_frames)
        predicted_label = output.argmax(dim=1).item()

    # Map the predicted label to the expression
    facial_expression_reverse = {v: k for k, v in facial_expression.items()}
    predicted_expression = facial_expression_reverse.get(predicted_label, "Unknown")

    # 找到背景圖片和 GIF
    folder_path = os.path.join(app.config['IMAGE_FOLDER'], predicted_expression)
    if not os.path.exists(folder_path) or not os.listdir(folder_path):
        return jsonify({'error': f'No background image found for expression: {predicted_expression}'})

    # Collect all image URLs
    image_list = [
        f"/get_image/{predicted_expression}/{filename}" for filename in os.listdir(folder_path)
    ]

    # Debug: 打印预测结果
    print(f"Predicted label: {predicted_label}")


    # 返回包含背景圖片/GIF 的回應
    return jsonify({
        'expression': predicted_expression,
        'images': image_list
    })
    
@app.route('/generated/<path:filename>')
def serve_generated_images(filename):
    return send_from_directory(app.config['GENERATED_FOLDER'], filename)


@app.route('/generate', methods=['POST'])
def generate_expression():
    # Ensure the request contains the desired expression
    data = request.json
    if 'expression' not in data:
        return jsonify({'error': 'No expression provided'}), 400
    expression = data['expression']

    # Ensure an image has been uploaded
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
    if not uploaded_files:
        return jsonify({'error': 'No image uploaded yet'}), 400

    # Get the first uploaded image
    input_image = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_files[0])

    # Paths for processed and generated images
    processed_folder = app.config['PROCESSED_FOLDER']
    generated_folder = app.config['GENERATED_FOLDER']
    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(generated_folder, exist_ok=True)

    try:
        # Step 1: Process the input image
        dataProcess.processor(input_dir=app.config['UPLOAD_FOLDER'], output_face_dir=processed_folder)
        if not os.listdir(processed_folder):
            raise FileNotFoundError("Processed folder is empty. Check the processor function.")

        processed_image = os.path.join(processed_folder, os.listdir(processed_folder)[0])

        # Step 2: Initialize and load the GAN model for the specific expression
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = myGAN.Generator().to(device)
        generator.load_state_dict(torch.load(f'saved_models/generator_{expression}.pth', map_location=device))
        generator.eval()

        # Step 3: Generate the expression-modified image
        test_loader = dataProcess.getTestLoader(input_test_dir=processed_folder)
        generated_image_path = os.path.join(generated_folder, f'generated_face.jpg')

        with torch.no_grad():
            for test_image, image_path in test_loader:
                test_image = test_image.to(device)
                generated_image = generator(test_image).cpu()

                # Convert the tensor to a PIL image and save
                generated_image = generated_image[0].permute(1, 2, 0) * 0.5 + 0.5
                generated_image = (generated_image.numpy() * 255).astype('uint8')
                img_pil = Image.fromarray(generated_image)
                img_pil.save(generated_image_path)

        # Step 4: Return the generated image URL
        if os.path.exists(generated_image_path):
            return jsonify({'image_url': f"/generated/generated_face.jpg"})
        else:
            return jsonify({'error': 'Generated image not found'}), 404

    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({'error': str(e)}), 500
    
# 清空文件夹
def clear_folder(folder_path):
    folder = Path(folder_path)
    if folder.exists():
        for file in folder.iterdir():
            if file.is_file():
                file.unlink()
    

    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  
    
    app.run(host='0.0.0.0', port=port, debug=True)
