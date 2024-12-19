const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startCameraButton = document.getElementById('start-camera');
const stopCameraButton = document.getElementById('stop-camera');
const flipCameraButton = document.getElementById('flip-camera');
const resultInput = document.getElementById('result');
let stream;
let currentFacingMode = 'user'; // 初始為前置攝像頭
let captureInterval;

// Function to start the camera
startCameraButton.addEventListener('click', async () => {
    await startCamera(currentFacingMode);

    captureInterval = setInterval(() => {
        captureAndPredict();
    }, 1000);
});

// Function to stop the camera
stopCameraButton.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;

        clearInterval(captureInterval);
    }
});

// Function to flip the camera
flipCameraButton.addEventListener('click', async () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
    await startCamera(currentFacingMode);
});

// Function to start the camera with a specified facing mode
async function startCamera(facingMode) {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: facingMode }
        });
        video.srcObject = stream;
    } catch (error) {
        console.error('Error accessing camera:', error);
    }
}

// Function to capture a frame and predict expression
async function captureAndPredict() {
    if (!stream) return;

    // Draw the current video frame to the canvas
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to Blob
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        try {
            // Send the captured frame to the server for prediction
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                resultInput.value = data.result; // Update the result input
            } else {
                console.error('Prediction failed:', response.statusText);
            }
        } catch (error) {
            console.error('Error during prediction:', error);
        }
    }, 'image/jpeg');
}
