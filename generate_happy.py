import dataProcess
import myGAN
import os
import torch
from PIL import Image


def generate_emotion(emotion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the generator model
    generator = myGAN.Generator().to(device)
    generator.load_state_dict(torch.load(f'saved_models/generator_{emotion}.pth', map_location=device))
    generator.eval()

    # Ensure directories exist
    processed_dir = './processed'
    generated_dir = './generated'
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)

    # Preprocess input images into the `processed` folder
    input_dir = './uploads'
    dataProcess.processor(input_dir=input_dir, output_face_dir=processed_dir)

    # Load processed images from the `processed` folder
    test_loader = dataProcess.getTestLoader(input_test_dir=processed_dir)

    # Generate and save output images to the `generated` folder
    with torch.no_grad():
        for test_image, image_path in test_loader:
            test_image = test_image.to(device)

            # Generate the modified image
            generated_image = generator(test_image).cpu()

            # Denormalize and convert the tensor to a PIL image
            generated_image = generated_image[0].permute(1, 2, 0) * 0.5 + 0.5
            generated_image = (generated_image.numpy() * 255).astype('uint8')
            img_pil = Image.fromarray(generated_image)

            # Extract the filename from the original path and save the generated image
            filename = os.path.basename(image_path[0])
            #output_path = os.path.join(generated_dir, f'generated_{emotion}_{filename}')
            output_path = os.path.join(generated_dir, f'generated_{emotion}_processed_face.jpg')
            img_pil.save(output_path)

    print(f"Generated images saved to {generated_dir}")
    
