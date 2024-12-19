import dataProcess
import myGAN
import os
import torch
from PIL import Image
def generate_expression():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = myGAN.Generator().to(device)
    generator.load_state_dict(torch.load('saved_models/generator.pth', map_location=device))
    generator.eval()

    # Preprocess input images
    dataProcess.processor(input_dir='./input', output_face_dir='./output_face')

    # Load test data
    test_loader = dataProcess.getTestLoader(input_test_dir='./output_face')


    output_dir = 'test_generated_images'
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the test dataset
    generator.eval()
    with torch.no_grad():
        for test_image, image_path in test_loader:
            test_image = test_image.to(device)

            # Generate output using the trained generator
            generated_image = generator(test_image).cpu()

            # Denormalize and save the generated image
            generated_image = generated_image[0].permute(1, 2, 0) * 0.5 + 0.5
            generated_image = (generated_image.numpy() * 255).astype('uint8')
            img_pil = Image.fromarray(generated_image)

            # Extract the file name from the image path and save it to output directory
            filename = os.path.basename(image_path[0])
            img_pil.save(f'{output_dir}/generated_{filename}')

    print("Test images processed and saved.")
