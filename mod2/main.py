import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from cnn import CNN  # Import your CNN model class

def load_model(model_path):
    model = CNN()  # Replace with your model class
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Modify normalization for grayscale
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def perform_inference(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    predicted_class = np.argmax(output.numpy())
    return predicted_class

def send_result_to_pipe(pipe_name, result):
    try:
        with open(pipe_name, "w") as pipe:
            pipe.write(str(result))
        print(f"Result sent to the C program: {result}")
    except FileNotFoundError:
        print("Named pipe not found. Make sure the C program is running and has created the pipe.")

if __name__ == "__main__":
    model_path = 'cnn_treinado.pth'  # Path to the trained model
    image_path = 'random_image.png'  # Path to a random dataset image
    #image_path = 'random_image_3_input.png'  # Path to a random dataset image
    pipe_name = '/tmp/my_pipe4'  # Match the pipe name with your C program

    try:
        model = load_model(model_path)
        input_tensor = preprocess_image(image_path)
        result = perform_inference(model, input_tensor)
        send_result_to_pipe(pipe_name, result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
