import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import cv2
import torchvision

Model = torch.load('best_accuracy.pth')

st.write("### Write a digit in the box below.")

stroke_width = st.sidebar.slider("Stroke width:", 1, 25, 9)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create canvas
canvas = st_canvas(
    fill_color="rgba(255, 165, 0, 0,3)",
    stroke_width=stroke_width,
    stroke_color='#000000',
    background_color="#FFFFFF",
    update_streamlit=realtime_update,
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas.image_data is not None:
    # Get numpy array (4-channel RGBA 100, 100, 4)
    input_numpy_array = np.array(canvas.image_data)

    # Get RGBA PIL Image
    input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
    input_image.save('user_input.png')

    # convert to grayscale
    input_image_gray = input_image.convert('L')
    input_image_gray_np = np.asarray(input_image_gray.getdata()).reshape(200,200)

    # create temporary image for opencv to read
    input_image_gray.save('temp_for_cv2.jpg')
    image = cv2.imread('temp_for_cv2.jpg', 0)
    height, width = image.shape
    x,y,w,h = cv2.boundingRect(image)

    # create new blank image and shift ROI to new coordinates
    ROI = image[y:y+h, x:x+w]
    mask = np.zeros([ROI.shape[0] + 10, ROI.shape[1] + 10])
    width, height = mask.shape
    x = width // 2 - ROI.shape[0] // 2
    y = height // 2 - ROI.shape[1] // 2
    mask[y:y+h, x:x+w] = ROI

    output_image = Image.fromarray(mask)
    compressed_output_image = output_image.resize((22, 22), Image.BILINEAR)
    convert_tensor = torchvision.transforms.ToTensor()
    tensor_image = convert_tensor(compressed_output_image)
    tensor_image = tensor_image / 255
    tensor_image = torch.nn.functional.pad(tensor_image, (3,3,3,3), "constant", 0)
    convert_tensor = torchvision.transforms.Normalize((0.1307), (0.3081))
    tensor_image = convert_tensor(tensor_image)

    im = Image.fromarray(tensor_image.detach().cpu().numpy().reshape(28,28), mode='L')
    im.save("processed_tensor.png", "PNG")
    plt.imsave('processed_tensor.png',tensor_image.detach().cpu().numpy().reshape(28,28), cmap='gray')

    device='cpu'

    with torch.no_grad():
        output0 = Model(torch.unsqueeze(tensor_image, dim=0).to(device=device))
        certainty, output = torch.max(output0[0], 0)
        certainty = certainty.clone().cpu().item()
        output = output.clone().cpu().item()
        certainty1, output1 = torch.topk(output0[0],3)
        certainty1 = certainty1.clone().cpu()
        output1 = output1.clone().cpu()

    st.write('### Prediction') 
    st.write('### '+str(output))