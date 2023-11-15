import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageChops
import torch
import cv2
import pandas as pd
from torchvision import transforms

from model import Model
from train import SAVE_MODEL_PATH

device = "cpu"
model = Model().to(device)
model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

st.write("### Write a digit in the box below.")

stroke_width = st.sidebar.slider("Stroke width:", 1, 25, 9)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create canvas
canvas_result = st_canvas(
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

if canvas_result.image_data is not None:
    image = canvas_result.image_data
    image_pil = Image.fromarray(image)  # Convert NumPy array to PIL image

    image1 = image_pil.copy()
    image1 = image1.convert('L')  # Convert to grayscale
    image1 = image1.resize((28, 28), Image.BICUBIC)

    tensor = transform(image1)
    tensor = tensor.unsqueeze_(0)

    model.eval()
    with torch.no_grad():
        preds = model(tensor)
        preds = preds.detach().numpy()[0]
    
    st.image(image1)
    st.title(np.argmax(preds))
