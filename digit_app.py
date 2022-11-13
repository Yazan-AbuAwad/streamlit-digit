import streamlit as st
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model
import cv2
import numpy as np

model = load_model('cnn_digits.h5')

st.title('Digit Recognizer')
st.markdown('Draw your digit')


canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=15,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=300,
    height=300,
    drawing_mode="freedraw",
    key='canvas')
    
if  canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)
    

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_x=test_x/255
    pred = model.predict(test_x.reshape(1, 28, 28, 1))
    st.write(f'Predicted digit is {np.argmax(pred[0])}')
    c = round(pred[0][np.argmax(pred)]*100,2)
    st.write(f'Confidence = {c}')