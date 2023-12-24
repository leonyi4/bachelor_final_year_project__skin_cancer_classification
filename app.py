import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('best_model.keras')

class_labels = ['Actinic keratoses(akiec)', 'Basal cell carcinoma(bcc)', 'Benign keratosis-like lesions(bkl)', 
                'dermatofibroma(df)', 'Melanoma(mel)', 'Melanocytic nevi(nv)', 'Vascular lesions(vasc)']


def preprocess_image(image):
    # Convert Gradio Image object to PIL Image
    pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
    # Resize the image to (32x32)
    resized_image = pil_image.resize((32, 32))
    # Convert the resized image to a NumPy array
    image_array = np.array(resized_image)
    # Normalize pixel values to [0, 1]
    image_array = image_array / 255.0
    # Reshape the image to match the model input shape
    image_array = image_array.reshape((1, 32, 32, 3)).astype('float32')
    return image_array

def check_img(image):
    if image is not None:
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        
        # Apply softmax to normalize probabilities
        probabilities = np.squeeze(np.exp(prediction) / np.sum(np.exp(prediction), axis=-1))

        # Get top 5 predicted classes and their probabilities using normalized probabilities
        top_classes = np.argsort(probabilities)[::-1][:5]  # Get indices of top 5 classes
        top_classes_probabilities = probabilities[top_classes]  # Get probabilities of top 5 classes

        # Create labels and probabilities for the top 5 classes
        top_classes_labels = [class_labels[idx] for idx in top_classes]
        top_classes_with_probabilities = {
            label: float(prob) for label, prob in zip(top_classes_labels, top_classes_probabilities)
        }
        
        return top_classes_with_probabilities
    else:
        return ''
    
iface = gr.Interface(
    fn=check_img,
    inputs=gr.Image(width=128,height=128,image_mode='RGB',sources='upload'),
    outputs=gr.Label(num_top_classes=5),
    live = True
)
iface.launch()


