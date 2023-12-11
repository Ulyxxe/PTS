import tensorflow as tf
from PIL import Image
import numpy as np

# Load TFLite model
model_path = 'v1\modelsCora\mobilenet_v2_1.0_224_inat_bird_quant.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load labels
labels_path = 'v1\modelsCora\labels.txt'
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
print("Number of labels:", len(labels))




# Prepare input data (modify this according to your model's input requirements)
image_path = 'v1\IMG_05719.jpg'
image = Image.open(image_path)
#image = image.resize((320, 320))  # Adjust size according to your model's input size
image = image.resize((224, 224))  # Adjust size according to your model's input size
#input_data = np.expand_dims(image, axis=0).astype(np.float32) / 255.0  # Normalize to [0, 1]
input_data = np.expand_dims(image, axis=0)

# Set input tensor data
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output tensor data
output_data = interpreter.get_tensor(output_details[0]['index'])

# Map output data to labels
predicted_label_index = np.argmax(output_data)
predicted_label = labels[predicted_label_index]
print("Predicted Label Index:", predicted_label_index)

# Print or use the predicted label
print("Predicted Label:", predicted_label)

# Print the output_data
print("Output Data:", output_data)



