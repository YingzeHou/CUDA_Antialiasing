from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np


# Create circle
img = Image.new('RGB', (128, 128), color='white')
draw = ImageDraw.Draw(img)
center_x, center_y = img.size[0] // 2, img.size[1] // 2
radius = 50
left_up_point = (center_x - radius, center_y - radius)
right_down_point = (center_x + radius, center_y + radius)
draw.ellipse([left_up_point, right_down_point], outline='black', width=3)
img.save('out/hollow_black_circle.png')


# TensorFlow resizing
img_tensor = tf.convert_to_tensor(np.array(img))
resized_tensorflow = tf.image.resize(img_tensor, (16, 16), antialias=True)
resized_tensorflow = Image.fromarray(resized_tensorflow.numpy().astype(np.uint8))
resized_tensorflow.save('out/blurred_tf.png')
