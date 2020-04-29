from tensorflow.keras.models import load_model
import tensorflow as tf
import os

def bilinear_resize(x, rsize):
  return tf.image.resize_bilinear(x, [rsize,rsize], align_corners=True)

model=load_model('models/slim-net-157-0.02.hdf5',compile=False)
model.save('slim-net.h5')
tflite_model = tf.lite.TFLiteConverter.from_keras_model_file('slim-net.h5').convert()
open("models/slim-net.tflite", "wb").write(tflite_model)
os.remove("slim-net.h5")
