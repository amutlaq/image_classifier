import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import json
from PIL import Image

parser = argparse.ArgumentParser(description="Image Classifier - Predction Part")
parser.add_argument('--input', default='./test_images/cautleya_spicata.jpg',action="store", type=str, help="Image path")
parser.add_argument('--model', default='./classifier.h5',action="store", type=str, help="Checkpoint file path/name")
parser.add_argument('--top_k', dest='top_k', default='5',action="store", type=int, help="Return top k most likely classes")
parser.add_argument('--category_names', dest='category_names',action="store", default='label_map.json',help="Mapping the categories to real names")


arg_parser = parser.parse_args()

image_path = arg_parser.input
model_path = arg_parser.model
top_k = arg_parser.top_k
category_names = arg_parser.category_names


def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()
    

def predict(image_path, model, top_k):
    
    image = Image.open(image_path)
    test_image = np.asarray(image)
    
    processed_test_image = process_image(test_image)
    
    expanded_image = np.expand_dims(processed_test_image, axis=0)
    
    probes_pred = model.predict(expanded_image)
    probes_pred = probes_pred.tolist()
    
    
    probs, classes= tf.math.top_k(probes_pred, k=top_k)
    
    probs=probs.numpy().tolist()[0]
    classes=classes.numpy().tolist()[0]

    return probs, classes


if __name__ == '__main__':
    print('Start Predction !')
    
    with open(category_names,'r') as f:
        class_names = json.load(f)
       
    classifier_2 = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
    
    probs, classes = predict(image_path, classifier_2, top_k)
  
    label_names = [class_names[str(int(key)+1)] for key in classes]


    print('Propabilties: ',probs)
    print('Labels: ',label_names)
    print('Classes: ',classes)
    
    print('End Predction !')
