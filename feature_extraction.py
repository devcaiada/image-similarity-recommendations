import os
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# Diretório das imagens
data_dir = 'data/'

# Carregar o modelo VGG16 pré-treinado
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem em {image_path}. Verifique se o caminho está correto e se o arquivo existe.")
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features.flatten()

# Extrair características das imagens e salvar em um arquivo
features = []
labels = []
classes = ['relogio', 'camiseta', 'bicicleta', 'sapato']

for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        try:
            img_features = extract_features(img_path)
            features.append(img_features)
            labels.append(class_name)
        except FileNotFoundError as e:
            print(e)

features = np.array(features)
labels = np.array(labels)

np.save('features.npy', features)
np.save('labels.npy', labels)
