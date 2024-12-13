import os
import numpy as np
import cv2
import joblib
import shutil
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# Diretórios de entrada e saída
input_dir = 'input'
output_dir = 'output'
data_dir = 'data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Carregar o modelo VGG16 pré-treinado
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features.flatten()

# Carregar o modelo de similaridade de imagem
similarity_model = joblib.load('image_similarity_model.pkl')

# Carregar características e rótulos
features = np.load('features.npy')
labels = np.load('labels.npy')

# Ler a imagem de entrada
input_image_path = os.path.join(input_dir, 'sua_imagem.jpg')
input_features = extract_features(input_image_path)

# Encontrar as imagens mais similares
distances, indices = similarity_model.kneighbors([input_features])
recommended_labels = labels[indices[0]]

# Copiar as imagens recomendadas para a pasta 'output'
for label in recommended_labels:
    class_dir = os.path.join(data_dir, label)
    img_name = os.listdir(class_dir)[0]  # Seleciona a primeira imagem da categoria
    src_path = os.path.join(class_dir, img_name)
    dst_path = os.path.join(output_dir, f'{label}_{img_name}')
    shutil.copyfile(src_path, dst_path)

print(f'Imagens recomendadas copiadas para a pasta: {output_dir}')
print('Imagens recomendadas:', recommended_labels)
