# Projeto de Recomendação de Imagens por Similaridade

Este projeto utiliza Machine Learning para recomendar imagens similares com base na similaridade de características extraídas de imagens. O sistema lê imagens de quatro classes (relógio, camiseta, bicicleta, sapato) e retorna recomendações de imagens similares.

## Estrutura do Projeto

```sh
image_similarity_project/
├── data/
│   ├── relógio/
│   ├── camiseta/
│   ├── bicicleta/
│   └── sapato/
├── input/
│   └── sua_imagem.jpg
├── output/
├── feature_extraction.py
├── train_model.py
└── recommend.py
```

## Pré-requisitos

Certifique-se de ter o Python e as seguintes bibliotecas instaladas:

- **TensorFlow**

- **scikit-learn**

- **OpenCV**

- **NumPy**

- **Requests**

- **Pillow**

- **Joblib**

Você pode instalar essas bibliotecas utilizando o pip:

```
pip install tensorflow scikit-learn opencv-python numpy requests pillow joblib
```

## Descrição dos Arquivos

> feature_extraction.py

Este script extrai características de imagens utilizando o modelo VGG16 pré-treinado e as salva em arquivos **.npy**.

```python
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
classes = ['relógio', 'camiseta', 'bicicleta', 'sapato']

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
```

<br></br>

> train_model.py

Este script treina um modelo de vizinhos mais próximos (**KNN**) utilizando as características extraídas e salva o modelo treinado.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib

# Carregar características e rótulos
features = np.load('features.npy')
labels = np.load('labels.npy')

# Treinar um modelo de vizinhos mais próximos
model = NearestNeighbors(n_neighbors=5, algorithm='auto')
model.fit(features)

# Salvar o modelo treinado
joblib.dump(model, 'image_similarity_model.pkl')
```

<br></br>

> recommend.py

Este script carrega o modelo treinado e faz recomendações de imagens similares. Ele copia uma imagem de cada categoria selecionada da pasta data para a pasta **output**.

```python
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
```

## Como Executar

### 1. Preparar as Imagens:

- Coloque as imagens das quatro classes nas respectivas pastas dentro de **data/**.

![relogio](https://github.com/devcaiada/image-similarity-recommendations/blob/main/assets/rel%C3%B3gio.png?raw=true)

- Coloque a imagem que você deseja analisar na pasta **input** com o nome **sua_imagem.jpg**.

![sua_imagem](https://github.com/devcaiada/image-similarity-recommendations/blob/main/input/sua_imagem.jpg?raw=true)

### 2. Extrair Características:

```sh
python feature_extraction.py
```

### 3. Treinar o Modelo de Similaridade de Imagem:

```sh
python train_model.py
```

### 4. Fazer Recomendações de Imagens Similares:

```sh
python recommend.py
```

### 5. As imagens recomendadas serão copiadas para a pasta output e as classes recomendadas serão exibidas no terminal.

![resultado](https://github.com/devcaiada/image-similarity-recommendations/blob/main/assets/resultado.png?raw=true)

<br></br>

![output](https://github.com/devcaiada/image-similarity-recommendations/blob/main/assets/output.png?raw=true)

## Contribuição <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="25" height="25" />

Sinta-se à vontade para contribuir com este projeto. Você pode abrir issues para relatar problemas ou fazer pull requests para melhorias.
