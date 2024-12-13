import numpy as np
from sklearn.neighbors import NearestNeighbors

# Carregar características e rótulos
features = np.load('features.npy')
labels = np.load('labels.npy')

# Treinar um modelo de vizinhos mais próximos
model = NearestNeighbors(n_neighbors=5, algorithm='auto')
model.fit(features)

# Salvar o modelo treinado
import joblib
joblib.dump(model, 'image_similarity_model.pkl')
