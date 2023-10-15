#!/usr/bin/env python
# coding: utf-8


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from keras.callbacks import ReduceLROnPlateau
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator



labels = ['PNEUMONIA', 'NORMAL']
img_size = 128


def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


# In[23]:


train = get_training_data('/Users/baba/Desktop/chest_xray/train')
test = get_training_data('/Users/baba/Desktop/chest_xray/test')
# val = get_training_data('/Users/baba/Desktop/chest_xray/val')


# In[24]:


## Visualistion des images dataset

l = []
for i in train:
    if (i[1] == 0):
        l.append('9')
    else:
        l.append('1')
sns.set_style('darkgrid')
# sns.countplot(l)

plt.figure(figsize=(5, 5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]])

plt.figure(figsize=(5, 5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(labels[train[-1][1]])

# In[ ]:


## Preparation des dataset


# In[25]:


x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

# for feature, label in val:
#    x_val.append(feature)
#    y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

# resize data for deep learning
x_train = x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

# x_val = x_val.reshape(-1, img_size, img_size, 1)
# y_val = np.array(y_val)

x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

datagen = ImageDataGenerator(
    featurewise_center=False,  
    samplewise_center=False, 
    featurewise_std_normalization=False,  
    samplewise_std_normalization=False,  
    zca_whitening=False, 
    rotation_range=30,  
    zoom_range=0.2,  
    width_shift_range=0.1, 
    height_shift_range=0.1,  
    horizontal_flip=True,
    vertical_flip=False) 

datagen.fit(x_train)

# In[61]:


import matplotlib.pyplot as plt

# occurrences de chaque classe dans y_train et y_test
train_counts = np.bincount(y_train)
test_counts = np.bincount(y_test)

# graphique à barres
fig, ax = plt.subplots()
ax.bar(['Pneumonie', 'Sain'], train_counts, label='Entraînement')
ax.bar(['Pneumonie', 'Sain'], test_counts, label='Test', bottom=train_counts)
ax.set_ylabel('Nombre d\'images')
ax.set_title('Répartition des classes dans les ensembles d\'entraînement et de test')
ax.legend()
plt.show()

num_rows = 4  # Nombre de lignes
num_cols = 4  # Nombre de colonnes

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
axes = axes.ravel() 


for i in range(num_rows * num_cols):
    img = x_train[i * 100]

    axes[i].imshow(img, cmap='gray')  
    axes[i].axis('off') 

    if y_train[i * 100] == 0:
        axes[i].set_title('Sain')
    else:
        axes[i].set_title('Pneumonie')
plt.tight_layout() 
plt.show()

img = x_train[0]
pixel_values = img.ravel()

plt.hist(pixel_values, bins=256, color='gray') 
plt.xlabel('Niveau de gris') 
plt.ylabel('Nombre de pixels') 
plt.title('Distribution des pixels dans l\'image de radiographie thoracique')
plt.show()

# ## PARTIE 1 :

# In[68]:


## Modèle MLP. 1


mnist_model = Sequential([
    Flatten(input_shape=(128, 128)),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(2, activation='softmax')
])

mnist_model.summary()

mnist_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

mnist_model.fit(x_train, y_train, epochs=5)

print("\nEvaluation")
mnist_model.evaluate(x_test, y_test)

max_indices = np.argmax(mnist_model.predict(x_test), axis=1)
mlp_metrics = {'Accuracy': accuracy_score(y_test, max_indices),
               'Precision': precision_score(y_test, max_indices, average='weighted'),
               'Recall': recall_score(y_test, max_indices, average='weighted')}
print(mlp_metrics)

# In[94]:


## Modèle MLP 2
Lx = []
Lyloss = []
Lyacc = []
for i in np.arange(0, 10, 0.5):
    mnist_model = Sequential([
        Flatten(input_shape=(128, 128)),
        Dense(128, activation='relu'),
        Dropout(i / 10),  # Prend un nombre entre 0 et 1
        Dense(2, activation='softmax')
    ])

    mnist_model.compile(optimizer=RMSprop(1e-3),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

    mnist_model.fit(x_train, y_train, epochs=5)

    # print("\nEvaluation")
    a = mnist_model.evaluate(x_test, y_test)
    Lx.append(i / 10)
    Lyloss.append(a[0])
    Lyacc.append(a[1])

plt.plot(Lx, Lyloss, label='loss')
plt.plot(Lx, Lyacc, label='acc')
plt.legend()
plt.show()

# In[69]:


## Modèle MLP 3

model = Sequential([
    Flatten(input_shape=(128, 128)),
    Dense(128, activation='relu'),
    # Dropout(0.1),
    Dense(64, activation='relu'),
    Dense(2, activation='sigmoid')  ## Quand on utilise softmax accurancy sur le test desend à 30%
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

max_indices = np.argmax(mnist_model.predict(x_test), axis=1)
mlp_metrics = {'Accuracy': accuracy_score(y_test, max_indices),
               'Precision': precision_score(y_test, max_indices, average='weighted'),
               'Recall': recall_score(y_test, max_indices, average='weighted')}

# In[ ]:


## Influence du nombre d'epoch pour le modèle MLP 3, meme démarche pour les autres modèle
model = Sequential([
    Flatten(input_shape=(128, 128)),
    Dense(128, activation='relu'),
    # Dropout(0.1),
    Dense(64, activation='relu'),
    Dense(2, activation='sigmoid')  ## Quand on utilise softmax accurancy sur le test desend à 30%
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

Lepoch = []
Lacc = []

for i in np.arange(1, 30, 2):
    # Entraînement du modèle
    model.fit(x_train, y_train, epochs=i, validation_data=(x_test, y_test))
    a = model.evaluate(x_test, y_test)
    Lepoch.append(i)
    Lacc.append(a[1])

plt.plot(Lepoch, Lacc)
plt.show()

# ## Modèle CNN

# In[54]:


# Modèle CNN

model = Sequential([
    Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPool2D((2, 2), strides=2, padding='same'),
    Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
    Dropout(0.1),
    BatchNormalization(),
    MaxPool2D((2, 2), strides=2, padding='same'),
    Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D((2, 2), strides=2, padding='same'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=6, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

model.save('cnn.h5')

max_indices = np.argmax(model.predict(x_test), axis=1)
cnn_metrics = {'Accuracy': accuracy_score(y_test, max_indices),
               'Precision': precision_score(y_test, max_indices, average='weighted'),
               'Recall': recall_score(y_test, max_indices, average='weighted')}

# ## Modèle SVM

# In[57]:


# Modèle SVM

from sklearn.svm import SVC
import numpy as np

x_train_flat = np.reshape(x_train, (x_train.shape[0], -1))
x_test_flat = np.reshape(x_test, (x_test.shape[0], -1))

model = SVC(kernel='linear', C=3, gamma='scale')

model.fit(x_train_flat, y_train)

score = model.score(x_test_flat, y_test)
print(f"Accuracy: {score}")

max_indices = model.predict(x_test_flat)
svm_metrics = {'Accuracy': accuracy_score(y_test, max_indices),
               'Precision': precision_score(y_test, max_indices, average='weighted'),
               'Recall': recall_score(y_test, max_indices, average='weighted')}

# ## Regression Logistique

# In[43]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train_flat, y_train)

score = model.score(x_test_flat, y_test)
print(f"Accuracy: {score}")

max_indices = model.predict(x_test_flat)
reg_metrics = {'Accuracy': accuracy_score(y_test, max_indices),
               'Precision': precision_score(y_test, max_indices, average='weighted'),
               'Recall': recall_score(y_test, max_indices, average='weighted')}

# ## Classifieur Bayésien Naif

# In[44]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

model = GaussianNB()

model.fit(x_train_flat, y_train)

y_pred = model.predict(x_test_flat)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

max_indices = model.predict(x_test_flat)
bay_metrics = {'Accuracy': accuracy_score(y_test, max_indices),
               'Precision': precision_score(y_test, max_indices, average='weighted'),
               'Recall': recall_score(y_test, max_indices, average='weighted')}

# ### Partie 2 :  2eme pré-traitement

# In[ ]:


# Bayes améiorer V1

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(x_train_flat, y_train)

print(grid_search.best_params_)

score = grid_search.score(x_test_flat, y_test)
print(f"Accuracy: {score}")

# In[ ]:


# Bayes améiorer V2

from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

pca = PCA(n_components=50)

x_train_pca = pca.fit_transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)

model = GaussianNB()

model.fit(x_train_pca, y_train)

score = model.score(x_test_pca, y_test)
print(f"Accuracy: {score}")

# In[ ]:


## Regression Logistique
# avec Analyse par Composante Principale


model = LogisticRegression()

model.fit(x_train_pca, y_train)

score = model.score(x_test_pca, y_test)
print(f"Accuracy: {score}")

# In[73]:


## Regression Logistique
# avec Analyse par Composante Principale et Optimisation des hyperparamètre

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# pipeline pour la régression logistique avec PCA
logistic_pipeline = make_pipeline(StandardScaler(), PCA(), LogisticRegression())

# hyperparamètres à optimiser
param_grid = {
    'pca__n_components': [2, 5, 10],
    'logisticregression__penalty': ['l1', 'l2'],
    'logisticregression__C': [0.1, 1, 10]
}

# la validation croisée
grid_search = GridSearchCV(logistic_pipeline, param_grid=param_grid, cv=5)
grid_search.fit(x_train_flat, y_train)

print("Meilleurs hyperparamètres : ", grid_search.best_params_)
print("Performance : ", grid_search.best_score_)

# In[75]:


## Regression Logistique
# avec Analyse par Composante Principale et Avec les hyperparamètre de l'optimisation précedente

# Meilleurs hyperparamètres :  {'logisticregression__C': 0.1,
#                              'logisticregression__penalty': 'l2',
#                              'pca__n_components': 10}
# Performance :  0.92906495777359


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# pipeline pour la régression logistique avec PCA
logistic_pipeline = make_pipeline(StandardScaler(), PCA(n_components=10), LogisticRegression(C=0.1, penalty='l2'))

logistic_pipeline.fit(x_train_flat, y_train)

score = logistic_pipeline.score(x_test_flat, y_test)
print(f"Accuracy: {score}")
# Accuracy: 0.7676282051282052


# In[77]:


## Modèle KNN avec Optimisation des paramètre et PCA

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Création du pipeline avec PCA, KNN et une transformation de distance de Manhattan à Euclidienne
knn_pipeline = make_pipeline(StandardScaler(), PCA(), KNeighborsClassifier(metric='euclidean'))

param_grid = {
    'pca__n_components': [2, 5, 10],
    'kneighborsclassifier__n_neighbors': [3, 5, 7, 9],
    'kneighborsclassifier__weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(estimator=knn_pipeline, param_grid=param_grid, cv=5)
grid_search.fit(x_train_flat, y_train)

print(grid_search.best_params_)

score = grid_search.score(x_test_flat, y_test)
print(f"Accuracy: {score}")

# In[ ]:


## MLP
# avec Analyse par Composante Principale


model = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dense(2, activation='sigmoid')  ## Quand on utilise softmax accurancy sur le test desend à 30%
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_pca, y_train, epochs=5, validation_data=(x_test_pca, y_test))
model.evaluate(x_test_pca, y_test)

# In[70]:


# Graphe bilan


labels = list(mlp_metrics.keys())
mlp_values = list(mlp_metrics.values())
cnn_values = list(cnn_metrics.values())
reg_values = list(reg_metrics.values())
svm_values = list(svm_metrics.values())
bay_values = list(bay_metrics.values())

x = np.arange(len(labels))  
width = 0.1  

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width / 2, mlp_values, width, label='MLP', color='blue')
rects2 = ax.bar(x + width / 2, cnn_values, width, label='CNN', color='orange')
rects3 = ax.bar(x + 3 * width / 2, reg_values, width, label='Reg.Log', color='green')
rects4 = ax.bar(x + 5 * width / 2, svm_values, width, label='SVM', color='red')
rects5 = ax.bar(x + 7 * width / 2, bay_values, width, label='Bayesien Naif', color='yellow')

ax.set_ylabel('Valeur des métriques')
ax.set_title('Comparaison des métriques entre Random Forest et Decision Tree')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0, 1])
ax.legend()

plt.show()




