# model_esc50.py
import os
from preprocess_esc50 import CLASSES
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from preprocess_esc50 import X_train, X_test, y_train, y_test, CLASSES
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from sklearn.metrics import classification_report

# Подготовка данных
num_classes = len(CLASSES)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Улучшенная архитектура модели
model = Sequential([
    Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение с ранней остановкой
history = model.fit(X_train, y_train_cat,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=32,
                    verbose=1)

# Оценка
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=CLASSES))