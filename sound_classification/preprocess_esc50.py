# preprocess_esc50.py
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Параметры
SAMPLE_RATE = 22050
DURATION = 5
N_MFCC = 40  # Увеличили количество MFCC коэффициентов

# Выбраны 20 классов
CLASSES = [
    'rain', 'chirping_birds', 'airplane', 'car_horn', 'train',
    'thunderstorm', 'wind', 'water_drops', 'church_bells', 'clock_alarm',
    'fireworks', 'footsteps', 'frog', 'cow', 'cat', 'dog', 'rooster',
    'sea_waves', 'siren', 'vacuum_cleaner'
]

# Загрузка метаданных
meta = pd.read_csv('ESC-50-master/meta/esc50.csv')
filtered_meta = meta[meta['category'].isin(CLASSES)].reset_index(drop=True)

def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    # Извлекаем MFCC и их статистики
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)
    # Добавляем другие признаки
    chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE)
    chroma_mean = np.mean(chroma.T, axis=0)
    return np.concatenate([mfcc_mean, mfcc_std, chroma_mean])

# Извлечение признаков
X, y = [], []
for _, row in filtered_meta.iterrows():
    file_path = f"ESC-50-master/audio/{row['filename']}"
    features = extract_features(file_path)
    X.append(features)
    y.append(row['category'])

# Кодирование меток
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Стратифицированное разделение
X_train, X_test, y_train, y_test = train_test_split(
    np.array(X), y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)