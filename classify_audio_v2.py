import librosa
import numpy as np
from tensorflow import keras

# Clase para cargar audio
class AudioLoader:
    def __init__(self, audio_path):
        self.audio_path = audio_path

    def load(self):
        try:
            audio_data, sample_rate = librosa.load(self.audio_path, sr=None)
            return audio_data, sample_rate
        except Exception as e:
            print("Error loading audio:", str(e))
            return None, None

# Clase para extraer características
class FeatureExtractor:
    def __init__(self, audio_data, sample_rate):
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    def extract_features(self):
        try:
            mfcc_features = np.mean(librosa.feature.mfcc(y=self.audio_data, sr=self.sample_rate, n_mfcc=2600).T, axis=0)
            return mfcc_features
        except Exception as e:
            print("Error extracting features:", str(e))
            return None

# Clase para cargar un modelo preentrenado
class ModelLoader:
    def __init__(self, model_filename):
        self.model_filename = model_filename

    def load_model(self):
        try:
            model = keras.models.load_model(self.model_filename)
            return model
        except Exception as e:
            print("Error loading the model:", str(e))
            return None

# Clase para evaluar audio con un modelo
class AudioEvaluator:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def evaluate(self):
        try:
            features = np.array(self.features).reshape(1, -1, 1)
            prediction = (self.model.predict(features) > 0.5).astype(int)
            return prediction
        except Exception as e:
            print("Error evaluating audio:", str(e))
            return None

# Ejemplo de uso
audio_path = "audios_for_testing/test_4.mp3"
model_filename = "model/prodetect_cnn.keras"

audio_loader = AudioLoader(audio_path)
audio_data, sample_rate = audio_loader.load()

if audio_data is not None:
    feature_extractor = FeatureExtractor(audio_data, sample_rate)
    mfcc_features = feature_extractor.extract_features()

    if mfcc_features is not None:
        model_loader = ModelLoader(model_filename)
        model = model_loader.load_model()

        if model is not None:
            audio_evaluator = AudioEvaluator(model, mfcc_features)
            prediction = audio_evaluator.evaluate()
            if prediction is not None:
                if prediction == 1:
                    print("Mal Pronunciado")
                else:
                    print("Bien Pronunciado")
            # print("Prediction:", prediction)
