import librosa
import numpy as np
from tensorflow import keras

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

class AudioDSL:
    def __init__(self):
        self.audio_loader = None
        self.feature_extractor = None
        self.model_loader = None
        self.audio_evaluator = None

    def load_audio(self, audio_path):
        self.audio_loader = AudioLoader(audio_path)
        return self.audio_loader.load()

    def extract_features(self, audio_data, sample_rate):
        self.feature_extractor = FeatureExtractor(audio_data, sample_rate)
        return self.feature_extractor.extract_features()

    def load_model(self, model_filename):
        self.model_loader = ModelLoader(model_filename)
        return self.model_loader.load_model()

    def evaluate_audio(self, model, features):
        self.audio_evaluator = AudioEvaluator(model, features)
        return self.audio_evaluator.evaluate()

# Example DSL usage
dsl = AudioDSL()

# Load audio
audio_path = "audios_for_testing/test_mis.WAV"
audio_data, sample_rate = dsl.load_audio(audio_path)

if audio_data is not None:
    # Extract features
    mfcc_features = dsl.extract_features(audio_data, sample_rate)

    if mfcc_features is not None:
        # Load model
        model_filename = "model/prodetect_cnn.keras"
        model = dsl.load_model(model_filename)

        if model is not None:
            # Evaluate audio
            prediction = dsl.evaluate_audio(model, mfcc_features)
            if prediction is not None:
                if prediction == 1:
                    print("Mal Pronunciado")
                else:
                    print("Bien Pronunciado")
