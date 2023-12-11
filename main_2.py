import os
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, learning_curve
from tensorflow.keras import models, layers
from tensorflow.keras.layers import LeakyReLU

class DataLoader:
    def __init__(self, audio_path, scores_path, wav_scp_path):
        self.audio_path = audio_path
        self.scores_path = scores_path
        self.wav_scp_path = wav_scp_path

    def load_audio_data(self, audio_path):
        try:
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            return audio_data, sample_rate
        except Exception as e:
            return None, None

    def load_scores(self):
        with open(self.scores_path, 'r') as scores_file:
            return json.load(scores_file)

    def load_wav_scp(self):
        utt2audio_path = {}
        with open(self.wav_scp_path, 'r') as wav_scp_file:
            for line in wav_scp_file:
                utt_id, audio_path = line.strip().split(maxsplit=1)
                utt2audio_path[utt_id] = audio_path
        return utt2audio_path

    def load_data(self):
        pronunciation_scores = self.load_scores()
        utt2audio_path = self.load_wav_scp()

        X = []
        y = []

        for utt_id, pronunciation_score in pronunciation_scores.items():
            text = pronunciation_score["text"]
            accuracy = pronunciation_score["accuracy"]
            total = pronunciation_score["total"]
            audio_path = utt2audio_path.get(utt_id)
            audio_path = os.path.join(self.audio_path, audio_path)
            if audio_path:
                audio_data, sample_rate = self.load_audio_data(audio_path)

                #  MFCC
                if audio_data is not None:
                    mfcc_features = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=2600).T, axis=0)
                    threshold = 6
                    label = 1 if total <= threshold else 0
                    X.append(mfcc_features)
                    y.append(label)
                
                # Spectrogram
                # if audio_data is not None:
                #     mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)
                #     log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
                #     features = np.mean(log_mel_spectrogram.T, axis=0)
                #     threshold = 6
                #     label = 1 if total <= threshold else 0
                #     X.append(features)
                #     y.append(label)

        X = np.array(X)
        y = np.array(y)

        return X, y

class CNNModel:
    def __init__(self, input_shape, output_units=1):
        self.model = models.Sequential()
        
        # Add Convolutional layer with Leaky ReLU activation
        self.model.add(layers.Conv1D(filters=32, kernel_size=3, input_shape=input_shape))
        self.model.add(layers.LeakyReLU(alpha=0.1))
        
        # Add more Convolutional layers with increasing filters
        self.model.add(layers.Conv1D(filters=64, kernel_size=3))
        self.model.add(layers.LeakyReLU(alpha=0.1))
        
        self.model.add(layers.Conv1D(filters=128, kernel_size=3))
        self.model.add(layers.LeakyReLU(alpha=0.1))
        
        # Add MaxPooling layer
        self.model.add(layers.MaxPooling1D(pool_size=2))
        
        # Add Dropout layer for regularization
        self.model.add(layers.Dropout(0.25))
        
        # Flatten the output
        self.model.add(layers.Flatten())
        
        # Add Dense layer with Leaky ReLU activation
        self.model.add(layers.Dense(256))
        self.model.add(layers.LeakyReLU(alpha=0.1))
        
        # Add Dropout layer for regularization
        self.model.add(layers.Dropout(0.5))
        
        # Output layer
        self.model.add(layers.Dense(output_units, activation='sigmoid'))

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train, epochs, batch_size, validation_split):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)

    def evaluate_model(self, X_test, y_test, threshold=0.5):
        y_pred = (self.model.predict(X_test) > threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def save_model(self, model_filename):
        self.model.save(model_filename)

class LearningCurvePlotter:
    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the learning curve.

        Parameters:
        - estimator: Your classifier.
        - title: Title for the chart.
        - X: Feature matrix.
        - y: Target vector.
        - ylim: Tuple (min, max) to define the y-axis limits.
        - cv: Cross-validation splitting strategy.
        - n_jobs: Number of CPU cores to use for parallel computation.
        - train_sizes: Array of training set sizes.
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")

        plt.legend(loc="best")
        return plt

def main():
    print("Preparing dataset...")
    audio_path = 'dataset'
    scores_path = os.path.join(audio_path, 'resource/scores.json')
    wav_scp_path = os.path.join(audio_path, 'resource/wav.scp')

    data_loader = DataLoader(audio_path, scores_path, wav_scp_path)
    X, y = data_loader.load_data()
    
    print("Reshaping dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print("Training model...")
    model = CNNModel(input_shape=(X_train.shape[1], 1))
    model.compile_model()
    history = model.train_model(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2)
    accuracy, report = model.evaluate_model(X_test, y_test)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    print("Exporting model...")
    model.save_model('model/prodetect_cnn_MFCC.keras')

    # Plot training history (optional).
    # plotter = LearningCurvePlotter()
    # plotter.plot_learning_curve(model, "Learning Curve", X, y)

if __name__ == "__main__":
    main()