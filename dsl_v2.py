import ply.lex as lex
import ply.yacc as yacc
import librosa
import numpy as np
from tensorflow import keras

# Define the AudioLoader class
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

# Define the FeatureExtractor class
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
        
# Define the ModelLoader class
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

# Define the AudioEvaluator class
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
        
# Define the reserved words and tokens
reserved = {
    'AUDIO': 'AUDIO',
    'MODEL': 'MODEL',
    'EVALUATE': 'EVALUATE',
    'USING': 'USING',
}

tokens = [
    'IDENTIFIER',
    'EQUALS',
    'STRING',
] + list(reserved.values())

# Define the regular expressions for simple tokens
t_EQUALS = r'='
t_STRING = r'\"[^\"]*\"'

# Define the rule for an identifier
def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'IDENTIFIER')
    return t

# Define the rule to track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

# Define the rule to ignore whitespace and tabs
t_ignore = ' \t'

# Define the rule for error handling
def t_error(t):
    print(f"Illegal character '{t.value[0]}' at line {t.lineno}")
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()

# Sample DSL statements
dsl_statements = '''
AUDIO a1 = "audios_for_testing/test_mis.WAV"
AUDIO a2 = "audios_for_testing/test_2.WAV"
MODEL m1 = "model/prodetect_cnn.keras"
EVALUATE a1 USING m1
EVALUATE a2 USING m1
'''

# Dictionary to store variables
variables = {}

# Build the parser
def p_start(t):
    '''start : statement
             | start statement'''
    pass

def p_statement_assign(t):
    '''statement : AUDIO IDENTIFIER EQUALS STRING
                 | MODEL IDENTIFIER EQUALS STRING'''
    variable_name = t[2]
    if t[1] == 'AUDIO':
        audio_loader = AudioLoader(t[4][1:-1])  # Extract the path from the quotes
        audio_data, sample_rate = audio_loader.load()
        audio_feature = FeatureExtractor(audio_data, sample_rate)
        features = audio_feature.extract_features()
        variables[variable_name] = {'audio_data': audio_data, 'sample_rate': sample_rate, 'features': features}
    elif t[1] == 'MODEL':
        model_loader = ModelLoader(t[4][1:-1])  # Extract the path from the quotes
        variables[variable_name] = {'model': model_loader.load_model()}

def p_statement_evaluate(t):
    '''statement : EVALUATE IDENTIFIER USING IDENTIFIER'''
    audio_var = variables.get(t[2])
    model_var = variables.get(t[4])
    if audio_var is not None and model_var is not None:
        audio_evaluator = AudioEvaluator(model_var['model'], audio_var['features'])
        prediction = audio_evaluator.evaluate()
        if prediction is not None:
            count_0 = 0
            count_1 = 0
            for i in prediction:
                if i[0] == 0:
                    count_0 += 1
                elif i[0] == 1:
                    count_1 += 1
            # print(prediction)
            if count_0 > count_1:
                print("Bien pronunciado")
            else:
                print("Mal pronunciado")

# Build the parser
parser = yacc.yacc(start='start')

# Parse DSL statements
parser.parse(dsl_statements)
