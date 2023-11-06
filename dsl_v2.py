import ply.lex as lex
import ply.yacc as yacc
import numpy as np
from audio_classes import AudioLoader, FeatureExtractor, ModelLoader, AudioEvaluator

# Token definitions
tokens = (
    'AUDIO',
    'MODEL',
    'EVALUATE',
    'EQUALS',
    'USING',
    'PLUS',
    'IDENTIFIER',
    'STRING',
)


t_AUDIO = r'AUDIO'
t_MODEL = r'MODEL'
t_EVALUATE = r'EVALUATE'
t_EQUALS = r'='
t_USING = r'USING'
t_STRING = r'"[^"]*"'

# Ignore whitespace
t_ignore = ' \t\n'

# Define a function to track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# Define a function to track identifiers
def t_IDENTIFIER(t):
    r'\w+'
    t.type = 'IDENTIFIER'
    return t

# Error handling
def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

# Error handling
def p_error(p):
    print("Syntax error in input!", p)

# Define a dictionary to store variables (audio and model objects)
variables = {}

# Define the parser
def p_program(p):
    '''program : program statement
               | statement
               | expression
    '''
    pass

def p_statement(p):
    '''
    statement : audio_assignment
              | model_assignment
              | evaluate
    '''
    pass

def p_expression_plus(p):
    '''
    expression : IDENTIFIER PLUS IDENTIFIER
    '''
    if p[1] in variables and p[3] in variables:
        audio_data1, sample_rate1 = variables[p[1]]
        audio_data2, sample_rate2 = variables[p[3]]

        if audio_data1 is not None and audio_data2 is not None:
            result_audio_data = np.concatenate((audio_data1, audio_data2))
            result_sample_rate = sample_rate1
            result_identifier = f"{p[1]}_{p[3]}" 
            variables[result_identifier] = result_audio_data, result_sample_rate
    else:
        print("Audio not found in variables.")

def p_audio_assignment(p):
    '''
    audio_assignment : AUDIO IDENTIFIER EQUALS STRING
    '''
    audio_loader = AudioLoader(p[4][1:-1])  # Remove quotes from the path
    audio_data, sample_rate = audio_loader.load_audio()
    if audio_data is not None:
        variables[p[2]] = audio_data, sample_rate

def p_model_assignment(p):
    '''
    model_assignment : MODEL IDENTIFIER EQUALS STRING
    '''
    model_loader = ModelLoader(p[4][1:-1])  # Remove quotes from the path
    model = model_loader.load_model()
    if model is not None:
        variables[p[2]] = model

def p_evaluate(p):
    '''
    evaluate : EVALUATE IDENTIFIER USING IDENTIFIER
    '''
    if p[2] in variables and p[4] in variables:
        audio_data, sample_rate = variables[p[2]]
        model = variables[p[4]]
        if audio_data is not None and model is not None:
            feature_extractor = FeatureExtractor(audio_data, sample_rate)
            features = feature_extractor.extract_features()
            audio_evaluator = AudioEvaluator(model, features)
            prediction = audio_evaluator.evaluate()
            print("Prediction:", prediction)
    else:
        print("Audio or model not found in variables.")



# Build the lexer
lexer = lex.lex()
# Build the parser
parser = yacc.yacc()

# Sample DSL code
dsl_code = """
AUDIO a1 = "audios_for_testing/test_mis.WAV"
AUDIO a2 = "audios_for_testing/test_2.WAV"
MODEL m1 = "model/prodetect_cnn.keras"
EVALUATE a1 USING m1
"""

# Tokenize the DSL code
lexer.input(dsl_code)

# Display the tokens
for token in lexer:
    print(token)

# Parse the DSL code
parser.parse(dsl_code)