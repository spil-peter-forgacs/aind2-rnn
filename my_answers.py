import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    # Only proceed, if there is enough data.
    if len(series) > window_size:
        # Create the input/output subsequences.
        for i in range(len(series) - window_size):
            X.append(series[i:i+window_size])
            y.append(series[i + window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # import keras network libraries
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    import keras

    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)


    # TODO: build an RNN to perform regression on our time series input/output data
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))


    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    chars = sorted(list(set(text)))
    print(len(chars), " characters: ", chars)

    # remove as many non-english characters and character sequences as you can 
    # They can have meaning in the text, but let's just use ASCII lower case characters and
    # punctuation includes [' ', '!', ',', '.', ':', ';', '?']
    # (space, eclamation mark, comma, period, colon, semicolon, question mark)
    text = text.replace('"',' ')
    text = text.replace('$',' ')
    text = text.replace('%',' ')
    text = text.replace('&',' ')
    text = text.replace("'",' ')
    text = text.replace('(',' ')
    text = text.replace(')',' ')
    text = text.replace('*',' ')
    text = text.replace('-',' ')
    text = text.replace('/',' ')
    text = text.replace('@',' ')

    # Numbers.
    text = text.replace('0',' ')
    text = text.replace('1',' ')
    text = text.replace('2',' ')
    text = text.replace('3',' ')
    text = text.replace('4',' ')
    text = text.replace('5',' ')
    text = text.replace('6',' ')
    text = text.replace('7',' ')
    text = text.replace('8',' ')
    text = text.replace('9',' ')

    # Convert special characters to English ones, if possible.
    text = text.replace('à','a')
    text = text.replace('â','a')
    text = text.replace('è','e')
    text = text.replace('é','e')

    # shorten any extra dead space created above
    text = text.replace('  ',' ')


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    # Only proceed, if there is enough data.
    if len(text) > window_size:
        # Create input/output subsequences.
        for i in range(0, len(text) - window_size, step_size):
            inputs.append(text[i:i+window_size])
            outputs.append(text[i + window_size])
    
    return inputs,outputs
