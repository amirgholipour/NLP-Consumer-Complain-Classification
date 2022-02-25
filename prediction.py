# Imports
import glob
import json
import os
from os.path import splitext,basename
import uuid
import base64

import tensorflow as tf
import joblib
import numpy as np
import json
import traceback
import sys
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load():
    print("Loading model",os.getpid())
    model = tf.keras.models.load_model('./models/model.h5', compile=False)
    labelencoder = joblib.load('./models/labelencoder.pkl')
    tokenizer = joblib.load('./models/tokenizer.pkl')

    print("Loaded model")
    return model, labelencoder,tokenizer




model, labelencoder, tokenizer = load()
print('Models have just loaded!!!!')
def predict(X):
    
    print ('Step1: Loading models')
    print (X['data'])
    # model, labelencoder, tokenizer = load()
    
    print ('Step1 finished!!!!')
    print(labelencoder.classes_)
    # data = request.get("data", {}).get("ndarray")
    # mult_types_array = np.array(data, dtype=object)
    print ('Step2: tokenise the input data.')
    output = tokenizer.texts_to_sequences(X['data'])
    print ('Step2 finished!!!!')
    print(output)
    print ('Step3: Do zero padding on the tokeize data.')
    model_ready_input = pad_sequences(output, maxlen=348,padding='post')
    print(model_ready_input)
    print ('Step3 finished!!!!')
    print ('Step4:  Do prediction!!!')
    result = model.predict(model_ready_input)
    print ('Step4 finished!!!!')
    print(result.shape)
    
    
    
    predicted_class =   np.argmax(result)                                  
    print('Predicted Class name: ', predicted_class)
    predicted_class_prob = str(np.max(result))
    print('Predicted class Certainty: ', predicted_class_prob)
    pred_label = labelencoder.inverse_transform([predicted_class])
    print(pred_label[0])
    print(type(pred_label))
#     print ('step5......')
#     result = tf.sigmoid(result)
#     print(result)
#     result = tf.math.argmax(result,axis=1)
#     print ('step6......')
#     print(result)
#     print(result.shape)
#     pred_label = labelencoder.inverse_transform(result)
#     print(pred_label)
#     print ('step7......')
#     print ('Step 8: Retrun Results!!!', str(pred_label))
    
    json_results = {"Predicted Class": str(predicted_class),"Predicted Class Label": pred_label.tolist(), "Predicted Certainty Score":predicted_class_prob}
    print(json_results)
    return json_results
    

class JsonSerializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (
        np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
