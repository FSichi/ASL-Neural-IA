from flask import Flask, jsonify,request
import json 
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib

app= Flask(__name__)

modelo_entrenado = tf.keras.models.load_model('./modelo2.h5')

OptionsSign = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Nothing', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
NumberOption = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

mapping=dict(zip(OptionsSign,NumberOption)) 
reverse_mapping=dict(zip(NumberOption,OptionsSign)) 


def mapper(value):
    return reverse_mapping[value]

def load_image_to_classify(signType, number):
    ruta = './Imagenes/{0}/{1}.jpg'.format(signType, number)
    image=load_img(ruta, target_size=(40,40))
    image=img_to_array(image) 
    image=image/255.0
    prediction_image=np.array(image)
    prediction_image= np.expand_dims(image, axis=0)
    return prediction_image

def predecir_imagen(signType, number):
    prediction=modelo_entrenado.predict(load_image_to_classify(signType, number))
    value=np.argmax(prediction)
    sign=mapper(value)
    
    predictReturn = { 
        "value": int(value), 
        "sign": sign, 
    }

    return json.dumps(predictReturn)


# RUTAS DE LA APLICACION

@app.route("/")
def home():
    return 'La pagina esta funcionando bien'


@app.route("/predecir", methods=["POST"])
def predecir():
    parametros = request.get_json(force=True)
    prediccionConvert = json.loads(predecir_imagen(parametros['signType'], parametros['number']))
    jsonReturn = { 
        "value": prediccionConvert['value'], 
        "sign": prediccionConvert['sign'],
        "mje":  'La Seña Corresponde a la Letra {0}'.format(prediccionConvert['sign'])
    } 
    return json.dumps(jsonReturn)


if __name__ == '__main__':
    app.run()