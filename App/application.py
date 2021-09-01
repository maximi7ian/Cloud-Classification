from flask import Flask, request, jsonify
from flask.templating import render_template

import numpy as np
from PIL import Image
import io


# Init model
from sagemaker.tensorflow import TensorFlowPredictor


endpoint_name = 'MODEL_ENDPOINT'

def lbl_switch(cls):
    switch = {
        0: 'Blue Sky',
        1: 'Patterned Clouds',
        2: 'Thick Dark Clouds',
        3: 'Thick White Clouds',
        4: 'Veil'
    }
    return switch.get(cls)

predictor = TensorFlowPredictor(endpoint_name)


# Init app
application = Flask(__name__, template_folder='.')


@application.route('/')
def index():
    return render_template('./index.html')
    
@application.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        fs = request.files.get('snap')
        if fs:
            imageStream = io.BytesIO(fs.read())
            imageFile = Image.open(imageStream)
            
            imgarr = np.asarray(imageFile).reshape(1,125,125,3)
            response = predictor.predict(data=imgarr)
            [predictions] = response['predictions']
            rtn = {
                'Class': lbl_switch(np.argmax(predictions)),
                'Pred': np.max(predictions)
            }
            return jsonify(rtn)
            

    
    return 'a mistake was made :('
    
    
if __name__ == '__main__':    
    application.run(debug=True, port=8000)


