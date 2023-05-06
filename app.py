from flask import *  
import os
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import json
import plotly
import plotly.express as px

model=load_model('predd.h5')
class_2_indices = {'melanoma': 0, 'nevus': 1, 'seborrheic_keratoses': 2}
indices_2_class = {v: k for k, v in class_2_indices.items()}

app = Flask(__name__) 
@app.route('/')  
def main():  
    return render_template("upload.html") 

 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  # getting uploaded video
        basepath = os.path.dirname(__file__)
        print("basepath:" + basepath)
        file_path = os.path.join(
            basepath, 'store', f.filename)
        f.save(file_path)
        img = load_img(file_path, target_size=(224,224))
        img = img_to_array(img)/255
        img_expand = np.expand_dims(img, axis=0)
        prediction = model.predict(img_expand, steps=1)
        image_idx = np.argmax(prediction[0])
        prediction_string = indices_2_class[image_idx]
        print("Prediction: {}".format(prediction_string))
        pred_df = pd.DataFrame({'Cancer_type':['melanoma', 'nevus', 'seborrheic keratosis'], 'val':prediction[0]})
        print(pred_df)
        fig = px.bar(pred_df, x='Cancer_type', y='val', title="Predictions",color="Cancer_type",color_discrete_sequence=px.colors.qualitative.Dark2)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('bar.html', graphJSON=graphJSON,type=    )
        
        
if __name__ == '__main__':  
    app.run(debug=True)