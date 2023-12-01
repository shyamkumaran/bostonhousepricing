import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# mdl_lr = pickle.load(open('housemodel.mks','rb'))
mdl_lr = pickle.load(open('house_pricing.pkl','rb'))

@app.route('/')

def home():
    return(render_template('home.html'))

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data = request.json['data']

    #new_data = np.array(list(data.values)).reshape(1,-1)
    new_data = np.array(data['square_feet']).reshape(1,-1)

    output = mdl_lr.predict(new_data)
   
    #print(output[0])
    return(jsonify(output[0]))


@app.route('/predict', methods=['POST'])

def predict():
    data = [float(x) for x in request.form.values()]
    
    new_data = np.array(data).reshape(1,-1) 
    output = mdl_lr.predict(new_data)
    return render_template("home.html",prediction_text="The house price prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
    


