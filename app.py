from flask import Flask, render_template, request, jsonify
import json
import pickle
import numpy as np

__locations = []
__data_columns = []
__model = None

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_location', methods = ["GET"])
def get_location():
    response = jsonify({
        'location': get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    percent_loss = request.form['%_material']
    oh_fast = request.form['OH_Fasteners']
    spar_fast = request.form['Spar_Fasteners']
    interface = request.form['Interface']
    location = request.form['Location']

    response = predict_repair_class(percent_loss, oh_fast, spar_fast, interface, location)

    return render_template('prediction.html',
                            prediction_text="Repair Catagory is {}".format(response))

def predict_repair_class(percent_loss, oh_fast, spar_fast, interface, location):
    try:
        loc_index = __data_columns.index(location)
    except:
        loc_index = -1

    x = np.zeros(13)
    x[0] = percent_loss
    x[1] = oh_fast
    x[2] = spar_fast
    x[3] = interface
    if loc_index >= 0:
        x[loc_index] = 1

    return __model.predict([x])[0]

def load_saved_artifacts():
    print("Loading the saved artifacts")
    global __data_columns
    global __locations

    with open("columns.json", 'rb') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[4:]

    global __model
    with open("model.pkl", 'rb') as f:
        __model = pickle.load(f)

    print("Loading artifacts done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

"""if __name__=="__main__":               # this code is written to test the predictions
    load_saved_artifacts()
    print(get_location_names())
    print(__data_columns)
    print(len(__data_columns))
    print(predict_repair_class(0.3, 15, 10, 1, "12-14"))
    print(predict_repair_class(0.4, 15, 10, 1, "12-14"))
    print(predict_repair_class(0.6, 15, 10, 0, "20-27"))
    print(predict_repair_class(0.6, 15, 10, 0, "10-11"))"""


if __name__=="__main__":
    print("Starting Python Flask Server for Repir Catagory Classification")
    load_saved_artifacts()
    app.run(debug=True)
