from flask import Flask, render_template, request
import json
import pickle
import numpy as np

__locations = []
__data_columns = []
__model = {}

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("Loading the saved artifacts")
    global __data_columns
    global __locations

    with open("columns.json", 'rb') as col:
        __data_columns = json.load(col)['data_columns']
        __locations = __data_columns[4:]

    global __model
    with open("model.pkl", 'rb') as mod:
        __model = pickle.load(mod)

    print("Loading artifacts done")

    percent_loss = request.form['%_material']
    oh_fast = request.form['OH_Fasteners']
    spar_fast = request.form['Spar_Fasteners']
    interface = request.form['Interface']
    location = request.form['Location']

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

    response = __model.predict([x])[0]

    return render_template('prediction.html',
                           prediction_text="Repair Catagory is {}".format(response))


def get_location_names():
    return __locations


def get_data_columns():
    return __data_columns


if __name__ == "__main__":
    print("Starting Python Flask Server for Repir Catagory Classification")
    """load_saved_artifacts()"""
    app.run(debug=True)
