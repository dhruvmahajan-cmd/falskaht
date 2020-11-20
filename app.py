from flask import Flask, redirect, url_for, render_template, request
import pickle
import numpy as np



app = Flask(__name__)
linear = pickle.load(open('Aht_Model.pickle', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods = ['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        X_Values = [int(x) for x in request.form.values()]
        X_test = [np.array(X_Values)]    
        predictions=linear.predict(X_test)
        output = round(predictions[0], 2)
           
        return render_template('prediction.html', prediction_text = f'Predicted After Call Work (in seconds) is: {output}')
    else:
        return render_template('prediction.html')


if __name__ == "__main__":
    app.run(debug=True)

