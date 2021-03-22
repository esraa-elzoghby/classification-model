
from flask import Flask, render_template, request,redirect
import pandas as pd
import numpy as np
import sys
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@app.route("/")
def modelpage():
    return render_template("/patient_form.html")

@app.route("/final_result.html", methods = ['POST','GET'])
def result():
    if request.method == 'POST':
        age = request.form['age']
        gender = request.form['gender']
        cp = request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        lst = list()
        lst.append((age))
        lst.append((gender))
        lst.append((cp))
        lst.append((trestbps))
        lst.append((chol))
        lst.append((fbs))
        lst.append((restecg))
        lst.append((thalach))
        lst.append((exang))
        lst.append((oldpeak))
        lst.append((slope))
        lst.append((ca))
        lst.append((thal))
        ans = model.predict([np.array(lst, dtype = 'float')])
        result = ans[0]
        print(ans)
        print(lst)
        return render_template("/final_result.html", result=result, lst=lst)
    else:
        return render_template("/patient_form.html")


if __name__ == "__main__":
    app.run(debug = True)
