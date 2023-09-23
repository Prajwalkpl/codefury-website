from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
vect= pickle.load(open('vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/model.html')
def fun_name():
    return render_template("model.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    form_data = [x for x in request.form.values()]
    output=model.predict(vect.transform(form_data).toarray())
    # print(output)
    return render_template('model.html',pred="Current Condition: {}".format(output))

@app.route('/remedy.html')
def remedy():
    return render_template('remedy.html')


if __name__ == '__main__':
    app.run(debug=True)

