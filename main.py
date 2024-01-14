from flask import Flask, render_template, request, redirect, url_for
from flask_bootstrap import Bootstrap
import joblib
import numpy as np
from keras.models import load_model
from keras import backend as K

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField
from wtforms.validators import DataRequired

model=load_model('models/model-174.model')

scaler_data=joblib.load('models/scaler_data.sav')
scaler_target=joblib.load('models/scaler_target.sav')

app=Flask(__name__)
app.config['SECRET_KEY'] = '8BYkEfBA6O6donzWlSihBXox7C0sKR6b'
bootstrap = Bootstrap(app)

class HeartDiseaseForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    gender = StringField("Gender (0-male, 1-female", validators=[DataRequired()])
    age = IntegerField("Age", validators=[DataRequired()])
    tc = IntegerField("TC(mg/dl)", validators=[DataRequired()])
    hdl = IntegerField("HDL(mg/dl)", validators=[DataRequired()])
    smoke = StringField("Smoke (0-no, 1-yes)", validators=[DataRequired()])
    bpm = StringField("Blood Pressure Medication (0-not taking, 1-taking)", validators=[DataRequired()])
    diab = StringField("Diabetics (0-no, 1-yes)", validators=[DataRequired()])
    submit = SubmitField("Submit Details")




@app.route('/')
def index():
    form = HeartDiseaseForm()
    return render_template('patient_details.html', form=form)

@app.route('/get_results',methods=['POST', 'GET'])
def get_results():
    form = HeartDiseaseForm(request.form)

    if form.validate_on_submit():
        test_data = np.array([float(form.gender.data),
                              float(form.age.data), 
                              float(form.tc.data), 
                              float(form.hdl.data),
                              float(form.smoke.data), 
                              float(form.bpm.data), 
                              float(form.diab.data)
                              ]).reshape(1, -1)

        test_data = scaler_data.transform(test_data)
        prediction = model.predict(test_data)
        prediction = scaler_target.inverse_transform(prediction)

        result_dict = {"name": form.name.data, "risk": round(prediction[0][0], 2)}
        # Redirect to the /get_results page with the result_dict
        return redirect(url_for('show_results', **result_dict))

    # If form is not valid, stay on the same page (patient_details.html)
    return render_template('patient_details.html', form=form)

@app.route('/show_results')
def show_results():
    # Retrieve parameters from the URL
    name = request.args.get('name')
    risk = request.args.get('risk')

    result_dict = {"name": name, "risk": risk}
    return render_template('patient_results.html', results=result_dict)




app.run()