from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import random
import string
import secrets
import csv
import math
from knn import k,predict,train_features,train_target
from DT import DecisionTree
from knnacc import acc
from DTACC import d_accuracy





app = Flask(__name__)


users=[]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/second')
def second():
    return render_template('sign.html')

@app.route('/third')
def third():
    return render_template('aboutus.html')

@app.route('/fourth')
def fourth():
    return render_template('contactus.html')

@app.route('/fifth')
def fifth():
    return render_template('main.html')

@app.route('/sixth')
def sixth():
    return render_template('signup.html')

@app.route('/seven')
def seven():
    return render_template('upload.html')

@app.route('/metrices')
def metrices():
    return render_template('algomet.html')

@app.route('/graph')
def graph():
    return render_template('graphs.html')



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # read the uploaded file into a pandas dataframe
            df = pd.read_csv(file)
            # render the view.html template with the dataframe
            return render_template('view.html', table=df.to_html(index=False))
    
    return redirect(request.url)



@app.route('/accuracy')
def make_accurate():
   knn_acc=acc 
   dec_acc=d_accuracy
   return render_template('accuracy.html',knn_prediction=knn_acc,dt_prediction=dec_acc)

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/result', methods=['POST'])
def make_prediction():
    patient_details=[]
    # Get user input from the HTML form
    name = request.form['name']
    age = int(request.form['age'])
    weight = float(request.form['weight'])
    sg = float(request.form['sg'])
    alb = float(request.form['alb'])
    egfr = float(request.form['egfr'])
    wbc = float(request.form['wbc'])
    algorithm = request.form['algorithm']
    # Create a list from the user input
    patient_details.append([age, weight, sg, alb, egfr, wbc])

    if algorithm == 'knn':
        prediction = predict(train_features, train_target, patient_details, k)[0]
        if prediction == 0:
            prediction_text = "Mild-Mod CKD"
        elif prediction == 1:
            prediction_text = "ESRD"
        elif prediction == 2:
            prediction_text = "Severe CKD"
        return render_template('result.html', name=name, age=age, weight=weight, sg=sg, alb=alb, egfr=egfr, wbc=wbc, prediction=prediction_text)
    

    elif algorithm == 'decisiontree':
        patient_data = np.array([[age, weight, sg, alb, egfr, wbc]])
        df = pd.read_csv("kidney.csv")
        X = df.drop(columns=["CLASS"]).values
        y = df["CLASS"].values
        train_size = int(0.7 * len(df))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        dt = DecisionTree(max_depth=5)
        dt.fit(X_train, y_train)

        prediction2 = dt.predict1(patient_data)[0]
        if prediction2 == 0:
            prediction_text2 = "Mild-Mod CKD"
        elif prediction2 == 1:
            prediction_text2 = "ESRD"
        elif prediction2 == 2:
            prediction_text2 = "Severe CKD"
        return render_template('resultdec.html', name=name, age=age, weight=weight, sg=sg, alb=alb, egfr=egfr, wbc=wbc, prediction2=prediction_text2)



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get the form data
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form.get('confirm-password', '') # Get the confirm_password field if it exists, otherwise use an empty string

        # Validate the form data
        if not name or not email or not password or not confirm_password:
            return "Please fill out all fields."
        elif password != confirm_password:
            return "Passwords do not match."
        elif email in [user['email'] for user in users]:
            return "An account with that email already exists."
        else:
            # Add the user data to the list
            users.append({
                'name': name,
                'email': email,
                'password': password
            })

            # Redirect to the main page
            return redirect('/main')

    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        # Get the form data
        email = request.form['email']
        password = request.form['password']

        # Validate the form data
        if not email or not password:
            return "Please enter your email and password."
        elif email not in [user['email'] for user in users]:
            return "No account with that email exists."
        else:
            # Check if the password is correct
            user = next(user for user in users if user['email'] == email)
            if password != user['password']:
                return "Incorrect password."
            else:
                # Redirect to the main page
                return redirect('/main')

    return render_template('sign.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/forgot', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']

        # Validate the form data
        if not email:
            return "Please enter your email."

        user = next((user for user in users if user['email'] == email), None)

        if not user:
            return "No account with that email exists."

        # Generate a new random password for the user
        new_password = generate_random_password()

        # Update the user's password in the users list or database
        user['password'] = new_password

        # Send an email to the user with the new password
        

        # Render the forgotpass.html template with the new password
        return render_template('newpass.html', new_password=new_password)

    return render_template('forgot.html')


def generate_random_password(length=12):
    alphabet = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(secrets.choice(alphabet) for i in range(length))
    return password



if __name__ == '__main__':
    app.run(debug=True)
