'''Change the bestmodel.h5 path as per your preference
    Download bestmodel.h5 from the drive link given in repository
'''

from flask import Flask, render_template,request,session
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import numpy as np
import pickle
import ast
from keras.models import load_model
from keras.utils import load_img, img_to_array
from PIL import Image
import tempfile
import os

app = Flask(__name__)
app.secret_key = 'India@23'

@app.route('/')
def index():
    return render_template('start.html')

@app.route('/start', methods=['POST'])
def home():
    if request.method == 'POST':
        name = request.form.get('name').capitalize()
        email = request.form.get('email')
        session['name'] = name
        session['email'] = email

        print(name,email)
        return render_template('home.html')

@app.route('/stroke')
def stroke():
    name = session.get('name').capitalize()
    print(name)
    email = session.get('email')
    print(email)
    message=''
    return render_template('stroke.html',name=name,email=email,message=message)

@app.route('/diabetes')
def diabetes():
    name = session.get('name').capitalize()
    print(name)
    email = session.get('email')
    print(email)
    message = ''
    return render_template('diabetes.html',name=name,email=email,message=message)

@app.route('/common')
def common():
    name = session.get('name').capitalize()
    print(name)
    email = session.get('email')
    print(email)
    message = ''
    data_ds = pd.read_csv(
        r"disease_symptoms.csv")
    desc = pd.read_csv(r"symptom_Description.csv")
    prec = pd.read_csv(r"symptom_precaution.csv")
    doct = pd.read_csv(r"Doctor_Versus_Disease.csv",
                       encoding='latin1')

    with open(r"unique_symptoms.pkl", "rb") as f:
        unique_sym = pickle.load(f)

    return render_template('common.html',name=name,email=email,message=message,unique_sym=unique_sym)


@app.route('/tumor')
def tumor():
    name = session.get('name').capitalize()
    print(name)
    email = session.get('email')
    print(email)
    message = ''
    return render_template('tumor.html',name=name,email=email,message=message)


@app.route('/diabetes_submit', methods=['GET', 'POST'])
def diabetes_sub():
    name = session.get('name').capitalize()
    email = session.get('email')
    data_dp_1 = pd.read_csv(
        r"diabetes.csv")
    data_dp_2 = pd.read_csv(
        r"diabetes-dataset.csv")
    data = pd.DataFrame(columns=data_dp_1.columns)
    data = pd.concat([data, data_dp_1, data_dp_2], ignore_index=True)

    data.dropna(inplace=True)
    data.dropna(subset=['Outcome'], inplace=True)

    x = data.drop('Outcome', axis=1)
    y = data['Outcome']

    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    y = y.astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

    classifier = svm.SVC(kernel='linear')
    classifier.fit(x_train, y_train)

    x_train_prediction = classifier.predict(x_train)
    training_data_accuracy = accuracy_score(x_train_prediction, y_train)

    x_test_prediction = classifier.predict(x_test)
    testing_data_accuracy = accuracy_score(x_test_prediction, y_test)

    if request.method == 'POST':
        age = request.form.get('age')
        bmi = request.form.get('bmi')
        glucose = request.form.get('glucose')
        bp = request.form.get('bplevel')
        pregnancies = request.form.get('pregnancies')
        insulin = request.form.get('insulin_level')
        skin = request.form.get('skin_level')
        diabetesfun = request.form.get('Diabetesfun')

        input_data = (pregnancies,glucose,bp,skin,insulin,bmi,diabetesfun,age)
        np_array_input_data = np.array(input_data).reshape(1, -1)

        std_inp_data = scaler.transform(np_array_input_data)

        prediction = classifier.predict(std_inp_data)
        print(prediction)
        if prediction[0]==0:
            message = f"{name} is safe"
        else:
            message = f"{name} is suffering from Diabetes"


        return render_template('diabetes.html',message=message)



@app.route('/stroke_submit', methods=['GET', 'POST'])
def stroke_sub():
    name = session.get('name').capitalize()
    email = session.get('email')
    data_bs = pd.read_csv(r"fulldata.csv")

    data_bs.isnull().any().any()
    data_bs.dropna(inplace=True)

    normal = data_bs[data_bs.stroke == 0]
    stroke = data_bs[data_bs.stroke == 1]
    normal = normal.sample(n=248)
    final = pd.concat([normal, stroke], axis=0)

    final.replace({'gender': {'Female': 0, 'Male': 1},
                   'work_type': {'children': 0, 'Govt_job': 1, 'Self-employed': 2, 'Private': 3},
                   'ever_married': {'No': 0, 'Yes': 1},
                   'smoking_status': {'never smoked': 0, 'formerly smoked': 0.5, 'smokes': 1, 'Unknown': 2},
                   'Residence_type': {'Rural': 0, 'Urban': 1}}, inplace=True)
    x = final.drop('stroke', axis=1)
    y = final['stroke']
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    y = y.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=2)
    model = LGBMClassifier()
    model.fit(x_train, y_train)
    x_train_prediction = model.predict(x_train)
    training_data_bs_accuracy = accuracy_score(x_train_prediction, y_train)
    x_test_prediction = model.predict(x_test)
    testing_data_bs_accuracy = accuracy_score(x_test_prediction, y_test)

    if request.method == 'POST':
        age = request.form.get('age')
        bmi = request.form.get('bmi')
        glucose = request.form.get('glucose')
        gender = request.form.get('gender')
        martial_status = request.form.get('marriage')
        hypertension = request.form.get('ht')
        heart = request.form.get('hd')
        local = request.form.get('locality')
        job = request.form.get('job')
        smoke =request.form.get('smoke')
        print(f'Age: {age}, BMI: {bmi}, Gender: {gender}, Glucose: {glucose}, Martial Status: {martial_status}, Hypertension: {hypertension}, Heart Disease: {heart}, Locality: {local}, Job: {job}, Smoking Status: {smoke}')

        if gender=='Male':
            gender=1
        else:
            gender=0

        if hypertension=='htyes':
            ht = 1
        else:
            ht=0

        if heart=='hdyes':
            hd = 1
        else:
            hd=0

        if martial_status == 'married':
            mar_status = 1
        else:
            mar_status = 0

        if smoke =='freq':
            smoke_stat = '1'
        elif smoke=='never':
            smoke_stat = '0'
        elif smoke=='formerly':
            smoke_stat = '0.5'
        else:
            smoke_stat = '2'

        if job == 'Government Job':
            job = 1
        elif job == 'Private Job':
            job = 3
        elif job == 'Self-Employed':
            job = 2
        else:
            job = 0

        if local=='Urban':
            local=1
        else:
            local=0

        input_data_bs = (gender,age,ht,hd,mar_status,job,local,glucose,bmi,smoke_stat)
        print(input_data_bs)
        np_array_input_data_bs = np.array(input_data_bs).reshape(1, -1)

        std_inp_data_bs = scaler.transform(np_array_input_data_bs)

        prediction = model.predict(std_inp_data_bs)
        print(prediction)

        if prediction[0]==0:
            message = f"{name} is safe"
        else:
            message = f"{name} may suffer from brain Stroke"



        return render_template('stroke.html',name=name,email=email,message=message)

@app.route('/diseases_submit', methods=['GET', 'POST'])
def diseases_sub():
    data_ds = pd.read_csv(
        r"disease_symptoms.csv")
    desc = pd.read_csv(r"symptom_Description.csv")
    prec = pd.read_csv(r"symptom_precaution.csv")
    doct = pd.read_csv(r"Doctor_Versus_Disease.csv",
                       encoding='latin1')

    with open(r"unique_symptoms.pkl", "rb") as f:
        unique_sym = pickle.load(f)
    name = session.get('name')
    print(name)
    email = session.get('email')
    print(email)
    message = ''
    if request.method=='POST':
        selected_option_1 = request.form.get('major1')
        selected_option_2 = request.form.get('major2')
        selected_option_3 = request.form.get('minor1')
        selected_option_4 = request.form.get('minor2')
        selected_option_5 = request.form.get('minor3')

        # Iterate over each index in data_ds['Symptoms']
        for index in range(len(data_ds['Symptoms'])):
            # Convert the string representation of the list to an actual list
            symptoms_list = ast.literal_eval(data_ds['Symptoms'][index])

            # Create an empty list to store unique symptoms
            unique_symptoms = []

            # Iterate over each symptom in the list
            for symptom in symptoms_list:
                # Check if the symptom is not already in the unique_symptoms list
                if symptom not in unique_symptoms:
                    # If it's not a duplicate, add it to the unique_symptoms list
                    unique_symptoms.append(symptom)

            # Assign the unique symptoms back to data_ds['Symptoms']
            data_ds['Symptoms'][index] = unique_symptoms

        # print(data_ds['Symptoms'][2])

        prediction = data_ds[data_ds['Symptoms'].apply(lambda
                                                           x: selected_option_1 in x and selected_option_2 in x and selected_option_3 in x and selected_option_4 in x and selected_option_5 in x)]
        if len(prediction) > 0:
            disease_name = prediction['Disease'].iloc[0]
            a = "Predicted Disease is : " + disease_name
        else:
            prediction_2 = data_ds[data_ds['Symptoms'].apply(
                lambda x: selected_option_1 in x and selected_option_2 in x and selected_option_3 in x and (
                        selected_option_4 in x or selected_option_5 in x))]
            if len(prediction_2) > 0:
                disease_name_2 = prediction_2['Disease'].iloc[0]
                a = "Predicted Disease is:  " + disease_name_2
            else:
                prediction_3 = data_ds[data_ds['Symptoms'].apply(
                    lambda x: selected_option_1 in x and selected_option_2 in x and (
                            selected_option_3 in x or selected_option_4 in x or selected_option_5 in x))]
                if len(prediction_3) > 0:
                    disease_name_3 = prediction_3['Disease'].iloc[0]
                    a = "Predicted Disease is : " + disease_name_3
                else:
                    prediction_4 = data_ds[data_ds['Symptoms'].apply(lambda x: selected_option_1 in x and (
                            selected_option_2 in x or selected_option_3 in x or selected_option_4 in x or selected_option_5 in x))]
                    if len(prediction_4) > 0:
                        disease_name_4 = prediction_4['Disease'].iloc[0]
                        a = "Predicted Disease is : " + disease_name_4
                    else:
                        prediction_5 = data_ds[data_ds['Symptoms'].apply(lambda x: selected_option_1 in x or (
                                selected_option_2 in x or selected_option_3 in x or selected_option_4 in x or selected_option_5 in x))]
                        if len(prediction_5) > 0:
                            disease_name_5 = prediction_5['Disease'].iloc[0]
                            a = "Predicted Disease is : " + disease_name_5
                        else:
                            a = "No matching disease found."

        # print(a[23:])
        row = desc[desc['Disease'] == a[23:]]
        row_1 = prec[prec['Disease'] == a[23:]]
        row_2 = doct[doct['Drug Reaction'] == a[23:]]

        # row.iloc[0][1]
        precaution = f'''1) {row_1['Precaution_1'].iloc[0]}
2) {row_1['Precaution_2'].iloc[0]}
3) {row_1['Precaution_3'].iloc[0]}
4) {row_1['Precaution_4'].iloc[0]}'''

        disease = a
        desc = row.iloc[0][1]
        prec =  precaution
        doctor = row_2['Allergist'].iloc[0]

    return render_template('common.html',name=name,email=email,unique_sym=unique_sym,disease=disease,desc=desc,prec=prec,doctor=doctor)

@app.route('/tumor_submit', methods=['GET', 'POST'])
def tumor_sub():
    name = session.get('name')
    print(name)
    email = session.get('email')
    print(email)

    model = load_model(r"bestmodel.h5")

    '''Change the bestmodel.h5 path as per your preference
        Download bestmodel.h5 from the drive link given in repository
    '''
    output_directory = r"mri_folder"  # Replace with the directory where you want to save the image
    output_filename = "mri_image.jpg"  # Replace with your desired output filename (e.g., "output_image.png")
    image_extension = ".jpg"

    # Set a default message in case no file is uploaded
    message = "Please upload an MRI image to get the prediction."
    message2 = ""
    message3 = "The accuracy of this prediction model is 96.97 !"

    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['mri']
        if file:
            # Create the output directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)
            output_path = os.path.join(output_directory, output_filename)
            file.save(output_path)
            print("Image saved successfully.")

            with open(r'mri_folder\mri_image.jpg', "wb") as f:
                f.write(file.getbuffer())

            # Read the image file
            image = Image.open(r'mri_folder\mri_image.jpg')

            # Convert image to grayscale
            image = image.convert("L")

            path = r'mri_folder\mri_image.jpg'
            img = load_img(path, target_size=(224, 224))
            input_arr = img_to_array(img) / 255
            # print(input_arr)
            input_arr = np.expand_dims(input_arr, axis=0)

            pred_prob = model.predict(input_arr)[0][0]
            print(pred_prob)
            if pred_prob > 0.5:
                message = f'The Probability of having Brain Tumor is {round(pred_prob * 100, 4)}%'
                message2 = "Do visit the doctor for further diagnosis and treatment"
            else:
                message = f'The Probability of having Brain Tumor is {round(pred_prob * 100, 4)}%'

    return render_template('tumor.html', name=name, email=email, message=message, message2=message2, message3=message3)

if __name__ == '__main__':
    app.run(debug=True)
