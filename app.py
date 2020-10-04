
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
with open('insurance_model','rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = list()

    sex = ['male','female']
    smoker = ['yes','no']

    sex_le = LabelEncoder()
    smoker_le = LabelEncoder()

    sex_code = sex_le.fit_transform(sex)
    smoker_code = smoker_le.fit_transform(smoker)

    age = request.form.get('age')
    sex = request.form.get('gender')
    hei = request.form.get('height')
    wei = request.form.get('weight')
    chi = request.form.get('children')
    smo = request.form.get('smoker')

    sex_ind = sex.index(sex)  
    sex_ip = sex_code[sex_ind]

    smo_ind = smoker.index(smo)  
    smoker_ip = smoker_code[smo_ind]

    bmi = float(wei) / (float(hei) * float(hei))
    print("BMI is :" , bmi)

    input_features.append(int(age))
    input_features.append(sex_ip)
    input_features.append(bmi)
    input_features.append(chi)
    input_features.append(smoker_ip)

    final_features = np.array(input_features)

    poly_features = PolynomialFeatures(degree=3, include_bias=False)

    final_features = np.reshape(final_features,(1,final_features.size))
    x_poly_test = poly_features.fit_transform(final_features)

    #final_features = np.reshape(final_features,(1,final_features.size))
    final_features = np.reshape(x_poly_test,(1,x_poly_test.size))

    print(final_features.shape)

    prediction = model.predict(final_features)

    print(prediction)
    prediction = str(prediction)
    prediction = prediction[1:len(prediction)-1]
    prediction =  float(prediction)

    return render_template('index.html', prediction_text='Projected Insurance Amount is : ${:.2f}'.format(prediction))

@app.route('/results',methods=['POST'])
def results():

    #data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])

    #output = prediction[0]
    #return jsonify(output)
    return "hi"

if __name__ == "__main__":
    app.run(debug=True)