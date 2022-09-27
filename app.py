from flask import Flask,render_template,url_for,request,redirect
import numpy as np
import pandas as pd
import joblib
import pickle


app = Flask(__name__)

model = joblib.load('regressor.pkl')
onehot = joblib.load('sdd.joblib')



@app.route('/')
@app.route('/main')
def main():
	return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features =[[x for x in request.form.values()]]
	print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
	print(int_features)
	c=['baseline value', 'accelerations', 'fetal_movement',
       'uterine_contractions', 'light_decelerations', 'severe_decelerations',
       'prolongued_decelerations', 'abnormal_short_term_variability',
       'mean_value_of_short_term_variability',
       'percentage_of_time_with_abnormal_long_term_variability',
       'mean_value_of_long_term_variability', 'histogram_width',
       'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
       'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
       'histogram_median', 'histogram_variance', 'histogram_tendency']

	df = pd.DataFrame(int_features,columns=c)
	l = onehot.transform(df)
	c = df.columns
	t = pd.DataFrame(l,columns=c)
	result = model.predict(t)
	print("The Result is :",result)


	print(int_features)
	print("*******************************&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&**************")
	print(result)

	return render_template("main.html",prediction_text="Your Vehicle Fuel Consumption  is : {}".format(result))


if __name__ == "__main__":
	app.debug=True
	app.run(host = '127.0.0.1', port =7000)