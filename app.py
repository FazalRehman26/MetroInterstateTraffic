from flask import Flask, request, render_template, jsonify
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html') 

@app.route('/insights')
def dashboard():
    return render_template('insights.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    else:
        data=CustomData(
            holiday=request.form.get('holiday'),
            temp=float(request.form.get('temp')),
            clouds_all=float(request.form.get('cloud_all')),
            weather_main=request.form.get('weather_main'),
            weekday=request.form.get('weekday'),
            hour=request.form.get('hour'),
            month=request.form.get('month')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('form.html',final_result=results)
    
if __name__=="__main__":
    app.run(host='127.0.0.1',debug=True, port=8000) 