# Metro Interstate Traffic Volume Prediction

This project predicts the traffic volume on the Metro Interstate highway based on weather and time-related features, using machine learning models. The application is built with Flask and provides insights from historical traffic data through static plots.

---

## Features

- Predicts traffic volume based on:
  - Holiday, Temperature, Cloud coverage, Weather condition, Day of the week, Hour of the day, Month
- Displays insights using static plots from the traffic dataset
- Built with Flask for local use

---

## Project Structure

```
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── raw.csv
│   ├── train.csv
│   ├── test.csv
├── Documentation/
│   ├── (Submission requirements)
├── notebooks/
│   ├── data/
│   │   ├── cleaned.csv
│   │   ├── data.csv
│   ├── notebook.ipynb
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   ├── pipelines/
│   │   ├── training_pipeline.py
│   │   ├── prediction_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   ├── utils.py
├── static/
│   ├── (images, videos used in frontend)
├── templates/
│   ├── form.html
│   ├── index.html
│   ├── insights.html
├── app.py
├── requirements.txt
├── setup.py
├── README.md
```

---

## How It Works

1. **Home Page:** Displays background image/video with basic introduction.
2. **Form Page:** User inputs weather and time details to predict traffic volume.
3. **Prediction:** Data is processed and passed to a trained model to display predicted traffic volume.
4. **Insights Page:** Displays static plots based on existing traffic data trends.

---

## Machine Learning Models Used

- Decision Tree Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor

The best-performing model is selected automatically based on the R² score.

---

## Author

Fazal Rehman