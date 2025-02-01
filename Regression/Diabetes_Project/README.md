# Diabetes Prediction

## Project Overview
This project predicts whether a person has diabetes based on input health parameters. It leverages multiple regression models, evaluates them using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score, and selects the best-performing model. A Streamlit application is built to provide an interactive user experience.

![Image](https://github.com/user-attachments/assets/72efc093-7b04-4170-b581-17ce31038bd4)

## Features
- **Machine Learning Models**: Various regression models evaluated based on MSE, MAE, and R² score.
- **Model Selection**: Best-performing model is chosen based on evaluation metrics.
- **Streamlit App**: Interactive web application for users to input health data and get predictions.

## Technologies Used
- **Python**
- **Machine Learning (Regression Models)**
- **Pandas, NumPy**
- **Matplotlib, Seaborn** (for data visualization)
- **Streamlit** (for the interactive web app)
- **Scikit-learn** (for model training and evaluation)

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/chaitanya-03c/Machine-Learning/Regression/Diabetes_Project.git
   cd diabetes-prediction
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## How It Works
1. Users input health-related parameters such as glucose level, BMI, age, etc.
2. The best-selected model processes the input and predicts whether the person has diabetes.
3. Evaluation metrics (MSE, MAE, R² score) are used to choose the best regression model.
4. The Streamlit app provides an easy-to-use interface for predictions.

## Dataset
The dataset consists of health-related data points, including glucose levels, BMI, age, and other factors that influence diabetes prediction.

## Future Enhancements
- Improve model accuracy with advanced techniques.
- Include additional health parameters for better prediction.
- Integrate real-time health data sources.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License.

---
Stay Healthy! 🏥📊

