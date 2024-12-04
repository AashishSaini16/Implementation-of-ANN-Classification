# Customer Churn Prediction Using Artificial Neural Networks

This repository contains a project that predicts customer churn using an Artificial Neural Network (ANN) model. The project includes a Jupyter notebook for training the model and a Streamlit app for deploying the trained model as a web-based prediction tool.

## Project Overview:
Customer churn prediction helps businesses identify customers likely to leave their services. This project builds a machine learning pipeline to preprocess customer data, train an ANN model, and serve predictions via an interactive web application.

## Repository Contents:
1. **`experiments.ipynb`**: Contains the data preprocessing, model building, training, and evaluation processes.
2. **`app.py`**: A Streamlit application for user-friendly deployment of the trained model.
3. **Model Artifacts**:
   - `model.h5`: The trained ANN model.
   - `scaler.pkl`: A scaler for normalizing input data.
   - `label_encoder_gender.pkl` and `onehot_encoder_geo.pkl`: Encoders for categorical variables.

---

## Dataset Source:
The dataset consists of customer demographic and account details, including:
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` (target variable indicating churn)

Ensure the dataset is available in the directory for training or testing the model.

---

## Key Features:
### Data Preprocessing:
- Encoded categorical features (`Geography`, `Gender`) using `LabelEncoder` and `OneHotEncoder`.
- Standardized numerical features using `StandardScaler` for ANN compatibility.
- Split the dataset into training and testing subsets.

### Model Development:
- Implemented a neural network using TensorFlow/Keras:
  - Input layer with features.
  - Two hidden layers with ReLU activation.
  - Output layer with a sigmoid activation function for binary classification.
- Regularization techniques like early stopping were used to prevent overfitting.

## Technologies Used:
- **Python**: Programming language for implementation.
- **TensorFlow/Keras**: Neural network model development.
- **Streamlit**: Web application framework for deployment.
- **Scikit-learn**: For data preprocessing.
- **Pandas**: Data manipulation and preparation.
- **Pickle**: Model and artifact storage.

---

### Deployment:
- **Streamlit Application**:
  - Interactive inputs for customer details such as `Geography`, `Gender`, `Age`, etc.
  - Real-time prediction of churn probability.
  - Outputs a customer churn likelihood with recommendations.
![Output](https://github.com/AashishSaini16/Implementation-of-ANN-Classification/blob/main/app.JPG)
---
