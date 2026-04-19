#  Airline Passenger Satisfaction Prediction (ML + Streamlit)

##  Overview
This project builds a machine learning model to predict airline passenger satisfaction and deploys it using a Streamlit web application.

Users can input passenger details and get real-time predictions.

---

##  Dataset
- 129,000+ passenger records
- Features include:
  - Customer details
  - Flight information
  - Service ratings
  - Delay metrics

---

##  Objectives
- Predict passenger satisfaction
- Identify key influencing factors
- Deploy model as an interactive web app

---

##  Data Preprocessing
- Handled missing values using mode
- Encoded categorical variables
- Feature scaling using MinMaxScaler

---

##  Models Used
- KNN
- GaussianNB
- Decision Tree
- Random Forest  (Best)
- Gradient Boost
- AdaBoost

---

##  Model Performance
- Random Forest achieved **96% accuracy**
- Best balance of precision and recall

---

###  Features
- User-friendly input interface  
- Real-time prediction  
- Instant results display  

---

##  Visualizations

###  Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

###  Model Comparison
![Model Comparison](images/model_comparison.png)

###  Streamlit App Preview
![Streamlit App](images/streamlit_app.png)

---

##  Handling Imbalanced Data
- Applied SMOTE for balancing the dataset  

---

##  Key Insights
- Online boarding strongly affects satisfaction  
- Service quality has more impact than delays  
- Random Forest performs best among all models  

---

##  Tech Stack
- Python (Pandas, NumPy)  
- Scikit-learn  
- Streamlit  
- Matplotlib, Seaborn  

---
##  Future Improvements
- Deploy app on Streamlit Cloud  
- Add feature importance visualization  
- Improve UI/UX design  

---

##  Author
**M S Aravind**
