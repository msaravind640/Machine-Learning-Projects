import streamlit as st
import pickle
from PIL import Image
import time

def airline():
    st.title("AIRLINE PASSENGER SATISFACTION")
    image = Image.open('image.png')
    st.image(image,width=670)
    model = pickle.load(open('airline.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))

    tab1, tab2 ,tab3, tab4= st.tabs(["**INFO**", "**TRAVEL DETAILS**", "**RATINGS**","**CONCLUSION**"])

    with tab1:
        st.header("About")
        st.write('The Airline Passenger Satisfaction dataset is a comprehensive collection of customer feedback from passengers. The dataset contains information on various aspects of the passengers’ travel experience, such as flight distance, gender, age, type of travel, class, seat comfort, inflight entertainment, onboard service, cleanliness, departure delay, arrival delay, and overall satisfaction. This dataset aims to provide insights into the factors that contribute to passenger satisfaction and dissatisfaction, which can be used by airlines to improve their services and enhance their customers’ travel experience')
        st.header("Links")
        st.page_link("https://colab.research.google.com/drive/1AQfL865U60DS_hpI6ZQ4FgcMN-yxkWn_", label="Colab Notebook")
        st.page_link("https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction", label="Dataset")
        st.divider()

        roc=st.selectbox("**ROC Curve and Correlation Heatmap**",["ROC Curve","Correlation Heatmap"],placeholder="Select the model",index=None)
        if roc:
            if roc=="ROC Curve":
                st.image('roc_curve.png')
            else:
                st.image('heatmap.png')
        st.divider()

        graph = st.selectbox("**Model Analysis Graphs**", ["Model vs Accuracy", "Model vs Metrics"], placeholder="Select the model",index=None)
        if graph:
            if graph == "Model vs Accuracy":
                st.image('modelaccuracy.png')
            else:
                st.image('modelvsmetrics.png')

        st.header('Observations')
        st.write('After performing machine learning models such as KNN, GaussianNB, Decision Tree, Random Forest etc gives high performane value. By implimenting feature selection, over sampling, under sampling and hyper parameter tuning the dataset become significantly optimized.')
        st.write('Random Forest Classifier gives performance model.')
    with tab2:
        options = ['Female', 'Male']
        values = {'Female': 0, 'Male': 1}
        selected_option = st.radio("GENDER", options,horizontal=True)
        gender = values[selected_option]
        st.divider()
        age = st.number_input("AGE",max_value=100,min_value=1,placeholder="Enter your age...")
        st.divider()
        options = ['First-Time', 'Returning']
        values = {'First-Time': 0, 'Returning': 1}
        selected_option = st.selectbox("CUSTOMER TYPE", options)
        customer_type = values[selected_option]
        st.divider()
        options = ['Business', 'Personal']
        values = {'Business': 0, 'Personal': 1}
        selected_option = st.selectbox("Type of Travel", options)
        travel_type = values[selected_option]
        st.divider()
        options = ['Business', 'Economy', 'Economy Plus']
        values = {'Business': 0, 'Economy':1, 'Economy Plus': 2}
        selected_option = st.selectbox("CABIN CLASS", options,placeholder="Select Cabin Class",)
        cabin_class = values[selected_option]
        st.divider()
        distance=st.slider('FLIGHT DISTANCE',max_value=10000,min_value=0)

    with tab3:
        col1,col2=st.columns(2)
        options = ['1', '2', '3', '4', '5']
        values = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}

        with col1:

            selected_option=st.radio('**ONLINE BOOKING**', options,horizontal=True)
            on_book=values[selected_option]
            st.divider()
            selected_option=st.radio('**CHECK IN SERVICE**', options,horizontal=True)
            check_in=values[selected_option]
            st.divider()
            selected_option=st.radio('**ONLINE BOARDING**', options,horizontal=True)
            board=values[selected_option]
            st.divider()
            selected_option=st.radio('**ON-BOARD SERVICE**', options,horizontal=True)
            on_board=values[selected_option]
            st.divider()
            selected_option=st.radio('**SEAT COMFORT**', options, horizontal=True)
            seat=values[selected_option]
            st.divider()
            selected_option=st.radio('**LEG ROOM SERVICE**', options, horizontal=True)
            leg_Room=values[selected_option]
            st.divider()
        with col2:

            selected_option=st.radio('**CLEANLINESS**', options, horizontal=True)
            clean=values[selected_option]
            st.divider()
            selected_option=st.radio('**FOOD AND DRINK**', options, horizontal=True)
            food=values[selected_option]
            st.divider()
            selected_option=st.radio('**IN-FLIGHT SERVICE**', options, horizontal=True)
            flight=values[selected_option]
            st.divider()
            selected_option=st.radio('**IN-FLIGHT WIFI SERVICE**', options, horizontal=True)
            wifi=values[selected_option]
            st.divider()
            selected_option=st.radio('**IN-FLIGHT ENTERTAINMENT**',options, horizontal=True)
            entertain=values[selected_option]
            st.divider()
            selected_option=st.radio('**BAGGAGE HANDLING**', options, horizontal=True)
            baggage=values[selected_option]
            st.divider()
        predict=st.button('**PREDICT**')

        if predict:
            with st.spinner('Please wait ...'):
                time.sleep(5)
            satisfaction=model.predict(scaler.transform([[gender,age,customer_type,travel_type,cabin_class,distance,on_book,check_in,board,on_board,seat,leg_Room,clean,food,flight,wifi,entertain,baggage]]))

            if satisfaction==0:
                st.success('Passenger is not Satisfied')
            else:
                st.success('Passenger is Satisfied')

    with tab4:
        st.header("Conclusion")
        st.write('After performing feature selection, oversampling, and hyperparameter tuning, the models were significantly optimized for predicting customer satisfaction. Key features influencing satisfaction were identified and prioritized, enhancing model efficiency and interpretability. Balancing the dataset improved the models ability to generalize across satisfaction classes, mitigating bias towards majority classes. Overall, these steps collectively strengthened the models predictive capabilities and reliability for practical applications in customer satisfaction prediction.')
airline()