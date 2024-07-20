import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load the model and scaler
model = joblib.load('C:\BootcampKec\medical_cost_prediction\ml_model\knn_model.pkl')
scaler = joblib.load('C:\BootcampKec\medical_cost_prediction\ml_model\scaler2.pkl')

# Function to create a pie chart
def plot_pie_chart(data, labels, title):
    fig = px.pie(data, values='Cost', names=labels, title=title)
    st.plotly_chart(fig, use_container_width=True)

# Function to create a bar chart by age
def plot_bar_chart_by_age(data, x, y, title):
    fig = px.bar(data, x=x, y=y, title=title)
    st.plotly_chart(fig, use_container_width=True)

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Medical Cost Prediction",
        page_icon=":hospital:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Medical Cost Prediction")
    
    # Create columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # User input for model features
        name = st.text_input("Name")
        age = st.slider("Age", 18, 100, 30)
        sex = st.selectbox("Sex", ("male", "female"))
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
        children = st.slider("Number of Children", 0, 10, 0)
        smoker = st.selectbox("Smoker", ("yes", "no"))
        region = st.selectbox("Region", ("northwest", "southeast", "southwest"))
        
        # Convert smoker to binary
        smoker_binary = 1 if smoker == 'yes' else 0
        
    with col2:
        # Predict button on the right side
        st.text("")  # Adjust spacing if necessary
        if st.button("Predict", key="predict_button"):
            # Prepare the input data
            input_data = {
                'age': age,
                'bmi': bmi,
                'children': children,
                'sex_male': 1 if sex == 'male' else 0,
                'smoker_yes': smoker_binary,
                'region_northwest': 1 if region == 'northwest' else 0,
                'region_southeast': 1 if region == 'southeast' else 0,
                'region_southwest': 1 if region == 'southwest' else 0
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Ensure the columns are in the same order as the training data
            expected_columns = ['age', 'bmi', 'children', 'sex_male',
                                'smoker_yes', 'region_northwest',
                                'region_southeast', 'region_southwest']
            input_df = input_df[expected_columns]
            
            # Standardize the input data using the loaded scaler
            input_scaled = scaler.transform(input_df)
            
            # Predict medical cost
            prediction = model.predict(input_scaled)[0]
            
            st.subheader(f"{name}, your predicted medical cost is: ${prediction:.2f}")
            
            # Example: Display remarks if the cost is higher than average
            average_cost = 13270  # Replace with your actual average cost value
            
            if prediction > average_cost:
                st.error("Your predicted cost is higher than average. Consider reviewing your options.")
            else:
                st.success("Your predicted cost is within the average range.")
    
    # Move to the next row for displaying charts below the Predict button
    st.write("")
    st.write("")
    
    # Example: Pie chart to show distribution by region
    data_pie = {
        "Region": ["Northwest", "Southeast", "Southwest"],
        "Cost": [12000, 15000, 10000]  # Replace with actual data
    }
    st.subheader("Distribution of Costs by Region")
    plot_pie_chart(data_pie, data_pie["Region"], "Distribution of Costs by Region")
    
    # Example: Bar chart to show distribution by age
    data_bar_age = {
        "Age": ["18-30", "31-50", "51-70"],  # Example age groups
        "Cost": [10000, 12000, 11000]  # Replace with actual data
    }
    st.subheader("Medical Costs by Age Group")
    plot_bar_chart_by_age(data_bar_age, x="Age", y="Cost", title="Medical Costs by Age Group")

if __name__ == "__main__":
    main()
