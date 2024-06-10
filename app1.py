import streamlit as st

# Placeholder function to simulate database connection and fetching recommendations
def get_diet_recommendations(symptoms):
    recommendations = {
        'sugar_craving': "Increase fiber intake and focus on whole foods.",
        'acne': "Reduce intake of high-glycemic foods and incorporate foods with anti-inflammatory properties.",
        'allergies': "Consult a dietitian to identify safe foods and alternatives.",
        'menstrual_irregularity': "Maintain a balanced diet rich in omega-3 and vitamin B.",
        'hirsutism': "Incorporate foods that naturally balance hormones, such as leafy greens.",
        'infertility': "Add foods rich in antioxidants and omega-3 fatty acids to support reproductive health.",
        'weight_management': "Focus on a diet high in protein and low in processed foods."
    }
    return [recommendations.get(symptom, "No specific recommendations.") for symptom in symptoms]

def get_meal_recommendations(has_pcos):
    if has_pcos:
        meals = {
            'breakfast': ['Chia seed pudding with berries', 'Protein smoothie', 'Egg muffins with vegetables'],
            'lunch': ['Salad with mixed greens and grilled chicken', 'Quinoa bowl with veggies', 'Lentil soup with mixed vegetables'],
            'dinner': ['Grilled salmon with asparagus', 'Stuffed bell peppers with turkey', 'Zucchini noodles with pesto']
        }
    else:
        meals = {
            'breakfast': ['Oatmeal with berries', 'Greek yogurt with nuts', 'Smoothie with spinach and banana'],
            'lunch': ['Quinoa salad with veggies', 'Grilled chicken with mixed greens', 'Lentil soup with whole grain bread'],
            'dinner': ['Baked salmon with broccoli', 'Stir-fried tofu with veggies', 'Beef stew with carrots']
        }
    return meals

def calculate_bmi(height, weight):
    return weight / ((height / 100) ** 2)

def interpret_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obesity"

def predict_pcos(symptoms):
    pcos_indicators = {'menstrual_irregularity', 'infertility', 'acne', 'hirsutism'}
    return bool(pcos_indicators.intersection(symptoms))

def app():
    st.markdown('<h1 style="color: #5f9ea0;">PCOS Diet Recommendation System</h1>', unsafe_allow_html=True)
    #st.title('PCOS Diet Recommendation System')

    with st.form("user_info_form"):
        st.write("Enter your details:")
        height = st.number_input("Height (in cm)", min_value=100, max_value=250)
        weight = st.number_input("Weight (in kg)", min_value=30, max_value=200)
        submit_info = st.form_submit_button("Submit")

    if submit_info:
        bmi = calculate_bmi(height, weight)
        bmi_category = interpret_bmi(bmi)
        st.subheader(f"Your BMI is: {bmi:.2f} ({bmi_category})")

    symptoms_list = ['sugar_craving', 'acne', 'allergies', 'menstrual_irregularity', 'hirsutism', 'infertility', 'weight_management']
    symptoms_names = ['Frequent sugar cravings', 'Acne or Oily skin', 'Food allergies/intolerances', 'Menstrual irregularity', 'Unwanted hair growth', 'Infertility', 'Difficulties with weight management']

    st.markdown('<h1 style="color: #5f9ea0;">Please select your symptoms:</h1>', unsafe_allow_html=True)
    #st.write("## Please select your symptoms:")
    selected_symptoms = []
    for symptom, name in zip(symptoms_list, symptoms_names):
        if st.checkbox(name):
            selected_symptoms.append(symptom)

    if selected_symptoms:
        has_pcos = predict_pcos(selected_symptoms)
        st.write("Based on your symptoms, there is a likelihood that you may have PCOS." if has_pcos else "Based on your symptoms, it's unlikely that you have PCOS.")

        recommendations = get_diet_recommendations(selected_symptoms)
        st.subheader("Diet Recommendations:")
        for recommendation in recommendations:
            st.info(recommendation)

        meals = get_meal_recommendations(has_pcos)
        st.subheader("Suggested Meals:")
        for meal_time, meal_options in meals.items():
            st.markdown(f"**{meal_time.capitalize()}**")
            for option in meal_options:
                st.markdown(f"- {option}")

if __name__ == "__main__":
    app()
