import streamlit as st
import os
import base64

def main():
    st.markdown("<h1 style='text-align: center; font-weight: bold;'>SmartEdu VQA HUB</h1>", unsafe_allow_html=True)

    # Subtext with enhanced visibility
    st.write("""
    <div style='color: #000000'; font-weight: bold;'>
        <p><strong>Welcome to SmartEdu VQA Hub!</strong></p>
        <p><strong>For Evaluators:</strong><br>
        Sign in to craft engaging questions, monitor student progress, and gain valuable insights into their learning journey. Your role is pivotal in shaping an enriching educational experience.</p>
        <p><strong>For Students:</strong><br>
        Log in to immerse yourself in interactive learning with our Smart Tutor, explore a range of educational resources, interactive learning and take on exciting exams to test your knowledge. Your adventure in personalized education begins here!</strong></p>
        <p style="color:#FF7900"><strong>Explore, learn, and achieve with SmartEdu VQA Hub. We’re here to support your growth every step of the way!</p>
        <p style="color:#FF7900"><strong>Login or Sign in now to get started!</strong></p>
    </div>
    """, unsafe_allow_html=True)
    page = st.selectbox(
        "Login or Signup as:",
        ("Evaluator", "Student"),        key="login_signup",
    )
    if st.button("Login"):
        if page == "Evaluator":
            os.system('python -m streamlit run evaluator.py')
        else:
            os.system('python -m streamlit run student.py')

if __name__ == "__main__":

    main()
        # Path to the image in the assets folder (corrected for Windows paths)
    image_path = r"assets/Bot .png"

        # Read the image and encode it in base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

        # Use the encoded image in the CSS
    st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("data:image/png;base64,{encoded_image}");
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
    )


