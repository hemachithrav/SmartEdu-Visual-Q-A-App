import json
import os
import streamlit as st
from streamlit import session_state
import streamlit as st
import datetime
import random
import string
import re
import smtplib
import json
import hashlib
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
load_dotenv()
session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0
    
    
st.set_page_config(
    page_title="Smart Examination Portal",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

GOOGLE_API_KEY_1 = os.getenv("GOOGLE_API_KEY_1")
GOOGLE_API_KEY_2 = os.getenv("GOOGLE_API_KEY_2")
GOOGLE_API_KEY_3 = os.getenv("GOOGLE_API_KEY_3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_PASSWORD = os.getenv("APP_PASSWORD")
SENDER_MAIL_ID = os.getenv("SENDER_MAIL_ID")


def user_exists(email, json_file_path):
    # Function to check if user with the given email exists
    with open(json_file_path, "r") as file:
        users = json.load(file)
        for user in users["users"]:
            if user["email"] == email:
                return True
    return False
def display_exam_details(st, exam):
    # Display exam details
    st.markdown(f"### Subject: {exam.get('Subject_name')}")
    st.write(f"Subject Code: {exam.get('Subject_code')}")
    st.write(f"Number of Questions: {exam.get('Number_of_questions')}")


def display_question_details(st, exam):
    # Iterate through each question in the exam and display its details
    for question_index, question in enumerate(exam.get("questions", []), start=1):
        st.write(f"Question {question_index}: {question.get('question')}")
        st.write(f"Answer: {question.get('answer')}")
        st.write(f"Evaluation: {question.get('evaluation')}")
        st.write(f"Marks: {question.get('marks')}")
        st.write('\n')


def calculate_marks(exam):
    # Calculate total marks and obtained marks for the exam
    total_marks = 0
    obtained_marks = 0
    for question in exam.get("questions", []):
        total_marks += int(question.get('total_marks'))
        obtained_marks += int(question.get('marks'))
    return total_marks, obtained_marks

def send_verification_code(email, code):
    RECEIVER = email
    server = smtplib.SMTP_SSL("smtp.googlemail.com", 465)
    server.login(SENDER_MAIL_ID, APP_PASSWORD)
    message = f"Subject: Your Verification Code\n\nYour verification code is: {code}"
    server.sendmail(SENDER_MAIL_ID, RECEIVER, message)
    server.quit()
    st.success("Email sent successfully!")
    return True


def generate_verification_code(length=6):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def signup(json_file_path="evaluator.json"):
    st.title("Evaluator Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")
        if (
            session_state.get("verification_code_eval") is None
            or session_state.get("verification_time_eval") is None
            or datetime.datetime.now() - session_state.get("verification_time_eval")
            > datetime.timedelta(minutes=5)
        ):
            verification_code = generate_verification_code()
            session_state["verification_code_eval"] = verification_code
            session_state["verification_time_eval"] = datetime.datetime.now()
        if st.form_submit_button("Signup"):
            if not name:
                st.error("Name field cannot be empty.")
            elif not email:
                st.error("Email field cannot be empty.")
            elif not re.match(r"^[\w\.-]+@[\w\.-]+$", email):
                st.error("Invalid email format. Please enter a valid email address.")
            elif user_exists(email, json_file_path):
                st.error(
                    "User with this email already exists. Please choose a different email."
                )
            elif not age:
                st.error("Age field cannot be empty.")
            elif not password or len(password) < 6:  # Minimum password length of 6
                st.error("Password must be at least 6 characters long.")
            elif password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            else:
                verification_code = session_state["verification_code_eval"]
                send_verification_code(email, verification_code)
                entered_code = st.text_input(
                    "Enter the verification code sent to your email:"
                )
                if entered_code == verification_code:
                    user = create_account(
                        name, email, age, sex, password, json_file_path
                    )
                    session_state["logged_in"] = True
                    session_state["user_info"] = user
                    st.success("Signup successful. You are now logged in!")
                elif len(entered_code) == 6 and entered_code != verification_code:
                    st.error("Incorrect verification code. Please try again.")


def check_login(username, password, json_file_path="evaluator.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                return user
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None


def initialize_database(
    json_file_path="evaluator.json", question_paper="question_paper.json", student = "students.json"
):
    try:
        if not os.path.exists(json_file_path):
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
        if not os.path.exists(question_paper):
            data = {"subjects": []}
            with open(question_paper, "w") as json_file:
                json.dump(data, json_file)
        if not os.path.exists(student):
            data = {"students": []}
            with open(student, "w") as json_file:
                json.dump(data, json_file)
        
    except Exception as e:
        print(f"Error initializing database: {e}")


def create_account(name, email, age, sex, password, json_file_path="evaluator.json"):
    try:
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        email = email.lower()
        password = hashlib.md5(password.encode()).hexdigest()
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
        }

        data["users"].append(user_info)

        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None


def login(json_file_path="evaluator.json", question_paper="question_paper.json"):
    st.title("Login Page")
    username = st.text_input("Email:")
    password = st.text_input("Password:", type="password")

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")


def get_user_info(email, json_file_path="evaluator.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None


def render_dashboard(user_info, json_file_path="evaluator.json"):
    try:
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        st.subheader("Evaluator Information")
        st.write(f"Name: {user_info['name']}")
        st.write(f"Sex: {user_info['sex']}")
        st.write(f"Age: {user_info['age']}")
    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")


def main(json_file_path="evaluator.json", question_paper="question_paper.json"):
    st.title("Evaluator Dashboard")
    page = st.sidebar.selectbox(
        "Go to",
        (
            "Signup/Login",
            "Dashboard",
            "Set Question Paper",
            "View Question Papers",
            "View Student Responses",
            "Logout"
            # "Class Analytics",
        ),
        key="page",
    )

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")

    elif page == "Set Question Paper":
        if session_state.get("logged_in"):

            st.title("Set Question Paper")
            user_info = session_state["user_info"]
            Subject_name = st.text_input("Subject Name:")
            Subject_code = st.text_input("Subject Code:")
            Number_of_questions = st.number_input("Number of Questions:", min_value=0)
            with st.form("question_paper_form"):
                st.write("Fill in the details below to create a question paper:")
                questions = []
                for i in range(Number_of_questions):
                    question = st.text_area(f"Question {i+1}:")
                    answer = st.text_area(f"Answer for Question {i+1}:")
                    marks = st.number_input(f"Marks for Question {i+1}:", min_value=0)
                    evaluation_criteria = st.text_area(f"Evaluation Criteria for Question {i+1}:")
                    questions.append((question, answer, marks, evaluation_criteria))
                if st.form_submit_button("Create Question Paper"):
                    question_paper = {
                        "Subject_name": Subject_name,
                        "Subject_code": Subject_code,
                        "Number_of_questions": Number_of_questions,
                        "questions": [],
                    }
                    for i in range(Number_of_questions):
                        question, answer, marks,evaluation_criteria = questions[i]
                        question_paper["questions"].append(
                            {"question": question, "answer": answer, "marks": marks, "evaluation": evaluation_criteria}
                        )
                    with open("question_paper.json", "r+") as json_file:
                        data = json.load(json_file)
                        subject_code_ = next(
                            (
                                i
                                for i, subject in enumerate(data["subjects"])
                                if subject["Subject_code"] == Subject_code
                            ),
                            None,
                        )
                        if subject_code_ is None:
                            data["subjects"].append(question_paper)
                            json_file.seek(0)
                            json.dump(data, json_file, indent=4)
                            json_file.truncate()
                            st.success("Question paper created successfully!")
                        else:
                            st.error("Subject code already exists. Please try again.")
            if st.button("Clear all questions"):
                with open("question_paper.json", "r+") as json_file:
                    data = json.load(json_file)
                    data["subjects"] = []
                    json_file.seek(0)
                    json.dump(data, json_file, indent=4)
                    json_file.truncate()
                    st.success("All questions cleared successfully!")
        else:
            st.warning("Please login/signup to access this page.")
    elif page == "View Question Papers":
        if session_state.get("logged_in"):
            st.title("View Question Papers")
            with open("question_paper.json", "r") as json_file:
                data = json.load(json_file)
            if len(data["subjects"]) == 0:
                st.warning("No question papers found.")
                return
            for i in range(len(data["subjects"])):
                st.write(f"Subject {i+1}:")
                st.write(f"Subject: {data['subjects'][i]['Subject_name']}")
                st.write(f"Subject Code: {data['subjects'][i]['Subject_code']}")
                st.write(f"Number of Questions: {data['subjects'][i]['Number_of_questions']}")
                for j in range(data["subjects"][i]["Number_of_questions"]):
                    st.markdown(f"### Question {j+1}:")
                    st.write(f"Question: {data['subjects'][i]['questions'][j]['question']}")
                    st.write(f"Answer: {data['subjects'][i]['questions'][j]['answer']}")
                    st.write(f"Marks: {data['subjects'][i]['questions'][j]['marks']}")
                    st.write(f"Evaluation Criteria: {data['subjects'][i]['questions'][j]['evaluation']}")
                    st.write("---")
                    st.write('\n')

                st.write('---')
        else:
            st.warning("Please login/signup to access this page.")
    elif page == "View Student Responses":
        if session_state.get("logged_in"):
            # Display the title
            st.title("View Student Responses")

            # Load student responses from JSON file
            try:
                with open("students.json", "r") as json_file:
                    data = json.load(json_file)
            except FileNotFoundError:
                st.error("Error: Unable to load student responses.")
                return

            # Check if there are any student responses
            students = data.get("students", [])
            if len(students) == 0:
                st.warning("No student responses found.")
            else:
                # Iterate through each student
                response_exists = False
                for student_index, student in enumerate(students, start=1):
                    # Skip students with no exam responses
                    if student.get("exams") is None:
                        continue
                    response_exists = True
                    # Display student details
                    st.markdown(f"## Name: {student.get('name')}:")
                    st.write(f"Email: {student.get('email')}")

                    # Iterate through each exam for the student
                    for exam in student.get("exams", []):
                        display_exam_details(st, exam)  # Display exam details
                        display_question_details(st, exam)  # Display question details

                        # Calculate and display marks obtained
                        total_marks, obtained_marks = calculate_marks(exam)
                        percentage_score = float(float(obtained_marks)/float(total_marks))
                        percentage_remainder = 1 - percentage_score

                        # Create a Plotly figure for the pie chart
                        fig = go.Figure(
                        data=[
                            go.Pie(
                                labels=["Obtained", "Missed"],
                                values=[percentage_score, percentage_remainder],
                                hole=0.3,
                                marker_colors=["rgba(0, 128, 0, 0.7)", "rgba(255, 0, 0, 0.7)"],
                            )
                        ]
                    )
                        fig.update_layout(title_text="Marks Obtained", title_x=0.5)

                        # Display the chart
                        st.plotly_chart(fig)
                        st.markdown(f"### Marks Obtained: {obtained_marks}/{total_marks}")
                        st.write('---')
                        st.write('\n')
                if not response_exists:
                    st.warning("No student responses found.")
        else:
            st.warning("Please login/signup to access this page.")
            
    elif page == "Logout":
        if st.button("Logout"):
            session_state["logged_in"] = False
            session_state["user_info"] = None
            st.success("You have been logged out successfully!")

if __name__ == "__main__":

    initialize_database()
    main()