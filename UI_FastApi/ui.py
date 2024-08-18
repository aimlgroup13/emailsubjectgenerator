import streamlit as st
import requests

st.title("Email Subject Generator")

model = st.selectbox("Select a model", ["Bart", "Mistral", "Llama"])
email_body = st.text_area("Enter the email body", height=300)

if st.button("Generate"):
    response = requests.post(
        "http://127.0.0.1:8000/generate_subject/",
        json={"email_body": email_body, "model": model}
    )

    if response.status_code == 200:
        subject = response.json().get("subject")
        # Display the subject in a readonly textbox
        st.text_area("Generated Subject", subject, height=200, disabled=True)
    else:
        st.error("Failed to generate subject. Please try again.")
