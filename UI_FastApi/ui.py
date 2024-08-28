import streamlit as st
import requests

def extract_response(text):
    # Check if text is a list and convert it to a single string if needed
    if isinstance(text, list):
        text = ' '.join(text)
    # Define markers to be removed
    markers_to_remove = ["<s>", "</s>", "<|begin_of_text|>", "<|end_of_text|>"]
    # Remove all markers from the text
    for marker in markers_to_remove:
        text = text.replace(marker, "")
    # Define the start marker for response extraction
    start_marker = "### Response:"
    # Find the start index
    start_index = text.find(start_marker) + len(start_marker)
    # Check if start_marker is found
    if start_index == len(start_marker) - 1:
        return "Start marker not found"
    # Extract the response from the start index
    response = text[start_index:].strip()
    return response

st.title("Email Subject Generator")
model_options = {
    "Bart": "sbtraining2020/email_bart_1",
    "Mistral": "nagthgr8/subject-mistrel",
    "Llama": "sbtraining2020/esubjectgen_llama31_clean"
}

model = st.selectbox("Select a model", list(model_options.keys()))
email_body = st.text_area("Enter the email body", height=300)

if st.button("Generate"):
    response = requests.post(
        "http://172.26.61.20:8000/predict/",
        json={"email_body": email_body, "model_name": model_options[model]}
    )

    if response.status_code == 200:
        subject = response.json().get("predicted_subject")
        # Display the subject in a readonly textbox
        st.text_area("Generated Subject", extract_response(subject), height=200, disabled=True)
    else:
        st.error("Failed to generate subject. Please try again.")

