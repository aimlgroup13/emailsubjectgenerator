import streamlit as st
import requests
import re

st.title("Automated Question-Answering")
model_options = {
    "T5": "nagthgr8/qa-t5-base",
    "GPT2": "nagthgr8/gpt2-qa",
    "QWEN": "sbtraining2020/qwen_qa1"
}

model = st.selectbox("Select a model", list(model_options.keys()))
question = st.text_input("Enter Question")

if st.button("Generate"):
    response = requests.post(
        "http://172.26.61.20:8000/predict/",
        json={"question": question, "model_name": model_options[model]}
    )

    if response.status_code == 200:
        answer = response.json().get("predicted_answer")
        # Display the subject in a readonly textbox
        if model_options[model] == "nagthgr8/gpt2-qa":
            answer = answer.partition("Answer: ")[2]
        elif model_options[model] == "sbtraining2020/qwen_qa1":
           # Extract response using regular expression
           pattern = r'### Response:\n(.*?)(<|endoftext|>)'
           match = re.search(pattern, answer[0])
           if match:
               answer = match.group(1).strip()
           else:
               answer = "No response found."
        st.text_area("Answer", answer, height=200, disabled=True)
    else:
        st.error("Failed to Answer. Please try again.")

