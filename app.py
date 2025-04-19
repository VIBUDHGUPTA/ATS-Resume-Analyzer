from dotenv import load_dotenv

load_dotenv()
# import torch
# from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import base64
import streamlit as st
import os
import io
from PIL import Image 
import pdf2image
import google.generativeai as genai



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_compatibility(job_description:str, resume:str) -> float:
    
    
    if isinstance(job_description, list):
        job_description = ' '.join(str(item) if not isinstance(item, dict) else ' '.join(str(v) for v in item.values()) for item in job_description)
    
    if isinstance(resume, list):
        resume = ' '.join(str(item) if not isinstance(item, dict) else ' '.join(str(v) for v in item.values()) for item in resume)  # Join list into a single string

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([job_description, resume])
    
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    
    compatibility_percentage = round(similarity_score * 100, 2)
    
    return compatibility_percentage


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input,pdf_content,prompt):
    model=genai.GenerativeModel('gemini-1.5-flash')
    response=model.generate_content([input,pdf_content[0],prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        images=pdf2image.convert_from_bytes(uploaded_file.read())

        first_page=images[0]

        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")


st.set_page_config(page_title="Resume EXPERT")
st.header("Resume Analyzer and screener")
input_text=st.text_area("Job Description: ",key="input")
uploaded_file=st.file_uploader("Upload your resume(PDF)...",type=["pdf"])


if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")


submit1 = st.button("Tell Me About the Resume")

submit2 = st.button("How Can I Improvise my Skills")

submit3 = st.button("Percentage match")

submit4 = st.button("Some Interview Tips?")

input_prompt1 = """
 You are an experienced Technical Human Resource Manager,your task is to review the provided resume against the job description. 
  Please share your professional evaluation on whether the candidate's profile aligns with the role. 
 Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""
input_prompt2 = """
You are experienced recommendation system which provide youtube vedio links(always give link) , your task is to give the youtube vedio that you suggest the candidate 
should watch which help candidate to learn missing skills for the given resume give output in form of clickable links of the youtube 
vedios (provide link always and can you a bit formal in answering you are profeesional and give response to second person)        
"""
input_prompt3 = """
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
the job description. First the output should come as percentage and then keywords missing and last final thoughts.
"""
input_prompt4 = """
You are professional recommendation model your task is provide  youtube vedio link of interview tips for the role which is decsribed
in job discription (and you have to provide link no matter what) given to you and suggest some most frequently asked question in 
interviews
you can use these link paste any one of them(alternatively always) in youtube link
interview_videos = ['https://youtu.be/Ji46s5BHdr0','https://youtu.be/seVxXHi2YMs',
                    'https://youtu.be/9FgfsLa_SmY','https://youtu.be/2HQmjLu-6RQ',
                    'https://youtu.be/DQd_AlIvHUw','https://youtu.be/oVVdezJ0e7w'
                    'https://youtu.be/JZK1MZwUyUU','https://youtu.be/CyXLhHQS3KY']
(be formal just give content no extra information dont ask me to search something do on your own)
"""

if submit1:
    if uploaded_file is not None:
        pdf_content=input_pdf_setup(uploaded_file)
        response=get_gemini_response(input_prompt1,pdf_content,input_text)
        st.subheader("The Repsonse is")
        st.write(response)
    else:
        st.write("Please uplaod the resume")
        
elif submit2 :
    if uploaded_file is not None:
        pdf_content=input_pdf_setup(uploaded_file)
        response=get_gemini_response(input_prompt2,pdf_content,input_text)
        st.subheader("The Repsonse is")
        st.write(response)
    else:
        st.write("Please uplaod the resume")

elif submit3:
    if uploaded_file is not None:
        pdf_content=input_pdf_setup(uploaded_file)
        
        response=get_gemini_response(input_prompt3,pdf_content,input_text)
        # response = compute_compatibility(input_text, pdf_content)
        st.subheader("The Repsonse is")
        st.write(response)
    else:
        st.write("Please uplaod the resume")

elif submit4:
    if uploaded_file is not None:
        pdf_content=input_pdf_setup(uploaded_file)
        
        response=get_gemini_response(input_prompt4,pdf_content,input_text)
        # response = compute_compatibility(input_text, pdf_content)
        st.subheader("The Repsonse is")
        st.write(response)
    else:
        st.write("Please uplaod the resume")       




   




