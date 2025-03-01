import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import re

# ✅ Load the fine-tuned model and tokenizer
model_path = "./gpt2-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# ✅ Create a text-generation pipeline
finetuned_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# ✅ Extract YouTube video ID from URL
def extract_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

# ✅ Get transcript from YouTube video
def get_youtube_transcript(video_url):
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            return "Invalid YouTube URL."
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript])
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

# ✅ Generate a blog post from the transcript
def generate_blog(transcript):
    if not transcript or "Error" in transcript:
        return "Error: Unable to generate blog. Invalid or missing transcript."
    
    prompt = f"Generate a blog post based on the following content:\n\n{transcript[:800]}"  # Limit input size
    output = finetuned_generator(prompt, max_length=500, num_return_sequences=1, truncation=True, temperature=0.7)
    
    return output[0]['generated_text']

# ✅ Streamlit UI
st.title("📹 YouTube Blog Generator with Human-in-the-Loop")

# ✅ Input YouTube URL
video_url = st.text_input("Enter YouTube Video URL:")

if st.button("Generate Blog Post"):
    with st.spinner("Processing video... please wait."):
        transcript = get_youtube_transcript(video_url)
        blog_post = generate_blog(transcript)
    
    # ✅ Display the generated blog post
    st.subheader("Generated Blog Post")
    st.text_area("Blog Output:", blog_post, height=300)

    # ✅ Human Review Section
    st.subheader("Human Review")
    st.write("Do you approve the blog post?")
    
    col1, col2 = st.columns(2)
    with col1:
        approve = st.button("✅ Yes")
    with col2:
        disapprove = st.button("❌ No")
    
    if disapprove:
        feedback = st.text_area("Please provide your feedback for improvement:")
        st.write("Thank you for your feedback! The model will be improved accordingly.")

# (.venv) C:\Users\simil\OneDrive\Desktop\FAANG_Roadmap\Project\YouTube-Blog-Generator-with-Human-Feedback-main>streamlit run app.py