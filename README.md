# 🧠 Resume EXPERT — AI-Powered Resume Analyzer

A professional-grade resume analyzer built with **Streamlit** and powered by **Google's Gemini AI**. Upload your resume, input a job description, and get detailed feedback, improvement tips, YouTube learning resources, compatibility scores, and interview guidance

---

## 🚀 Features

- ✅ **Resume Evaluation** — Checks strengths and weaknesses based on the job description
- 🔍 **Skill Gap Analysis** — Suggests YouTube videos to fill in missing skills
- 📊 **ATS Compatibility Score** — Provides a percentage match with missing keyword suggestions
- 🎯 **Interview Prep** — Offers curated tips and likely interview questions
- 🤖 **Gemini-Powered** — Uses `gemini-1.5-flash` for intelligent content generation

---

## 🛠 Tech Stack

- **Frontend:** Streamlit
- **NLP Engine:** Google Gemini via `google-generativeai`
- **Resume Parsing:** `pdf2image` + `Pillow`
- **Similarity Matching:** `TF-IDF` + `cosine similarity` from `scikit-learn`

---

## 📦 Installation

### 1. Clone the repository
git clone https://github.com/your-username/resume-expert.git
cd resume-expert

### Create a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate


### Install the requirements
pip install -r requirements.txt


### Configure environment variables

Create a .env file in the root directory:
GOOGLE_API_KEY=your_google_gemini_api_key


###🔧 Additional Setup
Install Poppler (required by pdf2image):

Ubuntu/Debian: sudo apt-get install poppler-utils

macOS: brew install poppler

Windows: Download poppler and add it to PATH

### ▶️ Run the App
streamlit run app.py


📌 Sample Use Cases
1. HRs screening candidates for technical jobs
2. Job seekers analyzing and improving their resumes
3. Resume optimization for ATS compatibility

📬 Contact
Open an issue or create a pull request for suggestions or improvements.

Level up your job application with AI. 💼✨
