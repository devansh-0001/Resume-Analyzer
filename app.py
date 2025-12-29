import streamlit as st
import nltk
import re
import pickle
import pdfplumber
# nltk.download('pnkt')
# nltk.download('stopwords')


clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))


def extract_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

    return text




def cleanResume(txt):
    cleanTxt = re.sub(r'http\S+',' ',txt)
    cleanTxt = re.sub(r'@\S+',' ',cleanTxt)
    cleanTxt = re.sub(r'#\S+',' ',cleanTxt)
    cleanTxt = re.sub(r'[%s]' % re.escape("""!"$*()+-_,<>{}.[]^|?&;="""),' ',cleanTxt)
    cleanTxt = re.sub(r'\s+',' ',cleanTxt)   
    return cleanTxt




def main():
    st.title("Resume Analyzer")
    st.markdown("### 📤 Upload Your Resume")
    uploaded = st.file_uploader(
        "",
        type=["txt", "pdf"],
        help="Upload resume in TXT or PDF format"
    )
    st.markdown("""
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }

    /* Upload box */
    .css-1cpxqw2, .css-1n543e5 {
        background-color: #1e293b !important;
        border-radius: 12px !important;
        padding: 25px !important;
        border: 2px dashed #38bdf8 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #38bdf8, #3b82f6);
        color: white;
        border-radius: 10px;
        font-weight: bold;
        padding: 10px 25px;
    }

    /* Result box */
    .result-box {
        background: #020617;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #38bdf8;
        margin-top: 20px;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

    
    if uploaded is not None:
        if uploaded.type == "application/pdf":
            resume_text = extract_pdf(uploaded)
        else:
            resume_text = uploaded.read().decode("utf-8",errors = 'ignore')
        cleaned = cleanResume(resume_text)
        encoded = tfidf.transform([cleaned])
        predicted = clf.predict(encoded)[0]
        categories = {
            6:"Data Science",12:"HR",0:"Advocate",1:"Arts",24:"Web Designing",16:"Mechanical Engineer",22:"Sale",14:"Health and Fitness",5:"Civil Engineer",15:"Java Developer",4:"Business Analyst",21:"SAP Developer",
            2:"Automation Tesing",11:"Electrical Engineer",18:"Operations Manager",20:"Python Developer",8:"DevOps Engineer",17:"Network and Security Engineer",19:"PMO",7:"Database",13:"Hadoop",10:"ETL Developer",9:"DotNet Developer",3:"Blockchain",23:"Testing"
        }
        category_predicted = categories.get(predicted)
        st.markdown(
            f"""
            <div class="result-box">
                <b>🎯 Predicted Category:</b><br>
                {category_predicted}
            </div>
            """,
            unsafe_allow_html=True
        )
        confidence = max(clf.predict_proba(encoded)[0]) * 100

        st.progress(int(confidence))
        st.caption(f"Prediction Confidence: {confidence:.2f}%")





if __name__ == "__main__":
    main()