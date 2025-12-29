# Resume-Analyzer
The Resume Analyzer is an example of a machine learning application that is used to automatically categorize resumes into specific job categories. The project involves TF-IDF feature extraction, Label Encoding and K-Nearest Neighbors (KNN) classification algorithm to identify and forecast the most appropriate job position based on resume content.

Jupyter Notebook is used to train the model and Deploy the model in a Streamlit web-based application so that users could upload their resumes and receive predictions immediately.

# Machine learning approach adopted
The proposed project is a **text-based similarity classification** approach:

- The text in resumes is transformed into numerical feature vectors using **TF-IDF Vectorizer**
- **LabelEncoder** converts categorical job labels into numerical values
- **K-Nearest Neighbors (KNN)** classifies resumes based on similarity with previously labeled resumes

# Dataset
Dataset used:  [Resume Dataset – Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)

Holds resumes that are mapped to several job categories that include:
- Data Science  
- Python Developer  
- Network & Security Engineer  
- Java Developer  
- DevOps Engineer  
- SAP Developer  
- And more

![image alt](https://github.com/devansh-0001/Resume-Analyzer/blob/8f7c27794319a957c868241b8efc4eb12178b70b/image/Screenshot%202025-12-29%20143957.png)


# Project Workflow
1. Load resume data from the Excel file  
2. Clean resume text using regular expressions  
3. Categorize job labels using **LabelEncoder**  
4. Convert resume text into numerical vectors using **TF-IDF**  
5. Split the dataset into training and testing data  
6. Train the **K-Nearest Neighbors (KNN)** classifier on vectorized resumes  
7. Evaluate the model performance  
8. Save the trained model and TF-IDF vectorizer using **Pickle**  
9. Load the trained model in the Streamlit application for prediction  

# Technologies Used
- Python
- Jupyter Notebook
- Streamlit
- Scikit-learn
- TF-IDF vectorizer
- Label Encoder
- K-Nearest Neighbor
- Pickle
- pdfplumber(Extracting text from pdf)

![image alt](https://github.com/devansh-0001/Resume-Analyzer/blob/371a39451ab2b813c65aea7c468dace2d0919fc8/image/Screenshot%202025-12-29%20150120.png)
