# 🎬 Movie Recommendation System

A **content-based movie recommendation system** that suggests similar movies based on their plot descriptions.  
The system uses **TF-IDF vectorization** and **cosine similarity** to compute movie similarity and is deployed as an interactive **Streamlit web application**.

---

## 🚀 Live Demo
🔗 **Live App:**  
https://movierecommendationsys-ffrlxq6pjrvqdqxusfxnqj.streamlit.app

---

## 📌 Features
- Select a movie from the dropdown
- Get top recommended similar movies
- Fast and interactive UI
- Fully deployed on Streamlit Cloud

---

## 🧠 How It Works
1. Movie overviews are converted into numerical vectors using **TF-IDF**
2. **Cosine similarity** is calculated between all movies
3. Movies with highest similarity scores are recommended

---

## 🛠️ Tech Stack
- **Python**
- **Pandas**
- **Scikit-learn**
- **Streamlit**
- **TF-IDF Vectorizer**
- **Cosine Similarity**

---

## 📂 Project Structure
Movie-Recommendation-System/
│
├── app.py # Streamlit app
├── requirements.txt # Dependencies
├── tmdb_5000_movies.csv # Dataset


---

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

📊 Dataset

TMDB 5000 Movies Dataset

Contains movie titles and overviews

🎯 Use Case

Learning content-based recommendation systems

Understanding NLP in real-world applications

ML + Deployment project for resume

👩‍💻 Author

Tanu Singh
GitHub: https://github.com/TanuS2428
