# 📚 Anime Recommendation System

A **Content-Based Anime Recommendation Engine** built using **TF-IDF**, **Cosine Similarity**, and **Nearest Neighbors**, wrapped inside an interactive **Streamlit** web application.

This system allows users to enter any anime title and instantly receive similar anime recommendations along with similarity scores.

---

## 🚀 Features

- Content-based recommendations  
- TF-IDF vectorization on combined anime metadata  
- Cosine similarity using NearestNeighbors  
- Fuzzy search for misspelled titles  
- Interactive Streamlit UI  
- Similarity score explained and displayed  
- Customizable dataset & artifacts  

---

## 📁 Project Structure

```
📦 Anime Recommendation System
 ┣ 📄 app.py
 ┣ 📄 anime-dataset-2023.csv
 ┣ 📄 anime_recommender_artifacts.pkl
 ┣ 📄 anime_recommender_notebook.ipynb
 ┣ 📄 requirement.txt
 ┗ 📄 README.md
```

---

## 🧠 How It Works

1. **Text fields** (title, genre, synopsis, etc.) are cleaned and combined.  
2. **TF-IDF vectorizer** converts combined text into vectors.  
3. **NearestNeighbors model** finds the closest anime using cosine distance.  
4. **Similarity Score = 1 - distance**  
5. Results displayed in a clean UI with optional fuzzy title matching.

---

## 📦 Installation

### 1. Clone the repository
```
git clone <your_repo_url>
cd <your_repo_name>
```

### 2. Install dependencies
```
pip install -r requirement.txt
```

### 3. Run the Streamlit app
```
streamlit run app.py
```

Then open:
```
http://localhost:8501
```

---

## 🧪 Using the App

- Enter or select an anime title  
- App automatically fuzzy-matches the closest valid title  
- Displays:
  - Recommended anime  
  - Genres  
  - Description  
  - **Similarity Score (0 to 1)**  

---

## 📈 Improving Accuracy

To improve recommendation quality, consider:
- Adding more metadata columns (studio, tags, themes, characters)  
- Using sentence-transformer embeddings  
- Cleaning inconsistencies in dataset  
- Adding lemmatization or NLP-based preprocessing  

---

## 🛠 Tech Stack

- Python  
- Pandas / NumPy  
- Scikit-learn  
- Streamlit  
- TF-IDF Vectorization  
- Cosine Similarity  

---

## 🤝 Contributing

Pull requests and suggestions are welcome!

---

## 📜 License

This project is licensed under the MIT License.

