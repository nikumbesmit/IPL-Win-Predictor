## Live Demo
[Open the app](https://ipl-win-predictor-onb.streamlit.app/)

# IPL Win Predictor

An end-to-end machine learning project that predicts the win probability of the chasing team in an IPL match using historical ball-by-ball IPL data.

## Features
- Predicts live win probability of the batting team during a chase
- Uses engineered match-state features:
  - runs left
  - balls left
  - wickets left
  - current run rate
  - required run rate
- Compares Logistic Regression, Random Forest, and Gradient Boosting
- Uses GroupKFold cross-validation grouped by match to prevent data leakage
- Deployed as an interactive Streamlit app

## Tech Stack
- Python
- Pandas
- NumPy
- scikit-learn
- Streamlit

## PROJECT STRUCTURE
IPL-Win-Predictor/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── artifacts/
├── data/
└── src/
