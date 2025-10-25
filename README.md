# Smart Crop Recommendation System

## Overview
A simple web app that recommends the top 3 crops for given soil and weather conditions using a Random Forest model.

## Setup
1. Clone this repository / extract folder.
2. Place the Figshare dataset CSV into `data/` folder as `crop_data.csv`.
3. Create a virtual environment and install dependencies:
   pip install -r requirements.txt
4. Train the model:
   python train_model.py
5. Run the Streamlit app:
   streamlit run app/app.py
6. Enter feature values and see the top 3 recommended crops.
