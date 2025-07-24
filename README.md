# RadiantGlow: AI-Powered Skincare Product Recommendation Web App 💫

**RadiantGlow** is an AI-powered web application that provides personalized skincare product recommendations based on user input or uploaded skin images. It uses a trained machine learning model to detect skin concerns and suggest the most suitable products accordingly.

---

## 🚀 Features

- 📷 **Image Upload for Skin Concern Detection**
- ✍️ **Manual Form Input for Skin Type & Concern**
- 🎯 **Product Recommendations** based on detected concern
- 🛒 **Add to Cart** and Wishlist functionality
- 🔐 **User Login and Registration**
- 🎨 Clean, responsive UI with HTML, CSS, and JavaScript

---

## 🧠 Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask)
- **Machine Learning**: scikit-learn, OpenCV, Pandas
- **Storage**: CSV file for product data
- **Tools**: joblib (model loading), Werkzeug (secure image upload)

---

## 🖼️ How It Works

1. **Upload Image** OR **Fill in a form** with skin type/concern.
2. ML model predicts the user's skin condition.
3. The app shows:
   - Detected skin concern (e.g., acne, dryness, pigmentation)
   - Recommended products with name, image and buy link
   - Explanation of why the product suits the condition



## 📦 How to Run Locally


git clone https://github.com/your-username/radiantglow-skincare-ai.git
cd radiantglow-skincare-ai
pip install -r requirements.txt
python app.py
Visit http://localhost:5000 in your browser.






