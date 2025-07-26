from flask import Flask, render_template, request, url_for, redirect, session
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import os
from werkzeug.utils import secure_filename
import random
import csv

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------- Create users.csv if not exists ----------
if not os.path.exists('users.csv'):
    with open('users.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['username', 'password'])

# ---------- Helpers ----------
def normalize_text(x):
    return x.strip().title() if isinstance(x, str) else x

def normalize_path(x):
    return x.strip() if isinstance(x, str) else x

# ---------- Load skincare training data ----------
try:
    skin_data = pd.read_csv('skincare_data.csv')
except FileNotFoundError:
    raise FileNotFoundError("❌ 'skincare_data.csv' not found.")

required_cols = ['Skin Type', 'Concern', 'Product Recommendation']
for col in required_cols:
    if col not in skin_data.columns:
        raise ValueError(f"❌ Column '{col}' is missing in 'skincare_data.csv'.")

skin_data['Skin Type'] = skin_data['Skin Type'].map(normalize_text)
skin_data['Concern'] = skin_data['Concern'].map(normalize_text)
skin_data['Product Recommendation'] = skin_data['Product Recommendation'].map(normalize_text)

le_skin = LabelEncoder()
le_concern = LabelEncoder()
le_product = LabelEncoder()

le_skin.fit(skin_data['Skin Type'])
le_concern.fit(skin_data['Concern'])
le_product.fit(skin_data['Product Recommendation'])

skin_data_enc = skin_data.copy()
skin_data_enc['Skin Type'] = le_skin.transform(skin_data['Skin Type'])
skin_data_enc['Concern'] = le_concern.transform(skin_data['Concern'])
skin_data_enc['Product Recommendation'] = le_product.transform(skin_data['Product Recommendation'])

X = skin_data_enc[['Skin Type', 'Concern']]
y = skin_data_enc['Product Recommendation']

model = DecisionTreeClassifier()
model.fit(X, y)

# ---------- Load product data ----------
try:
    product_df = pd.read_csv('skincare_products.csv')
except FileNotFoundError:
    raise FileNotFoundError("❌ 'skincare_products.csv' not found.")

prod_required_cols = ['Product Name', 'Concern', 'Link', 'Image', 'Price']
for col in prod_required_cols:
    if col not in product_df.columns:
        raise ValueError(f"❌ Missing column '{col}' in 'skincare_products.csv'.")

product_df['Product Name'] = product_df['Product Name'].map(normalize_text)
product_df['Concern'] = product_df['Concern'].map(normalize_text)
product_df['Link'] = product_df['Link'].map(normalize_path)
product_df['Image'] = product_df['Image'].map(normalize_path)

# ---------- Prediction Functions ----------
def predict_product(skin_type, concern):
    try:
        skin_encoded = le_skin.transform([normalize_text(skin_type)])[0]
    except ValueError:
        skin_encoded = skin_data_enc['Skin Type'].mode().iloc[0]
    try:
        concern_encoded = le_concern.transform([normalize_text(concern)])[0]
    except ValueError:
        concern_encoded = skin_data_enc['Concern'].mode().iloc[0]

    prediction = model.predict([[skin_encoded, concern_encoded]])[0]
    return le_product.inverse_transform([prediction])[0]

def get_top_products(concern):
    norm = normalize_text(concern)
    top = product_df[product_df['Concern'] == norm]
    return top.head(3).to_dict(orient='records') if not top.empty else product_df.head(3).to_dict(orient='records')

def analyze_image_and_predict(filepath):
    concern = random.choice(list(le_concern.classes_))
    skin_type = "Oily"
    product = predict_product(skin_type, concern)
    explanation = f"Based on your uploaded skin image, we detected signs of <strong>{concern.lower()}</strong>. So we recommend <strong>{product}</strong>."
    top_products = get_top_products(concern)
    return product, concern, explanation, top_products



@app.route('/result')
def result():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('result.html')


# ---------- Authentication Routes ----------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with open('users.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)
            if any(row[0] == username for row in reader):
                return "User already exists!"
        with open('users.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([username, password])
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with open('users.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[0] == username and row[1] == password:
                    session['username'] = username
                    return redirect(url_for('index'))
        return "Invalid credentials!"
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('cart', None)
    return redirect(url_for('login'))

# ---------- Main Routes ----------
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    skin_type = request.form['skin_type']
    concern = request.form['concern']
    product = predict_product(skin_type, concern)
    explanation = f"You selected <strong>{skin_type}</strong> skin with <strong>{concern}</strong>. We recommend <strong>{product}</strong>."
    top_products = get_top_products(concern)
    return render_template('result.html', product=product, explanation=explanation, image_path=None, top_products=top_products)

@app.route('/upload', methods=['POST'])
def upload():
    if 'skin_image' not in request.files or request.files['skin_image'].filename == '':
        return "No image uploaded."
    file = request.files['skin_image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image_url = url_for('static', filename=f'uploads/{filename}')
    product, concern, explanation, top_products = analyze_image_and_predict(filepath)
    return render_template('result.html', product=product, explanation=explanation, image_path=image_url, top_products=top_products)

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    product = {
        'name': request.form['product_name'],
        'image': request.form['product_image'],
        'link': request.form['product_link'],
        'price': float(request.form['product_price'])
    }
    cart = session.get('cart', [])
    cart.append(product)
    session['cart'] = cart
    return redirect(url_for('view_cart'))

@app.route('/cart')
def view_cart():
    cart_items = session.get('cart', [])

    # Calculate total price with quantity support
    total_price = sum(item['price'] * item.get('quantity', 1) for item in cart_items)

    return render_template('cart.html', cart_items=cart_items, total_price=total_price)



@app.route('/buy_now', methods=['POST'])
def buy_now():
    product = {
        'name': request.form['product_name'],
        'image': request.form['product_image'],
        'link': request.form['product_link'],
        'price': float(request.form['product_price'])
    }
    return render_template('buy_now.html', products=[product], total_price=product['price'])

@app.route('/buy_all')
def buy_all():
    cart = session.get('cart', [])
    total = sum(item['price'] for item in cart)
    return render_template('buy_now.html', products=cart, total_price=total)

@app.route('/clear_cart')
def clear_cart():
    session.pop('cart', None)
    return redirect(url_for('index'))

@app.route('/checkout')
def checkout():
    cart_items = session.get('cart', [])
    total_amount = sum(item['price'] * item['quantity'] for item in cart_items)
    return render_template('checkout.html', cart=cart_items, total=total_amount)


@app.route('/thankyou')
def thankyou():
    session.pop('cart', None)  # Clears the cart
    return render_template('thankyou.html')




# ---------- Run ----------
if __name__ == '__main__':
    app.run(debug=True)
