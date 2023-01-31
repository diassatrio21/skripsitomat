from flask import Flask, render_template, flash, redirect, url_for, session, request, jsonify, Response
from flask_mysqldb import MySQL
from wtforms import Form, StringField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
from wtforms.fields.html5 import EmailField
import os
import numpy as np
import re
import base64
import cv2
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as image_utils


app = Flask(__name__)

####################################
# Begin Database MySQL setup
####################################
app.secret_key = os.urandom(24)

# mysql = MySQL()
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'skripsi_silvi'
# app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# Initialize the app for use with this MySQL class
# mysql.init_app(app)

# Decorators used for checking login or logout
# def is_logged_in(f):
#     @wraps(f)
#     def wrap(*args, **kwargs):
#         if 'logged_in' in session:
#             return f(*args, *kwargs)
#         else:
#             flash('Unauthorized, Please login', 'danger')
#             return redirect(url_for('login'))
#     return wrap

# Decorator, extracted from the Wraps class.
# def not_logged_in(f):
#     @wraps(f)
#     def wrap(*args, **kwargs):
#         if 'logged_in' in session:
#             flash('Unauthorized, You logged in', 'danger')
#             return redirect(url_for('register'))
#         else:
#             return f(*args, *kwargs)
#     return wrap

# Panggil Model
quality_model = load_model('model/dauntomat.h5')

# Panggil camera webcame
# camera = cv2.VideoCapture(0)

#Fungsi Prediksi Gambar 
def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")


def model_predict(image, model):
    image = image.resize((150, 150))           
    image = image_utils.img_to_array(image)
    image = image.reshape(-1,150, 150, 3)
    image = image.astype('float32')
    image = image / 255.0
    preds = model.predict(image)
    return preds
# End Fungsi Prediksi Gambar


# Route untuk pindah halaman
@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/home')
# def home():
#     return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/image")
def image_classify():
    return render_template("image.html")

@app.route("/webcam")
def webcam_classify():
    return render_template("webcam.html")


# Login Form Class
# class LoginForm(Form):  
#     username = StringField('Username', [validators.length(min=1)], render_kw={'autofocus': True})
    
# User Login
# @app.route('/', methods=['GET', 'POST'])
# @not_logged_in
# def login():
#     # Set variable message None at first
#     message = None
#     form = LoginForm(request.form)
#     if request.method == 'POST' and form.validate():
#         # User form
#         username = form.username.data
#         password_candidate = request.form['Password']

#         # Create cursor
#         cur = mysql.connection.cursor()

#         # Query for get user by username
#         result = cur.execute("SELECT * FROM user WHERE username=%s", [username])

#         if result > 0:
#             # Get stored value
#             data = cur.fetchone()
#             userid = data['User_Id']
#             email = data['Email']
#             password = data['Password']
#             # Condition when correct password
#             if sha256_crypt.verify(password_candidate, password):
#                 # passed
#                 session['logged_in'] = True
#                 session['Username'] = username
#                 session['User_Id'] = userid
#                 session['Email'] = email
#                 message = 'You are now logged in'
#                 return redirect(url_for('home'))
#             # Condition when incorrect password
#             else:
#                 message = 'Incorrect password'
#                 return render_template('login.html', form=form, message=message)
#         # Condition when incorrect username
#         else:
#             message = 'Username not found'
#             # Close connection
#             cur.close()
#             return render_template('login.html', form=form, message=message)
#     return render_template('login.html', form=form, message=message)


# @app.route('/logout')
# def logout():
#     # Create cursor
#     cur = mysql.connection.cursor()
#     session.clear()
#     flash('You are logged out', 'success')
#     return redirect(url_for('login'))


# Registration Form Class
# class RegisterForm(Form):
#     username = StringField('Username', [validators.length(min=3, max=255)])
#     password = PasswordField('Password', [validators.length(min=3)])
#     email = EmailField('Email', [validators.DataRequired(), validators.Email(), validators.length(min=4, max=255)])


# @app.route('/register', methods=['GET', 'POST'])
# @not_logged_in
# def register():
#     # Set variable message None at first
#     message = None
#     form = RegisterForm(request.form)
#     if request.method == 'POST' and form.validate():
#         # User form
#         username = form.username.data
#         password = sha256_crypt.encrypt(str(form.password.data))
#         email = form.email.data

#         # Create Cursor
#         cur = mysql.connection.cursor()

#         # Query for check email if already taken and available on db
#         check_email = cur.execute("SELECT * FROM user WHERE Email = (%s)", ([email]))

#         # Condition when email if already taken and available on db
#         if int(check_email) > 0:
#            message = 'That email is already taken, please choose another'
#            return render_template('register.html', form=form, message=message)
#         # Condition when email is new
#         else: 
#            # Query for insert user input into db
#            cur.execute("INSERT INTO user (username, password, email) VALUES(%s, %s, %s);",
#                     (username, password, email))
#            # Commit cursor
#            mysql.connection.commit()
#            # Close Connection
#            cur.close()
#            return redirect(url_for('login'))

#     return render_template('register.html', form=form, message=message)

    
@app.route('/prediction-image', methods=['GET','POST'])
def prediction_image():
    if request.method=='POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        # Make prediction
        preds = model_predict(img, quality_model)
        target_names = ['bakteri', 'bercakdaun', 'busukdaun', 'sehat']     
        hasil_label = target_names[np.argmax(preds)]
        hasil_prob = "{:.2f}".format(100 * np.max(preds))
        return jsonify(result=hasil_label, probability=hasil_prob)

    return render_template('image.html')

def prediction_webcam(): 
    camera = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            print("[INFO] loading and preprocessing image...")
            image = Image.fromarray(frame, 'RGB')
            image = image.resize((150,150))
            image = image_utils.img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # Classify the image
            print("[INFO] classifying image...")
            preds = quality_model.predict(image)
            target_names = ['bakteri', 'bercakdaun', 'busukdaun', 'sehat']     
            hasil_label = target_names[np.argmax(preds)]
            hasil_prob = "{:.2f}".format(100 * np.max(preds))
            prediction = f'{hasil_prob}% {hasil_label}'

            cv2.putText(frame, "Label: {}".format(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (209, 80, 0, 255), 2)

            if cv2.waitKey(5) & 0xFF == 27:
                break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concat frame one by one and show result
    camera.release()
    
@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(prediction_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)

