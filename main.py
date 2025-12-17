import os
import secrets
import time
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from flask_wtf.file import FileAllowed, FileRequired
from werkzeug.utils import secure_filename
from xgboost import XGBRegressor
import pandas as pd
import deploy as dp 
import empty as emp

# --- CONFIGURATION ---
app = Flask(__name__)
app.config["SECRET_KEY"] = secrets.token_hex(16)

# IMPORTANT: Images must be in 'static' to be displayed in HTML
app.config['UPLOAD_FOLDER_FRONT'] = 'static/uploads/front'
app.config['UPLOAD_FOLDER_SIDE'] = 'static/uploads/side'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER_FRONT'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER_SIDE'], exist_ok=True)

def get_bmi_status(bmi):
    """Returns the category and a CSS color class based on BMI."""
    if bmi < 18.5:
        return "Underweight", "status-blue"
    elif 18.5 <= bmi < 24.9:
        return "Normal Weight", "status-green"
    elif 25 <= bmi < 29.9:
        return "Overweight", "status-yellow"
    else:
        return "Obese", "status-red"

# --- FORMS ---
class ImageUploadForm(FlaskForm):
    front_view = FileField('Front View', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')
    ])
    side_view = FileField('Side View', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')
    ])
    submit = SubmitField('Analyze BMI')

# --- HELPERS ---
def prepare_files_for_model():
    """Renames uploaded files to 'image.jpg' as required by the feature extractor."""
    try:
        # Standardize Front
        f_list = os.listdir(app.config['UPLOAD_FOLDER_FRONT'])
        if f_list:
            old_p = os.path.join(app.config['UPLOAD_FOLDER_FRONT'], f_list[0])
            new_p = os.path.join(app.config['UPLOAD_FOLDER_FRONT'], "image.jpg")
            # If image.jpg already exists (from a previous overwrite), remove it first to avoid errors on some OS
            if os.path.exists(new_p) and old_p != new_p:
                os.remove(new_p)
            if old_p != new_p:
                os.rename(old_p, new_p)

        # Standardize Side
        s_list = os.listdir(app.config['UPLOAD_FOLDER_SIDE'])
        if s_list:
            old_p = os.path.join(app.config['UPLOAD_FOLDER_SIDE'], s_list[0])
            new_p = os.path.join(app.config['UPLOAD_FOLDER_SIDE'], "image.jpg")
            if os.path.exists(new_p) and old_p != new_p:
                os.remove(new_p)
            if old_p != new_p:
                os.rename(old_p, new_p)
    except Exception as e:
        print(f"Error renaming files: {e}")

# --- ROUTES ---
@app.route("/")
def home():
    return render_template("layout.html", page="home")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    form = ImageUploadForm()
    
    if form.validate_on_submit():
        # 1. Clean previous runs
        emp.empty_directory(app.config['UPLOAD_FOLDER_FRONT'])
        emp.empty_directory(app.config['UPLOAD_FOLDER_SIDE'])

        # 2. Save new files
        f_img = form.front_view.data
        s_img = form.side_view.data
        
        f_path = os.path.join(app.config['UPLOAD_FOLDER_FRONT'], secure_filename(f_img.filename))
        s_path = os.path.join(app.config['UPLOAD_FOLDER_SIDE'], secure_filename(s_img.filename))
        
        f_img.save(f_path)
        s_img.save(s_path)

        # 3. Rename files for processing
        prepare_files_for_model()
        
        try:
            # Point feature extractor to the static uploads folder
            # The 'deploy.py' likely expects the folder containing 'front' and 'side' folders
            base_path = 'static/uploads/' 
            our_features = dp.extract_features(base_path, 1).iloc[:, 1:]
            
            xgb_model = XGBRegressor()
            xgb_model.load_model("XGBM.bin")
            
            prediction = xgb_model.predict(our_features)[0]
            
            # Note: We do NOT empty directories here. We keep them so the HTML can display them.
            # They will be emptied at the start of the NEXT request.
            
            # We pass time.time() as a timestamp to force the browser to refresh the images
            bmi_value = round(prediction, 2)
            
            # Get category and color
            category, color_class = get_bmi_status(bmi_value) # <--- NEW LINE
            
            return render_template(
                "layout.html", 
                page="result", 
                bmi=bmi_value, 
                category=category,       # <--- Pass to template
                color_class=color_class, # <--- Pass to template
                timestamp=time.time()
            )
        
        except Exception as e:
            flash(f"Error during processing: {str(e)}", "error")
            print(f"DEBUG ERROR: {e}") # Print error to console for debugging
            return redirect(url_for('predict'))

    return render_template("layout.html", page="upload", form=form)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    app.run(debug=True)