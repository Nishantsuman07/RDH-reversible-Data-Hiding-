from flask import Flask, render_template, request, redirect, url_for
import os
import uuid
import cv2
from pevo import pevo_embed  # Import PEVO functions from pevo.py

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Generate unique filenames for uploaded and marked images
            filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the original uploaded image
            file.save(original_path)
            
            # Generate secret data (10,000 bits as per PEVO paper)
            secret_bits = '1' * 10000
            
            try:
                # Embed data using PEVO algorithm
                marked_img, psnr_value = pevo_embed(original_path, secret_bits)
                
                # Save the marked image
                marked_filename = f"marked_{filename}"
                marked_path = os.path.join(app.config['UPLOAD_FOLDER'], marked_filename)
                cv2.imwrite(marked_path, marked_img)

                # Render the result page with original and marked images
                return render_template('result.html', 
                                       original=filename,
                                       marked=marked_filename,
                                       psnr=psnr_value)
                                       
            except Exception as e:
                return f"Error processing image: {str(e)}"

    return render_template('upload.html')

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
