import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

model = load_model('Octopus.h5')

model.make_predict_function()

def predict_label(img_path):
	i = load_img(img_path, target_size=(200,200))
	i = img_to_array(i)/255.0
	i = i.reshape(1,200,200,3)

	p = model.predict(i)
	text = []
	if p < 0.9:
		p[0][0] *= 100
		txt = 'ปลาหมึกไม่มีพิษ'
	else:
		p[0][0] *= 100
		txt = 'ปลาหมึกมีพิษ!!!'
	pre = '%.2f'%p[0][0]
	return pre,txt


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	

		img.save(img_path)

		p1 = predict_label(img_path)


	return render_template("index.html", prediction = p1[0], TEXT = p1[1], img_path = img_path)


if __name__ =='__main__':
	app.run(port = 3000 , debug = True)