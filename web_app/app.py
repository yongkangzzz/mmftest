from flask import Flask, make_response,Response,render_template, request, url_for, redirect, session, send_from_directory
from datetime import datetime
from lime4mmf import lime_mmf
from flask_apscheduler import APScheduler
from config import APSchedulerJobConfig
#from shap4mmf import shap_mmf
from text_removal.smart_text_removal import SmartTextRemover
from datetime import timedelta
import random
from file_manage import *
from flask_sqlalchemy import SQLAlchemy
import socket

app = Flask(__name__)
app.config.from_object(APSchedulerJobConfig)
app.config['SECRET_KEY'] = os.urandom(24)
#app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=120)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'Secret Key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user_info.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# 查询时会显示原始SQL语句
app.config['SQLALCHEMY_ECHO'] = True
db = SQLAlchemy(app)

class user_info(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name =db.Column(db.String(20))
    ip_addr = db.Column(db.String(20))

    def __init__(self,file_name,ip_addr):
        self.file_name = file_name
        self.ip_addr = ip_addr

def generate_random_str(randomlength=16):
  """
  生成一个指定长度的随机字符串
  """
  random_str = ''
  base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
  length = len(base_str) - 1
  for i in range(randomlength):
    random_str += base_str[random.randint(0, length)]
  return random_str

@app.before_request
def before_request():
    id = generate_random_str(8)
    if session.get('user') == None:
        mkdir('./static./user/' + id)
        print(id + "created")
        session['user'] = id
    else:
        print("Not none")

# @app.after_request
# def after_request(response):
#     now = datetime.now()
#     try:
#         last_active = session['last_active']
#         delta = now - last_active
#         print(delta.seconds)
#         if delta.seconds > 5:
#             session['last_active'] = now
#             file_path = './static/user/' + session['user']
#             print(file_path)
#             clean(file_path)
#             if(os.path.exists(file_path)):
#                 os.rmdir(file_path)
#             print("expired")
#             response = Response("expired")
#             response.response = render_template('home.html')
#             return response
#     except:
#         pass
#
#     try:
#         session['last_active'] = now
#         return response
#     except:
#         pass



@app.route('/')
def home():
    ip_addr = request.remote_addr
    users_delete = user_info.query.filter_by(ip_addr=ip_addr).all()
    if len(users_delete) != 0:
        for user_delete in users_delete:
            db.session.delete(user_delete)
            db.session.commit()
        file_name = session['user']
        user_insert = user_info(file_name,ip_addr)
        db.session.add(user_insert)
        db.session.commit()
    return render_template('home.html')


@app.route('/docs/')
def docs():
    return render_template('docsify.html')


@app.route('/docs/<path:path>')
def send_js(path):
    return send_from_directory('docs', path)


@app.route('/explainers/hateful-memes')
def hateful_memes():
    filename_img = request.args.get('imgName')
    filename_exp = request.args.get('imgExp')
    if filename_img is None:
        filename_img = 'logo.png'
    if filename_exp is None:
        filename_exp = 'logo.png'
    text = request.args.get('imgTexts')
    result = request.args.get('explanations')
    print("in home:", result)
    session['imgName'] = filename_img
    session['imgTexts'] = text
    session['imgExp'] = filename_exp
    session['explanations'] = result
    return render_template('explainers/hateful-memes.html', expMethod='Select a method', imgName=filename_img, imgTexts=text, imgExp=filename_exp, explanations=result)


@app.route('/uploadImage', methods=['POST'])
def uploadImage():
    file = request.files['inputImg']
    print('.//static//' + 'user/' + session['user'] + file.filename)
    img_path = './/static//' + 'user/' + session['user'] + '/' + file.filename
    img_name = 'user/' + session['user'] + '/' + file.filename
    file.save(img_path)
    return redirect(url_for('hateful_memes', imgName=img_name))


@app.route('/uploadModel', methods=['POST'])
def uploadModel():
    option = request.form['selectOption']
    if option == 'internalModel':
        selectedModel = request.form['selectModel']
        file = request.files['inputCheckpoint1']
        print(file.filename)
        print(selectedModel)
        file.save('.//static//' + file.filename)
    elif option == 'selfModel':
        file = request.files['inputCheckpoint2']
        print(file.filename)
        file.save('.//static//' + file.filename)
    elif option == 'noModel':
        selectedExistingModel = request.form['selectExistingModel']
        print(selectedExistingModel)
    else:
        raise Exception("Sorry, you must select an option to continue!!!")
    return redirect(url_for('hateful_memes'))

@app.route('/inpaint/<imgName>')
def inpaint(imgName):
    # Prepare the image path and names
    folder_path = './static/'
    image_path = folder_path + imgName
    
    img_name_no_extension = os.path.splitext(imgName)[0]
    img_extension = os.path.splitext(imgName)[1]

    inpainted_image_name = img_name_no_extension + "_inpainted" + img_extension
    save_path = folder_path + inpainted_image_name

    if not os.path.isfile(save_path):
        # Load the inpainter
        inpainter = SmartTextRemover("text_removal/frozen_east_text_detection.pb")

        # Inpaint image
        img_inpainted = inpainter.inpaint(image_path)

        # save inpainted image
        img_inpainted.save(save_path)

    return redirect(url_for('hateful_memes', imgName=inpainted_image_name))

@app.route('/explainers/hateful-memes/predict/<imgName>', methods=['POST'])
def predict(imgName):
    text = request.form['texts']
    expMethod = request.form['expMethod']
    imgName = imgName.strip("'()")
    if expMethod == 'shap':
        #result, imgExp = shap_mmf.predict_HM(imgName, text)
        result, imgExp = lime_mmf.lime_multimodal_explain(imgName, text)
    elif expMethod == 'lime':
        result, imgExp = lime_mmf.lime_multimodal_explain(imgName, text)
    elif expMethod == 'torchray':
        result = ['coming soon!']
    else:
        #result = shap_mmf.predict_HM(imgName, text)
        result, imgExp = lime_mmf.lime_multimodal_explain(imgName, text)

    print(result)
    # response = make_response(render_template('home.html', imgName=imgName, imgTexts=text, imgExp="shap4mmf.png", explanations=result))
    # response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    # response.headers['Pragma'] = 'no-cache'
    return render_template('explainers/hateful-memes.html', imgName=imgName, imgTexts=text, imgExp=imgExp, explanations=result)


if __name__ == '__main__':
    app.run()

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()
