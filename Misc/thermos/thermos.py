from flask import Flask,render_template,url_for

app = Flask(__name__)

@app.route('/')
@app.route('/root')
def index():
    return render_template('index.html')

# @app.route('/home')
# def home():
#     return 'Home'

if __name__ == "__main__":
    print(app.url_map)
    app.run(debug="true")
