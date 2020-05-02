from flask import Flask
from flask import jsonify, render_template, request
from model.core import predict

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def homapage():
    """Return my homepage."""
    return render_template("index.html")

@app.route('/handle_data', methods=['POST'])
def handle_data():
    data_dict = {}
    data_dict["age"] = request.form['age']
    data_dict["workclass"] = request.form['workclass']
    data_dict["fnlwgt"] = request.form['fnlwgt']
    data_dict["education"] = request.form['education']
    data_dict["educationnum"] = request.form['education-num']
    data_dict["maritalstatus"] = request.form['marital-status']
    data_dict["occupation"] = request.form['occupation']
    data_dict["relationship"] = request.form['relationship']
    data_dict["race"] = request.form['race']
    data_dict["sex"] = request.form['sex']
    data_dict["capital­gain"] = request.form['capital­-gain']
    data_dict["capital­loss"] = request.form['capital­-loss']
    data_dict["hours­perweek"] = request.form['hours-­per­-week']
    data_dict["native­country"] = request.form['native­-country']

    instance = [
        int(data_dict["age"]),
        data_dict["workclass"].replace('\xad', ''),
        int(data_dict["fnlwgt"]),
        data_dict["education"].replace('\xad', ''),
        int(data_dict["educationnum"]),
        data_dict["maritalstatus"].replace('\xad', ''),
        data_dict["occupation"].replace('\xad', ''),
        data_dict["relationship"].replace('\xad', ''),
        data_dict["race"].replace('\xad', ''),
        data_dict["sex"].replace('\xad', ''),
        int(data_dict["capital­gain"]),
        int(data_dict["capital­loss"]),
        float(data_dict["hours­perweek"]),
        data_dict["native­country"].replace('\xad', '')
    ]
    instances = [instance]

    if predict(instances)[0] > 0.5:
        return render_template("true.html")

    else:
        return render_template("false.html")

@app.route('/home', methods=['POST'])
def home():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)