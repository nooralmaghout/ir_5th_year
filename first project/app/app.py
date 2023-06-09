from preprocessing import preprocess,query_preprocess
from flask import Flask,render_template,request

app = Flask(__name__ , template_folder='template')


@app.route("/" ,methods=['GET','POST'] )
def welcome():
    return  render_template('welcome.html')

@app.route("/" ,methods=['POST'])
def getDatasetName():
    dataset_name= request.form['dataset']
    preprocess(dataset_name)
    return render_template('Serach.html',d=dataset_name)

@app.route("/result" ,methods=['POST'])
def getQuery():
    query= request.form['query']
    doc =  query_preprocess(query)
    # print(query)
    query_preprocess(query)
    return render_template('result.html' ,q=query ,d=doc )

if __name__ == '__main__':
    app.run(debug=True)