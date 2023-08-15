from flask import Flask,render_template,request
import pickle

app=Flask(__name__)
model=pickle.load(open('D:\\flask\\fraud\\saveModelfraud.sav','rb'))

@app.route('/',methods=['GET','POST'])


def transaction():
    if request.method=='POST':
        V1=float(request.form['v1'])
        V2 = float(request.form['v2'])
        V3 = float(request.form['v3'])
        V4 = float(request.form['v4'])
        V5=float(request.form['v5'])
        V6 = float(request.form['v6'])
        V7 = float(request.form['v7'])
        V8 = float(request.form['v8'])
        V9=float(request.form['v9'])
        V10 = float(request.form['v10'])
        V11 = float(request.form['v11'])
        V12 = float(request.form['v12'])
        V13 =float(request.form['v13'])
        V14 = float(request.form['v14'])
        V15 = float(request.form['v15'])
        V16 = float(request.form['v16'])
        V17=float(request.form['v17'])
        V18 = float(request.form['v18'])
        V19 = float(request.form['v19'])
        V20 = float(request.form['v20'])
        V21 = float(request.form['v21'])
        V22 = float(request.form['v22'])
        V23 =float(request.form['v23'])
        V24 = float(request.form['v24'])
        V25 = float(request.form['v25'])
        V26 = float(request.form['v26'])
        V27=float(request.form['v27'])
        V28 = float(request.form['v28'])
        amount =float(request.form['amount'])
        y = model.predict([[V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,amount]])[0]
        result=''
        if y==0:
            result="It is normal transaction"
        else:
            result="It is fraud transaction"

    return render_template('fraud.html',**locals())

if __name__== '__main__':
    app.run(debug=True)