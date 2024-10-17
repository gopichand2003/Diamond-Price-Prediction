from flask import Flask,render_template,request
import pickle
import pandas as pd
applicaton=Flask(__name__)
app=applicaton


@app.route('/',methods=['POST','GET'])
def home():
    if request.method=='GET':
        return render_template('index.html')
    else:
        carat=int(request.form.get('carat'))
        cut=request.form.get('cut')
        color=request.form.get('color')
        clarity=request.form.get('clarity')
        depth=int(request.form.get('depth'))
        table=int(request.form.get('table'))
        x=int(request.form.get('x'))
        y=int(request.form.get('y'))
        z=int(request.form.get('z'))
        x=[carat,cut,color,clarity,depth,table,x,y,z]
        columns=['carat', 'cut', 'color', 'clarity', 'depth',
               'table', 'x', 'y', 'z']
        dictonary=dict()
        idx=0
        for i in columns:
            dictonary[i]=[x[idx]]
            idx+=1
        test_df=pd.DataFrame(dictonary)
        processor=pickle.load(open('processor.pkl','rb'))
        values=processor.transform(test_df)
        loaded_model = pickle.load(open('model.pkl', 'rb'))
        predicted_price=loaded_model.predict(values)[0]
        print(predicted_price)
        return render_template('index.html',result=predicted_price)

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)
