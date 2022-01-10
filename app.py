from flask import Flask, escape, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# change from model
m = pickle.load(open("model.pkl", 'rb'))

app = Flask(__name__)
@app.route('/analysis')
def analysis():
    return render_template("stroke.html")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method =="POST":
        gender = request.form['gender']
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        disease = int(request.form['disease'])
        married = request.form['married']
        work = request.form['work']
        residence = request.form['residence']
        glucose = float(request.form['glucose'])
        bmi = float(request.form['bmi'])
        smoking = request.form['smoking']

        # gender
        if (gender == "Male"):
           
            genderx=1
           
        elif(gender == "Other"):
            genderx=0
            
        else:
            
           genderx=0
        
        # married
        if(married=="Yes"):
        
            marriedx = 1
        else:
            marriedx=0

        # work  type
        if(work=='Self-employed'):
           workx=3
        elif(work == 'Private'):
             workx=2
        elif(work=="children"):
           workx=1
        elif(work=="Never_worked"):
            workx=4
        else:
          workx=0

        # residence type
        if (residence=="Urban"):
            residencex=1
            Residence_type_Urban=1
        else:
            residencex=0

        # smoking sttaus
        if(smoking=='formerly smoked'):
            smokex=1
        elif(smoking == 'smokes'):
            smokex=3
        elif(smoking=="never smoked"):
            smokex=2
        else:
            smokex=0

        feature = scaler.fit_transform([[age, hypertension, disease, glucose, bmi, genderx, marriedx, workx,residencex,smokex]])

        prediction = m.predict(feature)
        print(prediction) 
        # 
        if prediction==0:
            prediction = "NO" 
        else:
            prediction = "YES" 

        return render_template("index.html", prediction_text="Chance of Stroke Prediction is --> {}".format(prediction))   
         

    else:
        return render_template("index.html")


 


if __name__ == "__main__":
    app.run(debug=True)