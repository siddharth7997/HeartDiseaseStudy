from flask import Flask, render_template, request
import sys
import pandas
import numpy
import pickle



app = Flask(__name__)
app.debug = True

@app.route('/')
def home():
    return render_template('Test.html')
    
@app.route('/', methods=['POST'])
def home_post():
    #Taking variable data from web page
    Age__1 = request.form['Age__1']
    Age__1 = Age__1.upper()
    Gender__2 = request.form['Gender__2']
    Gender__2 = Gender__2 = Gender__2.upper()
    Chest_Pain_Type__3 = request.form['Chest_Pain_Type__3']
    Chest_Pain_Type__3 = Chest_Pain_Type__3.upper()
    Blood_Pressure_In_mmHg__4 = request.form['Blood_Pressure_In_mmHg__4']
    #Blood_Pressure_In_mmHg__4 = Blood_Pressure_In_mmHg__4.upper()
    Cholesterol_In_mgdl__5 = request.form['Cholesterol_In_mgdl__5']
    #Cholesterol_In_mgdl__5 = Cholesterol_In_mgdl__5.upper()
    Fasting_Blood_Sugar__6 = request.form['Fasting_Blood_Sugar__6']
    Fasting_Blood_Sugar__6 = Fasting_Blood_Sugar__6.upper()
    Resting_ECG__7 = request.form['Resting_ECG__7']
    Resting_ECG__7 = Resting_ECG__7.upper()
    Maximum_Heart_Rate__8 = request.form['Maximum_Heart_Rate__8']
    Maximum_Heart_Rate__8 = Maximum_Heart_Rate__8.upper()
    Exercise_Induced_Angina__9 = request.form['Exercise_Induced_Angina__9']
    Exercise_Induced_Angina__9 = Exercise_Induced_Angina__9.upper()
    Old_Peak__10 = request.form['Old_Peak__10']
    Old_Peak__10 = Old_Peak__10.upper()
    Slope__11 = request.form['Slope__11']
    Slope__11 = Slope__11.upper()
    Number_of_Vessels_Colored__12 = request.form['Number_of_Vessels_Colored__12']
    Number_of_Vessels_Colored__12 = Number_of_Vessels_Colored__12.upper()
    Thal__13 = request.form['Thal__13']
    Thal__13 = Thal__13.upper()
    
    
    #Converting data to whatever the model accepts
    EXANG = 0
    if Exercise_Induced_Angina__9 == "YES":
        EXANG = 1
    THALACH = int(Maximum_Heart_Rate__8)
    FBS = 0
    if Fasting_Blood_Sugar__6 == ">120 mg/dl":
        FBS = 1
    
    AGE = int(Age__1)
    SEX = 1
    if Gender__2 == 'MALE':
        SEX = 1
    else:
        SEX = 0
        
    #Splitting nominal to binary
    CP_1 = 0
    CP_2 = 0
    CP_3 = 0
    CP_4 = 0
    
    if Chest_Pain_Type__3 == 'TYPICAL ANGINA':
        CP_1 = 1
    elif Chest_Pain_Type__3 == 'ATYPICAL ANGINA':
        CP_2 = 1
    elif Chest_Pain_Type__3 ==  'NON-ANGINA PAIN':
        CP_3 = 1
    elif Chest_Pain_Type__3 == 'ASYMPTOMATIC':
        CP_4 = 1
        
    THRESTBPS = int(Blood_Pressure_In_mmHg__4)
    CHOL = int(Cholesterol_In_mgdl__5)
    #EXANG = int(Exercise_Induced_Angina__9)
    OLDPEAK = float(Old_Peak__10)
    CA = int(Number_of_Vessels_Colored__12)
    RESTECG_0 = 0
    RESTECG_1 = 0
    RESTECG_2 = 0
    if Resting_ECG__7 == "NORMAL":
        RESTECG_0 = 1
    elif Resting_ECG__7 == "ABNORMAL":
        RESTECG_1 = 1
    elif Resting_ECG__7 == "HYPERTROPHY":
        RESTECG_2 = 1
    
    SLOPE_1 = 0
    SLOPE_2 = 0
    SLOPE_3 = 0
    if Slope__11 == "UPSLOPING":
        SLOPE_1 = 1
    elif Slope__11 == "FLAT":
        SLOPE_2 = 1
    elif Slope__11 == "DOWNSLOPING":
        SLOPE_3 = 1
    
    THAL_3 = 0
    THAL_6 = 0
    THAL_7 = 0
    if Thal__13 == "NORMAL":
        THAL_3 = 1
    elif Thal__13 == "FIXED DEFECT":
        THAL_6 = 1
    elif Thal__13 == "REVERSIBLE DEFECT":
        THAL_7 = 1
    
    AGE=(AGE - 53.5108695652)/9.42468520958
    THRESTBPS = (THRESTBPS - 131.354347826)/18.6821090831
    CHOL = (CHOL - 199.126086957)/108.957636065
    THALACH = (THALACH - 137.513043478)/25.1388276709
    OLDPEAK = (OLDPEAK - 0.880217391304)/1.0537873849
    CA = (CA - 0)/3
    arr=[AGE, SEX, THRESTBPS, CHOL, FBS, THALACH, EXANG, OLDPEAK, CA, CP_1, CP_2, CP_3, CP_4,  RESTECG_0, RESTECG_1, RESTECG_2, SLOPE_1, SLOPE_2, SLOPE_3, THAL_3, THAL_6, THAL_7]
    
    svm_main=pickle.load(open('svm_main','rb'))
    svm_prob=pickle.load(open('svm_prob','rb'))
    
    
    ans=svm_main.predict([arr])[0]
    #return "|".join([str(x) for x in arr])+"<br>prediction is "+str(ans)+"<br> probablity"+str(svm_prob.predict_proba([arr])[0][1])
    return render_template("result1.html", value = [str(svm_prob.predict_proba([arr])[0][1]), str(ans)])

if __name__ == "__main__":
    app.run(debug=True)
