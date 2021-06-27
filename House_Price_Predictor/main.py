from flask import Flask,render_template,request
import ML_Project
from Filter_locations import final_locations
from ML_Project import predict_price
app = Flask(__name__)

@app.route('/',methods =["GET", "POST"])
def get_details():
    context={}
    if request.method == "POST": 
       # getting input with name = fname in HTML form 
        location = request.form.get("name") 

       # getting input with name = lname in HTML form  
        area = request.form.get("aname")
        
       # getting input with name = fname in HTML form 
        bhk = request.form.get("bhknumber") 
        
       # getting input with name = lname in HTML form  
        bath = request.form.get("bnumber")

        result = predict_price(location, area, bhk, bath)
        context['result']=round(result,4)
    
    context['location']=final_locations
    
    return render_template('index.html',context=context)

if __name__ == "__main__":
    app.run(debug=True)