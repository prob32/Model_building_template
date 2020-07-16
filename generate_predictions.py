import numpy as np
import joblib

####Creating and testing input function for car estimates predictions
###Load list
filename0 = 'input_list.sav.gz'
df_list = joblib.load(filename0)

##### Test inputs ### In full version these will be dash inputs
mileage = 169000
Year = '2015'
Make = 'Toyota'
Model = 'Tacoma'

##### Creating a function that will accept all user inputs for make model year, check if they are in the model then
###converts into array for predictions

def model_inputs (mileage,year,make,state,model, my_list):
    inputs = [0] * 40
    Year = "Year_"+year
    Make = "Make_"+make
    Model = "Model_"+model
    State = "State_"+state
    inputs[0] = mileage
    if Year in my_list:
        year_pos = my_list.index(Year)
        inputs[year_pos] = 1
    if Make in my_list:
        make_pos = my_list.index(Make)
        inputs[make_pos] = 1
    if Model in my_list:
        model_pos = my_list.index(Model)
        inputs[model_pos] = 1
    if State in my_list:
        year_pos = my_list.index(State)
        inputs[year_pos] = 1	
    list = np.array(inputs)
    list = list.reshape(-1,40)
    return list


### Load the Random forest model
estimate = model_inputs(mileage,Year,Make,Model,df_list)
filename = 'RF_model.sav.gz'
rfc = joblib.load(filename)
####### Loading OLS Model
filename2 = 'OLS_model.sav.gz'
reg = joblib.load(filename2)

##### Print input function array for debugging and OLS/ RF Model predictions
rf_predictions = rfc.predict(estimate)
ols_predictions = reg.predict(estimate)
print(estimate)
print("Random forest Prediction" , rf_predictions)
print("OLS Prediction" , ols_predictions)