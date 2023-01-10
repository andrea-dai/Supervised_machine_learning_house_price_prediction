import streamlit as st

st.title("Housing Prices Prediction")
 
st.write("""
### Project description
I trained several models to predict the price of a house based on features such as the area of the house and the condition and quality of their different rooms, etc. Finally I choode to use the model based on xgboost to predict the house price""") 

import pickle
model = pickle.load(open('trained_pipe_xgboost.sav', 'wb'))

ID = st.number_input("ID")
MSSubClass = st.number_input("The building class")
MSZoning = st.text_input("The general zoning classification")
LotFrontage = st.number_input("Linear feet of street connected to property")
LotArea = st.number_input("Lot size in square feet")
Street = st.text_input("Type of road access")
Alley = st.text_input("Type of alley access")
LotShape = st.text_input("General shape of property")
LandContour = st.text_input("Flatness of the property")
Utilities = st.text_input("Type of utilities available")
LotConfig =  st.text_input("Lot configuration")
LandSlope = st.text_input("Slope of property")
Neighborhood = st.text_input("Physical locations within Ames city limits")
Condition1 = st.text_input("Proximity to main road or railroad")
Condition2 = st.text_input("Proximity to main road or railroad (if a second is present)")
BldgType = st.text_input("Type of dwelling")
HouseStyle = st.text_input("Style of dwelling")
OverallQual = st.number_input("Overall material and finish quality")
OverallCond = st.number_input("Overall condition rating")
YearBuilt = st.number_input("Original construction date")
YearRemodAdd = st.number_input("Remodel date")
RoofStyle = st.text_input("Type of roof")
RoofMatl = st.text_input("Roof material")
Exterior1st = st.text_input("Exterior covering on house")
Exterior2nd = st.text_input("Exterior covering on house (if more than one material)")
MasVnrType = st.text_input("Masonry veneer type")
MasVnrArea = st.number_input("Masonry veneer area in square feet")
ExterQual = st.text_input("Exterior material quality")
ExterCond = st.text_input("Present condition of the material on the exterior")
Foundation = st.text_input("Type of foundation")
BsmtQual = st.text_input("Height of the basement")
BsmtCond = st.text_input("General condition of the basement")
BsmtExposure = st.text_input("Walkout or garden level basement walls")
BsmtFinType1 = st.text_input("Quality of basement finished area")
BsmtFinSF1 = st.number_input("Type 1 finished square feet")
BsmtFinType2 = st.text_input("Quality of second finished area (if present)")
BsmtFinSF2 = st.number_input("Type 2 finished square feet")
BsmtUnfSF = st.number_input("Unfinished square feet of basement area")
TotalBsmtSF = st.number_input("Total square feet of basement area")
Heating = st.text_input("Type of heating")
HeatingQC = st.text_input("Heating quality and conditionv")
CentralAir = st.text_input("Central air conditioning")
Electrical = st.text_input("Electrical system")
firstFlrSF = st.number_input("First Floor square feet")
secondFlrSF = st.number_input("Second floor square feet")
LowQualFinSF = st.number_input("Low quality finished square feet (all floors)")
GrLivArea = st.text_input("Above grade (ground) living area square feet")
BsmtFullBath = st.number_input("Basement full bathrooms")
BsmtHalfBath = st.number_input("Basement half bathrooms")
FullBath = st.number_input("Full bathrooms above grade")
HalfBath = st.number_input("Half baths above grade")
BedroomAbvGr = st.number_input("Number of Bedrooms")
KitchenAbvGr = st.number_input("Number of kitchens")
KitchenQual= st.number_input(" Kitchen quality")
TotRmsAbvGrd = st.number_input( "Total rooms above grade (does not include bathrooms)")
Functional = st.text_input( "Home functionality rating")
Fireplaces = st.number_input( "Number of fireplaces")
FireplaceQu = st.text_input( "Fireplace quality")
GarageType = st.text_input( "Garage location")
GarageYrBlt = st.number_input( "Year garage was built")
GarageFinish = st.text_input( "Interior finish of the garage")
GarageCars = st.number_input( "Size of garage in car capacity")
GarageArea = st.number_input( "Size of garage in square feet")
GarageQual = st.text_input( "Garage quality")
GarageCond = st.text_input( "Garage condition")
PavedDrive = st.text_input( "Paved driveway")
WoodDeckSF = st.number_input( "Wood deck area in square feet")
OpenPorchSF = st.number_input("Open porch area in square feet")
EnclosedPorch = st.number_input( "Enclosed porch area in square feet")
threeSsnPorch = st.number_input( "Three season porch area in square feet")
ScreenPorch = st.number_input( "Screen porch area in square feet")
PoolArea = st.number_input( "Pool area in square feet")
PoolQC = st.text_input( "Pool quality")
Fence = st.text_input( "Fence quality")
MiscFeature = st.text_input( "Miscellaneous feature not covered in other categories")
MiscVal = st.number_input( "$Value of miscellaneous feature")
MoSold = st.number_input( "Month Sold")
YrSold = st.number_input( "Year Sold")
SaleType = st.text_input("Type of sale")
SaleCondition = st.text_input("Condition of sale")

 
import pandas as pd
new_house = pd.DataFrame({
    'LotArea':[LotArea],
    'TotalBsmtSF':[TotalBsmtSF], 
    'BedroomAbvGr':[BedroomAbvGr], 
    'GarageCars':[GarageCars],
    'MSSubClass':[MSSubClass],
    'MSZoning':[MSZoning], 
    'LotFrontage':[LotFrontage],
    'Street':[Street],
    'LotShape':[LotShape],
    'LandContour':[LandContour],
    'LotConfig':[LotConfig],
    'LandSlope':[LandSlope],
    'Neighborhood':[Neighborhood],
    'Condition1':[Condition1],
    'BldgType':[BldgType],
    'HouseStyle':[HouseStyle],
    'OverallQual':[OverallQual],
    'OverallCond':[OverallCond],
    'YearBuilt':[YearBuilt],
    'YearRemodAdd':[YearRemodAdd],
    'RoofStyle':[RoofStyle],
    'RoofMatl':[RoofMatl],
    'Exterior1st':[Exterior1st],
    'Exterior2nd':[Exterior2nd],
    'MasVnrType':[MasVnrType],
    'MasVnrArea':[MasVnrArea],
    'ExterQual':[ExterQual],
    'ExterCond':[ExterCond],
    'Foundation':[Foundation],
    'BsmtQual':[BsmtQual],
    'BsmtCond':[BsmtCond],
    'BsmtExposure':[BsmtExposure],
    'BsmtFinType1':[BsmtFinType1],
    'BsmtFinSF1':[BsmtFinSF1],
    'BsmtFinType2':[BsmtFinType2],
    'BsmtFinSF2':[BsmtFinSF2],
    'BsmtUnfSF':[BsmtUnfSF],
    'Heating':[Heating],
    'HeatingQC':[HeatingQC],
    'CentralAir':[CentralAir],
    'Electrical':[Electrical],
    '1stFlrSF':[firstFlrSF],
    '2ndFlrSF':[secondFlrSF],
    'LowQualFinSF':[LowQualFinSF],
    'GrLivArea':[GrLivArea],
    'BsmtHalfBath':[BsmtHalfBath],
    'FullBath':[FullBath],
    'HalfBath':[HalfBath],
    'KitchenAbvGr':[KitchenAbvGr],
    'KitchenQual':[KitchenQual],
     'TotRmsAbvGrd':[TotRmsAbvGrd],
    'Functional':[Functional],
    'Fireplaces':[Fireplaces], 
    'FireplaceQu':[FireplaceQu],
    'GarageType':[GarageType],
    'GarageYrBlt':[GarageYrBlt],
    'GarageFinish':[GarageFinish],
    'GarageArea':[GarageArea],
    'GarageQual':[GarageQual],
    'GarageCond':[GarageCond],
    'PavedDrive':[PavedDrive],
    'WoodDeckSF':[WoodDeckSF],
    'OpenPorchSF':[OpenPorchSF],
    'EnclosedPorch':[EnclosedPorch],
    '3SsnPorch':[threeSsnPorch],
    'ScreenPorch':[ScreenPorch],
    'PoolArea':[PoolArea],
    'MiscFeature':[MiscFeature],
    'MiscVal':[MiscVal],
    'MoSold':[MoSold],
    'YrSold':[YrSold],
    'SaleType':[SaleType],
    'SaleCondition':[SaleCondition],
    
    
})

prediction = model.predict(new_house)

st.write("The price of the house is:", prediction)
