import panda as pd
from sklearn.tree import DecisionTreeRegressor

home_data_path = 'data/home_data.csv'
home_data = pd.read.csv(home_data_path)

Y= home_data['SalePrice']

features = ['LotArea', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea', 'GarageCars']
X =home_data[features]
#Create the model
home_data_model =DecisionTreeRegressor(random_state=1)
#Fit the model
home_data_model.fit(X,Y)

#make Predicition
Predictions=home_data_model.predict(X)
print(Predictions)