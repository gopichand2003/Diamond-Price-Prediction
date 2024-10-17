import pandas as pd

df = pd.read_csv('./data/gemstone.csv')
df.head()

df.corr()

df=df.drop(labels=['id'],axis=1)

X = df.drop(labels=['price'],axis=1)
Y = df[['price']]

Y

categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns

numerical_cols

categorical_cols

df['cut'].value_counts()

df['color'].value_counts()

df['clarity'].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt
for col in numerical_cols:
    sns.histplot(df[col],kde=True)
#     plt.show()

cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OrdinalEncoder 

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_pipeline=Pipeline(
    steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())

    ]

)

cat_pipeline=Pipeline(
    steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories]))
    ]

)

preprocessor=ColumnTransformer([
('num_pipeline',num_pipeline,numerical_cols),
('cat_pipeline',cat_pipeline,categorical_cols)
])

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.30,random_state=30)

xtrain=pd.DataFrame(preprocessor.fit_transform(xtrain),
                     columns=preprocessor.get_feature_names_out())
import pickle
pickle.dump(preprocessor,open('processor.pkl','wb'))
xtest=pd.DataFrame(preprocessor.transform(xtest),
                    columns=preprocessor.get_feature_names_out())

xtrain.head()

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

regression=LinearRegression()
regression.fit(xtrain,ytrain)

regression.coef_

regression.intercept_

slopes=regression.coef_[0]
inter=regression.intercept_[0]
# print(var)
idx=0
ans=0
for i in xtest.iloc[10,:]:
    ans+=(i*slopes[idx])
    idx+=1
ans=ans+inter
# print(f"actual : {ytest.iloc[10,:][0]} \npredicted : {ans}")

import numpy as np
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

models={
    'LinearRegression':LinearRegression(),
    'Lasso':Lasso(),
    'Ridge':Ridge(),
    'Elasticnet':ElasticNet()
}
trained=[]
for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(xtrain,ytrain)

    y_pred=model.predict(xtest)

    mae, rmse, r2_square=evaluate_model(ytest,y_pred)
    trained.append(model)
    # print(list(models.keys())[i])
    # print('Model Training Performance')
    # print("RMSE:",rmse)
    # print("MAE:",mae)
    # print("R2 score",r2_square*100)
    # print('='*35)
    # print('\n')


pickle.dump(trained[0],open('model.pkl','wb'))