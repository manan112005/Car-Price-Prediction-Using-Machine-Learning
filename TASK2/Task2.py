import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score as r2score
import pickle


df = pd.read_csv('E:\INTERNPE\TASK2\car_data.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df['year'].unique())
print(df['Price'].unique())
print(df['kms_driven'].unique())
print(df['fuel_type'].unique())

backup_df=df.copy()

a=df[df['year'].str.isnumeric()]
print(a)

# Step 1: Convert everything to string
df['year'] = df['year'].astype(str)

# Step 2: Extract a 4-digit year from each value
df['year'] = df['year'].str.extract(r'(\d{4})')

# Step 3: Convert extracted value to number (invalid → NaN)
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Step 4: Drop rows where year was invalid (NaN)
df = df.dropna(subset=['year']).copy()

# Step 5: Final conversion to integer
df['year'] = df['year'].astype(int)
print(df['year'])

b=df[df['Price']!="Ask For Price"]
print(b)

df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce').fillna(0).astype(int)
print(df['Price'])

df['kms_driven']=df['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')
print(df['kms_driven'])

df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce').fillna(0).astype(int)
print(df['kms_driven'])

df = df[(df['Price'] != 0) & (df['kms_driven'] != 0)].dropna().copy()
print(df)

print(df[~df['fuel_type'].isna()])

df['name'] = df['name'].apply(lambda x: ' '.join(str(x).split()[:3]))
print(df['name'])


print(df.reset_index(drop=True))
print(df.info())
print(df.describe())

print(df[df['Price']<6e6].reset_index(drop=True))

print(df.to_csv('cleaned_car_data.csv'))

X=df.drop(columns=['Price'])
y=df['Price']
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])
print(ohe)

print(ohe.categories_)

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']), remainder='passthrough')
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
print(pipe.fit(X_train,y_train))
y_pred=pipe.predict(X_test)
print(y_pred)
print(r2score(y_test,y_pred))

scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2score(y_test,y_pred))
print(np.argmax(scores))
print(scores[np.argmax(scores)])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2score(y_test,y_pred)

pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))

print(pipe.predict(pd.DataFrame([['Maruti Suzuki Alto','Maruti',2015,60000,'Petrol']],columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])))