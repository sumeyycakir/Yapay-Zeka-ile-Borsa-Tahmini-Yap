import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

def download_data(op,start_date,end_date):
    df = yf.download(op,start=start_date,end=end_date,progress=False)
    return df

def model_engine(model,num):
    df = data[['Close']]

    df['preds'] = df.Close.shift(-num)

    x = df.drop(['preds'],axis=1).values
    x = scaler.fit_transform(x)
    x = x[:-num]
    x_forecast = x[-num:]
    y = df.preds.values
    y = y[:-num]

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=7)
    model.fit(x_train,y_train)
    preds = model.predict(x_test)
    print(f'Predicted with the accuracy of : {r2_score(y_test, preds)}')

    forecasted_pred = model.predict(x_forecast)
    day = 1
    for i in forecasted_pred:
        print(f'Predicted Closing Price For Day {day} is : {i}')
        day = day+1


stock = "AAPL" #Apple Hisse
today = datetime.date.today()
duration = 3000
before = today - datetime.timedelta(days=duration)
start_date = before
end_date = today

data = download_data(stock,start_date,end_date)

num = 3
scaler = StandardScaler()
engine = LinearRegression()

model_engine(engine, num)