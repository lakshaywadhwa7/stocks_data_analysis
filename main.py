import nasdaqdatalink
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')

class Factory:
    ##function to load the data
    def load_data(self):
        data = nasdaqdatalink.get("FSE/BDT_X")
        return data
    ##function to preprocess monthly data
    def preprocess_data_monthly_average(self, data):
        data_price = data[['Close']]
        data_price['Date'] = data_price.index
        data_price.reset_index(drop=True, inplace=True)
        data_price['Date'] = data_price['Date'].astype(str)
        data_price = data_price.dropna()
        ##converting records to tuple
        records = data_price.to_records(index=False)
        ##converting records to list of tuples
        records = list(records)
        dic = {}
        for val, date in records:
            # split the date string at '-' and assign the first  2 items to  year,month in list of records
            year, month = date.split('-')[:2]
            # now check if (month,year) is there in the dict
            if (month, year) not in dic:
                # if the tuple was not found then initialise one with an empty list
                dic[month, year] = []
            # appending the value to the (month,year) key
            dic[month, year].append(val)  

        #  Iterate over key,value items to calculate average
        lis = []
        for key, val in dic.items():
            new_key = "-".join(key)
            avg = sum(val) / len(val)
            lis.append((avg, new_key))
        average_monthly_data_df = pd.DataFrame(lis, columns=['value', 'date'])

        average_monthly_data_df['date'] = pd.to_datetime(average_monthly_data_df['date'], format='%m-%Y')
        return average_monthly_data_df
     #  calculate rolling average for window(number of days) defined
    def preprocess_data_rolling_average(self, data, window):
        rolling_data = data['Close'].rolling(window=window).mean()
        rolling_data = pd.DataFrame(rolling_data, columns=['Close'])
        return rolling_data
    
    #preprocess data based on streak (increase)
    def preprocess_data_price_streak_increase(self, data):
        data = data[['Close']]
        data['Date'] = data.index
        data.reset_index(drop=True, inplace=True)
        data['Date'] = data['Date'].astype(str)
        #data = data.dropna()
        data['streak_up'] = data['Close'].rolling(5).apply(lambda x: np.all(np.diff(x) > 0)).astype('boolean')
        data['streak_up'] = data['streak_up'].fillna(False)
        #data['streak_up'] = data['streak_up'].fillna(False)
        return data
    #preprocess data based on streak (decrease)
    def preprocess_data_price_streak_decrease(self, data):
        data = data[['Close']]
        data['Date'] = data.index
        data.reset_index(drop=True, inplace=True)
        data['Date'] = data['Date'].astype(str)
        # data = data.dropna()
        data['streak_down'] = data['Close'].rolling(4).apply(lambda x: np.all(np.diff(x) < 0)).astype('boolean')
        data['streak_down'] = data['streak_down'].fillna(False)
        return data
    
    ##calculate the regression 
    def regression(self, data,n):
        data = data[['Close']]
        data=data.dropna()
        data=data[-n:]
        data.reset_index(drop=True, inplace=True)
        data['Time'] = np.arange(len(data.index))
        # Training data
        X = data.loc[:, ['Time']]  # features
        y = data.loc[:, 'Close']
        model = LinearRegression()
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=X.index)
        return y, y_pred

    def plot_regression(self, y,y_pred,n):
        ax = y.plot(label='Closing price in euros')
        ax = y_pred.plot(ax=ax, linewidth=3)
        ax.set_title('Regression plot');
        plt.savefig('last_'+str(n)+'_regression.png')
        plt.close()

    def plot_prices(self, data,seven_rolling,thirty_rolling,ninty_rolling):
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(data.index, data['Close'], label='Price')
        ax.plot(seven_rolling.index, seven_rolling['Close'], label='Span 7-days')
        ax.plot(thirty_rolling.index, thirty_rolling['Close'], label='Span 30-days')
        ax.plot(ninty_rolling.index, ninty_rolling['Close'], label='Span 90-days')
        ax.legend(loc='best')
        ax.set_ylabel('Price in Euros')
        plt.savefig( 'plot.png' )
        ##ax.xaxis.set_major_formatter(my_year_month_fmt)

    def plot_monthly_average(self,data):
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(data['date'], data['value'], label='Monthly average')
        ax.legend(loc='best')
        ax.set_ylabel('Monthly Average Price in Euros')
        plt.savefig("average_monthly_data.png")

    def plot_prices_increase_streak(self, data, seven_rolling, thirty_rolling, ninty_rolling,streak_data,column_streak,color):
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(data.index, data['Close'], label='Price')
        ax.plot(seven_rolling.index, seven_rolling['Close'], label='Span 7-days')
        ax.plot(thirty_rolling.index, thirty_rolling['Close'], label='Span 30-days')
        ax.plot(ninty_rolling.index, ninty_rolling['Close'], label='Span 90-days')
        ax.fill_between(streak_data['Date'], 0, 1, where=streak_data[column_streak], color=color)
        ax.legend(loc='best')
        ax.set_ylabel('Price in Euros')
        plt.savefig('increase_streak_5_days.png')

    def plot_prices_decrease_streak(self, data, seven_rolling, thirty_rolling, ninty_rolling,streak_data,column_streak,color):
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(data.index, data['Close'], label='Price')
        ax.plot(seven_rolling.index, seven_rolling['Close'], label='Span 7-days')
        ax.plot(thirty_rolling.index, thirty_rolling['Close'], label='Span 30-days')
        ax.plot(ninty_rolling.index, ninty_rolling['Close'], label='Span 90-days')
        ax.fill_between(streak_data['Date'], 0, 1, where=streak_data[column_streak], color=color)
        ax.legend(loc='best')
        ax.set_ylabel('Price in Euros')
        plt.savefig('decrease_streak_4_days.png')




class Main:
    def __init__(self, factory):
        self.factory = factory
        self.data=factory.load_data(self)
        self.seven_day_rolling=self.factory.preprocess_data_rolling_average(self,self.data,7)
        self.thirty_day_rolling = self.factory.preprocess_data_rolling_average( self,self.data, 30)
        self.ninty_day_rolling = self.factory.preprocess_data_rolling_average( self,self.data, 90)
        self.five_day_streak_up=self.factory.preprocess_data_price_streak_increase(self,self.data)
        self.four_day_streak_down = self.factory.preprocess_data_price_streak_decrease(self,self.data)
        self.monthly_average_data=self.factory.preprocess_data_monthly_average(self,self.data)
        self.regression_90,self.regression_90_pred = self.factory.regression(self, self.data,90)
        self.regression_30,self.regression_30_pred = self.factory.regression(self, self.data, 30)
        self.regression_7,self.regression_7_pred = self.factory.regression(self, self.data, 7)

    def plot_prices_main(self):
        self.factory.plot_prices(self,self.data,self.seven_day_rolling,self.thirty_day_rolling,self.ninty_day_rolling)
    def plot_prices_main_increase_streak(self):
        self.factory.plot_prices_increase_streak(self,self.data,self.seven_day_rolling,self.thirty_day_rolling,self.ninty_day_rolling,self.five_day_streak_up,'streak_up','tab:red')

    def plot_prices_main_decrease_streak(self):
        self.factory.plot_prices_decrease_streak(self,self.data,self.seven_day_rolling,self.thirty_day_rolling,self.ninty_day_rolling,self.four_day_streak_down,'streak_down','tab:brown')

    def plot_prices_monthly_average(self):
        self.factory.plot_monthly_average(self,self.monthly_average_data)
    def regression_plot_90(self):
        self.factory.plot_regression(self,self.regression_90,self.regression_90_pred,90)
    def regression_plot_30(self):
        self.factory.plot_regression(self,self.regression_30,self.regression_30_pred,30)
    def regression_plot_7(self):
        self.factory.plot_regression(self,self.regression_7,self.regression_7_pred,7)






# Press the green button in the gutter to run the script.
Main(Factory).regression_plot_90()
Main(Factory).regression_plot_30()
Main(Factory).regression_plot_7()
Main(Factory).plot_prices_main()
Main(Factory).plot_prices_main_increase_streak()
Main(Factory).plot_prices_main_decrease_streak()
Main(Factory).plot_prices_monthly_average()




