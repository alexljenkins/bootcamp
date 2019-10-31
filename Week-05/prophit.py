
#create ds and y columns

from fbprophet import Prophet
m= Prophet()

m.fit(df_climate.tail(1000))

future = m.make_future_dataframe(periods = 120, freq='M', include_history = False)

forecast = m.predict(future)

forecast[['ds','yhat','yhatlower','yhatupper']] = forecast.columns???#not sure what this is meant to equal

plot_components = m.plt_components(forecast)
