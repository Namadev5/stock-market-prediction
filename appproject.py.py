import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import datetime
import streamlit as st
import keras 
import keras as loaded_model
loaded_model=keras.load(open("https://drive.google.com/drive/folders/1-E0O9funX6xjE3OszX5-3kNtgmccK2cJ?usp=share_link","rb"))


with st.sidebar:
    st.markdown(" TATAMOTORS Stock Market Prediction")
    user_input = st.multiselect('Please select the stock',['TATAMOTORS.NS'])

    # user_input = st.text_input('Enter Stock Name', "TATAMOTORS.NS")
    st.markdown("### Choose Date for your anaylsis")
    START = st.date_input("From",datetime.date(2018, 4, 5))
    END = st.date_input("To",datetime.date(2023, 4, 4))
    bt = st.button('Submit') 

#adding a button
if bt:

# Importing dataset------------------------------------------------------
    df = yf.download('TATAMOTORS.NS', start=START, end=END)
    plotdf, future_predicted_values =loaded_model.predict(df)
    df.reset_index(inplace = True)
    st.title('TATAMOTORS Stock Market Prediction')
    st.header("Data We collected from the source")
    st.write(df)

    tata_1=df.drop(["Adj Close"],axis=1).reset_index(drop=True)
    tata_2=tata_1.dropna().reset_index(drop=True)

    TATAMOTORS=tata_2.copy()
    TATAMOTORS['Date']=pd.to_datetime(TATAMOTORS['Date'],format='%Y-%m-%d')
    TATAMOTORS=TATAMOTORS.set_index('Date')
    st.title('EDA')
    st.write('TATAMOTORS')


# ---------------------------Graphs--------------------------------------

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Visualizations')

    st.header("Graphs")
    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    plt.plot(TATAMOTORS['Open'],color='green')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.title('Open')
    #Plot 2
    plt.subplot(2,2,2)
    plt.plot(TATAMOTORS['Close'],color='red')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Close')
    #Plot 3
    plt.subplot(2,2,3)
    plt.plot(TATAMOTORS['High'],color='green')
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.title('High')
    #Plot 4
    plt.subplot(2,2,4)
    plt.plot(TATAMOTORS['Low'],color='red')
    plt.xlabel('Date')
    plt.ylabel('Low Price')
    plt.title('Low')
    st.pyplot()

#------------------------box-plots---------------------------------

    # Creating box-plots
    st.header("Box Plots")

    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    plt.boxplot(TATAMOTORS['Open'])
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.title('Open')
    #Plot 2
    plt.subplot(2,2,2)
    plt.boxplot(TATAMOTORS['Close'])
    plt.xlabel('Date')
    plt.ylabel('Cloes Price')
    plt.title('Close')
    #Plot 3
    plt.subplot(2,2,3)
    plt.boxplot(TATAMOTORS['High'])
    plt.xlabel('Date')
    plt.ylabel('High Price')
    plt.title('High')
    #Plot 4
    plt.subplot(2,2,4)
    plt.boxplot(TATAMOTORS['Low'])
    plt.xlabel('Date')
    plt.ylabel('Low Price')
    plt.title('Low')
    st.pyplot()

#---------------------Histogram---------------------------------------

    st.header("Histogram")
    # Ploting Histogram
    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    plt.hist(TATAMOTORS['Open'],bins=50, color='green')
    plt.xlabel("Open Price")
    plt.ylabel("Frequency")
    plt.title('Open')
    #Plot 2
    plt.subplot(2,2,2)
    plt.hist(TATAMOTORS['Close'],bins=50, color='red')
    plt.xlabel("Close Price")
    plt.ylabel("Frequency")
    plt.title('Close')
    #Plot 3
    plt.subplot(2,2,3)
    plt.hist(TATAMOTORS['High'],bins=50, color='green')
    plt.xlabel("High Price")
    plt.ylabel("Frequency")
    plt.title('High')
    #Plot 4
    plt.subplot(2,2,4)
    plt.hist(TATAMOTORS['Low'],bins=50, color='red')
    plt.xlabel("Low Price")
    plt.ylabel("Frequency")
    plt.title('Low')
    st.pyplot()


#-------------------------KDE Plots-----------------------------------------

    st.header("KDE Plots")
    # KDE-Plots
    plt.figure(figsize=(20,10))
    #Plot 1
    plt.subplot(2,2,1)
    sns.kdeplot(TATAMOTORS['Open'], color='green')
    plt.title('Open')
    #Plot 2
    plt.subplot(2,2,2)
    sns.kdeplot(TATAMOTORS['Close'], color='red')
    plt.title('Close')
    #Plot 3
    plt.subplot(2,2,3)
    sns.kdeplot(TATAMOTORS['High'], color='green')
    plt.title('High')
    #Plot 4
    plt.subplot(2,2,4)
    sns.kdeplot(TATAMOTORS['Low'], color='red')
    plt.title('Low')
    st.pyplot()


    st.header('Years vs Volume')
    st.line_chart(TATAMOTORS['Volume'])


#-------------------Finding long-term and short-term trends---------------------

    st.title('Finding long-term and short-term trends')
    TATAMOTORS_ma=TATAMOTORS.copy()
    TATAMOTORS_ma['30-day MA']=TATAMOTORS['Close'].rolling(window=30).mean()
    TATAMOTORS_ma['200-day MA']=TATAMOTORS['Close'].rolling(window=200).mean()

    st.write(TATAMOTORS_ma)


    st.subheader('Stock Price vs 30-day Moving Average')
    plt.plot(TATAMOTORS_ma['Close'],label='Original data')
    plt.plot(TATAMOTORS_ma['30-day MA'],label='30-MA')
    plt.legend()
    plt.title('Stock Price vs 30-day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot()


    st.subheader('Stock Price vs 200-day Moving Average')
    plt.plot(TATAMOTORS_ma['Close'],label='Original data')
    plt.plot(TATAMOTORS_ma['200-day MA'],label='200-MA')
    plt.legend()
    plt.title('Stock Price vs 200-day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot()

    df1 = pd.DataFrame(future_predicted_values)
    st.markdown("### Next 30 days forecast")
    df1.rename(columns={0: "Predicted Prices"}, inplace=True)
    st.write(df1)

    st.markdown("### Original vs predicted close price")
    fig= plt.figure(figsize=(20,10))
    sns.lineplot(data=plotdf)
    st.pyplot(fig)
    
    
else:
    #displayed when the button is unclicked
    st.write('Please click on the submit button to get the EDA ans Prediction') 
