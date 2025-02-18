import asyncio
import pandas as pd
import numpy as np
import torch
import yfinance as yf
from datetime import datetime, timedelta
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load NLP Model
MODEL_NAME = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Supported languages for RoBERTa sentiment model
ROBERTA_SUPPORTED_LANGUAGES = ('ar', 'en', 'fr', 'de', 'hi', 'it', 'es', 'pt')


class DataProcessor:
    """Loads financial & Twitter data, prepares for feature extraction."""

    def __init__(self, stock_symbol: str, crypto_data_path: str, tweets_data_path: str):
        self.stock_symbol = stock_symbol

        # Load financial data
        self.crypto_df = pd.read_csv(crypto_data_path)
        self.crypto_df['Datetime'] = pd.to_datetime(self.crypto_df['Datetime'])
        self.crypto_df.set_index('Datetime', inplace=True)

        # Load Twitter data
        self.tweets_df = pd.read_csv(tweets_data_path)
        self.tweets_df.drop(columns=['i'], inplace=True)  # Drop unwanted column
        self.tweets_df['Date'] = pd.to_datetime(self.tweets_df['Date'])


class FeatureGenerator:
    """Extracts meaningful features from financial data."""

    def __init__(self):
        self.WEEK = 7 * 24
        self.MONTH = 30 * 24

    def add_original_feature(self, df, df_new):
        df_new['open'] = df['Open']
        df_new['close'] = df['Close']
        df_new['high'] = df['High']
        df_new['low'] = df['Low']
        df_new['open_1'] = df['Open'].shift(1)
        df_new['close_1'] = df['Close'].shift(1)
        df_new['high_1'] = df['High'].shift(1)
        df_new['low_1'] = df['Low'].shift(1)

    def add_average_price(self, df, df_new):
        df_new['avg_price_5'] = df['Close'].rolling(self.WEEK).mean().shift(1)
        df_new['avg_price_30'] = df['Close'].rolling(self.MONTH).mean().shift(1)
        df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']

    def generate_features(self, df) -> pd.DataFrame:
        """Generates features for ML models."""
        df_new = pd.DataFrame()
        self.add_original_feature(df, df_new)
        self.add_average_price(df, df_new)
        df_new.dropna(inplace=True)
        return df_new


class SentimentAnalyzer:
    """Processes tweets, translates if necessary, and generates sentiment scores."""

    def __init__(self):
        self.translator = Translator()

    @staticmethod
    async def translate_text(original_text: str):
        translator = Translator()
        translation = await translator.translate(original_text, dest='en')
        return translation.text, translation.src

    def preprocess_text(self, text: str) -> str:
        """Cleans tweet text by replacing usernames and links."""
        return " ".join(
            ['@user' if word.startswith('@') else 'http' if word.startswith('http') else word 
             for word in text.split()]
        )

    async def analyze_sentiment(self, text: str) -> np.ndarray:
        """Runs sentiment analysis on a tweet, translating if necessary."""
        text = self.preprocess_text(text)

        # Translate if needed
        translated_text, detected_lang = await self.translate_text(text)
        if detected_lang not in ROBERTA_SUPPORTED_LANGUAGES:
            print(f"Translated [{detected_lang}] → {translated_text}")

        # Run Sentiment Analysis
            encoded_input = tokenizer(translated_text, return_tensors='pt')
        else:
            encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        return output.logits.detach().numpy().flatten()


class DataMerger:
    """Joins tweets data with financial data for ML training."""

    @staticmethod
    async def merge_data(df_tweets: pd.DataFrame, df_exchange: pd.DataFrame) -> pd.DataFrame:
        """Merges tweet data with financial data based on timestamps."""
        
        sentiment_analyzer = SentimentAnalyzer()

        # Define final dataframe structure
        sentiment_columns = ['sentiment_1', 'sentiment_2', 'sentiment_3']
        final_columns = list(df_exchange.columns) + \
                        ['username', 'followers_count', 'text', 'retweet_count', 
                         'like_count', 'view_count', 'tags'] + sentiment_columns
        df_new = pd.DataFrame(columns=final_columns)

        for _, row in df_tweets.iterrows():
            date = row['Date'].floor('h') + pd.Timedelta(hours=1)

            if date in df_exchange.index:
                sentiment_scores = await sentiment_analyzer.analyze_sentiment(row['text'])

                # Extract exchange data
                exchange_data = df_exchange.loc[date].tolist()

                # Extract Twitter data (excluding Date column)
                tweet_data = row.drop('Date').tolist()

                # Combine all data
                combined_row = exchange_data + tweet_data + sentiment_scores.tolist()

                # Append to dataframe
                df_new.loc[date] = combined_row
            else:
                print(f"Skipping: {date} not in exchange data.")

        return df_new


async def main():
    """Main function to run the pipeline."""
    processor = DataProcessor(
        stock_symbol='BTC-USD', 
        crypto_data_path='BTC-USD.csv', 
        tweets_data_path='twitter_data.csv'
    )

    # Generate financial features
    generator = FeatureGenerator()
    df_exchange = generator.generate_features(processor.crypto_df)

    # Merge Twitter & Exchange Data
    df_final = await DataMerger.merge_data(processor.tweets_df, df_exchange)

    df_final.to_csv('tweets_BTC.csv',mode='a',header=False)


if __name__ == '__main__':
    asyncio.run(main())
