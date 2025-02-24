import asyncio
import time
import datetime
import os
from twikit import Client, TooManyRequests
from configparser import ConfigParser
import pandas as pd
import random
import time

class TwitterScraper:
    def __init__(self, query, min_tweets, new_account=False , tweet_count=0):
        self.query = query
        self.min_tweets = min_tweets
        self.new_account = new_account
        self.client = Client(language='en-US')
        self.all_tweets_data = []
        self.tweet_count = tweet_count
        self.config = self.get_config()
        #self.username = self.config['X']['username']
        #self.password = self.config['X']['password']
        #self.email = self.config['X']['email']
        self.tweets = None
        
    def get_config(self):
        """Load configuration from config.ini."""
        config = ConfigParser()
        config.read('config.ini')
        return config

    async def login(self):
        """Handles login and cookie management."""
        if self.new_account and os.path.exists('Twitter_scraper/cookies.json'):
            os.remove('cookies.json')

        if self.new_account:
            await self.client.login(auth_info_1=self.username, auth_info_2=self.email, password=self.password)
            self.client.save_cookies('cookies.json')
            print("Cookies saved")
        else:
            self.client.load_cookies('Twitter_scraper/cookies.json')

    async def fetch_tweets(self):
        """Fetch tweets using Twikit and handle rate limits."""
        try:
            if not self.tweets:
                tweets = await self.client.search_tweet(self.query, product='Top')
            else:
                
                tweets = await tweets.next()
        except TooManyRequests as e:

            print("Rate limit exceeded. Retrying after 60 seconds...")
            time.sleep(60 + random.uniform(5,10))
            tweets = await self.client.search_tweet(self.query, product='Top',count=100)

        if not tweets:
            print("No tweets found. Try a different query or check authentication.")
            return []
        else:
            tweet_data = []
            for tweet in tweets:
                self.tweet_count += 1
                tweet_info = [
                    tweet.user.name,
                    tweet.user.followers_count,
                    tweet.text.strip().replace('\n','') + ' ',
                    tweet.retweet_count,
                    tweet.favorite_count,
                    tweet.view_count,
                    tweet.hashtags,
                    tweet.created_at_datetime,
                ]
                tweet_data.append(tweet_info)
                print(tweet_info)

            return tweet_data

    async def scrape_tweets(self):
        """Main function to scrape tweets until the minimum count is reached."""
        await self.login()

        while self.tweet_count < self.min_tweets:
            wait_time = random.uniform(5,10)
            print(f'Waiting for {wait_time:.3f} then fetching tweets')
            time.sleep(wait_time)
            print(f'Fetching tweets now -- {datetime.datetime.now()}')
            new_tweets = await self.fetch_tweets()
            if new_tweets == []: break
            self.all_tweets_data.extend(new_tweets)


        print(f"Collected {self.tweet_count} tweets.")
        return self.all_tweets_data

def Mine_tweets(queries,min_tweets):

    for query in queries:
        scraper = TwitterScraper(query=query, min_tweets=min_tweets, new_account=False)
        #self.tweet_count,tweet.user.name,tweet.user.followers_count,tweet.text,tweet.retweet_count,tweet.favorite_count,tweet.created_at
        all_tweets = asyncio.run(scraper.scrape_tweets())
        
        df = pd.DataFrame(all_tweets, columns=[ "username", "followers_count", "text", "retweet_count", "like_count" ,'view_count','tags',"Date"])
        df.to_csv('twitter_data.csv',mode='a',header=False)
        #try:tweet_count = df.iloc[-1]['tweet_count']
        #except IndexError:
        #    tweet_count = pd.read_csv('twitter_data.csv')['tweet_count'].iloc[-1]

        #print(tweet_count)




# Usage Example
#['Bitcoin', 'BTC', 'Ethereum', 'ETH', 'Tether', 'USDT', 'XRP', 'XRP', 'Solana', 'SOL', 'Binance Coin', 'BNB', 'USD Coin', 'USDC', 'Dogecoin', 'DOGE', 'Cardano', 'ADA', 'TRON', 'TRX', 'Wrapped Bitcoin', 'WBTC', 'Chainlink', 'LINK', 'Stellar Lumens', 'XLM', 'Avalanche', 'AVAX', 'Toncoin', 'TON', 'Sui', 'SUI', 'Unus Sed Leo', 'LEO', 'SHIBA INU', 'SHIB', 'Hedera Hashgraph', 'HBAR', 'USDS', 'USDS', 'Litecoin', 'LTC', 'Hyperliquid', 'HYPE', 'Bitget Token', 'BGB', 'Polkadot', 'DOT', 'Bitcoin Cash', 'BCH', 'Ethena USDe', 'USDE', 'MANTRA DAO', 'OM', 'Wrapped Beacon ETH', 'WBETH', 'Dai', 'DAI', 'Uniswap', 'UNI', 'Wrapped eETH', 'WEETH', 'Ondo', 'ONDO', 'Monero', 'XMR', 'Pepe', 'PEPE', 'NEAR Protocol', 'NEAR', 'Mantle', 'MNT', 'Aave', 'AAVE', 'OFFICIAL TRUMP', 'TRUMP', 'Internet Computer', 'ICP', 'Aptos', 'APT', 'Ethereum Classic', 'ETC', 'OKB', 'OKB', 'Bittensor', 'TAO', 'VeChain', 'VET', 'Polygon Ecosystem Token', 'POL', 'Cronos', 'CRO', 'Algorand', 'ALGO', 'Kaspa', 'KAS', 'Render', 'RENDER', 'Jupiter', 'JUP', 'Filecoin', 'FIL', 'Arbitrum', 'ARB', 'Gatechain Token', 'GT', 'First Digital USD', 'FDUSD', 'Fasttoken', 'FTN', 'Cosmos', 'ATOM', 'DeXe', 'DEXE', 'Fetch.AI', 'FET', 'Render Token', 'RNDR', 'Binance Staked SOL', 'BNSOL', 'Ethena', 'ENA', 'KuCoin Token', 'KCS', 'Celestia', 'TIA', 'Raydium', 'RAY', 'Optimism', 'OP', 'Lido DAO Token', 'LDO', 'XDC Network', 'XDC', 'Immutable X', 'IMX', 'Bonk', 'BONK', 'Theta Network', 'THETA', 'Stacks', 'STX', 'Injective', 'INJ', 'Flare', 'FLR', 'Movement', 'MOVE', 'The Graph', 'GRT', 'Worldcoin', 'WLD']

if __name__ == "__main__":
    queries = ['Bitcoin min_faves:500 lang:pt until:2023-03-01 since:2023-02-01', 'Bitcoin min_faves:500 lang:pt until:2023-04-01 since:2023-03-01', 'Bitcoin min_faves:500 lang:pt until:2023-05-01 since:2023-04-01', 'Bitcoin min_faves:500 lang:pt until:2023-06-01 since:2023-05-01', 'Bitcoin min_faves:500 lang:pt until:2023-07-01 since:2023-06-01', 'Bitcoin min_faves:500 lang:pt until:2023-08-01 since:2023-07-01', 'Bitcoin min_faves:500 lang:pt until:2023-09-01 since:2023-08-01', 'Bitcoin min_faves:500 lang:pt until:2023-10-01 since:2023-09-01', 'Bitcoin min_faves:500 lang:pt until:2023-11-01 since:2023-10-01', 'Bitcoin min_faves:500 lang:pt until:2023-12-01 since:2023-11-01', 'Bitcoin min_faves:500 lang:pt until:2024-01-01 since:2023-12-01', 'Bitcoin min_faves:500 lang:pt until:2024-02-01 since:2024-01-01', 'Bitcoin min_faves:500 lang:pt until:2024-03-01 since:2024-02-01', 'Bitcoin min_faves:500 lang:pt until:2024-04-01 since:2024-03-01', 'Bitcoin min_faves:500 lang:pt until:2024-05-01 since:2024-04-01', 'Bitcoin min_faves:500 lang:pt until:2024-06-01 since:2024-05-01', 'Bitcoin min_faves:500 lang:pt until:2024-07-01 since:2024-06-01', 'Bitcoin min_faves:500 lang:pt until:2024-08-01 since:2024-07-01', 'Bitcoin min_faves:500 lang:pt until:2024-09-01 since:2024-08-01', 'Bitcoin min_faves:500 lang:pt until:2024-10-01 since:2024-09-01', 'Bitcoin min_faves:500 lang:pt until:2024-11-01 since:2024-10-01', 'Bitcoin min_faves:500 lang:pt until:2024-12-01 since:2024-11-01', 'Bitcoin min_faves:500 lang:pt until:2025-01-01 since:2024-12-01']
    Mine_tweets( queries, 1)
    
