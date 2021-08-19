
#%%
import os
import shutil
import glob
from random import randrange
from tqdm import tqdm
import pandas as pd
#from functions.logger-setup import logger
import logging
import datetime as dt
from functions.selenium_funcs import RedfinScraper
from resources.redfin_login import rf_username, rf_password
from functions.cleaning import sales_clean, mls_clean, merge_data

class RedfinData():

    def __init__(self):
        '''
        Pulls in inital data saved in CSVs
        Sets directory
        '''
        #Logging
        logging.info('Initializing...')

        #Current Directory
        self.dir = os.path.dirname(__file__)
        print(f'Dir set to: {self.dir}')

        self.logged_in = False

        #Data
        self.zipcodes = pd.read_csv(os.path.join(
            self.dir, 'resources', 'zipcodes.csv'), low_memory=False)  # Zipcodes for data pulls
        try:
            self.sales_data = pd.read_csv(os.path.join(
                self.dir, 'Sales_Data.csv'), low_memory=False).convert_dtypes()  # Current Sales Data
        except:
            self.sales_data = pd.DataFrame()
        try:
            self.mls_data = pd.read_csv(os.path.join(
                self.dir, 'MLS_Data.csv'), low_memory=False).convert_dtypes()  # Current MLS Data
        except:
            self.mls_data = pd.DataFrame()
        try:
            self.model_data = pd.read_csv(os.path.join(
                self.dir, 'Model_Data.csv'), low_memory=False).convert_dtypes()  # Current MLS Data
        except:
            self.model_data = pd.DataFrame()

    def redfin_login(self, username=rf_username, password=rf_password):
        '''Starts selenium browser and logs into Redfin'''
        if self.logged_in == False:
            self.scraper = RedfinScraper(headless=True)
            self.scraper.Login(username, password)
            self.logged_in = True

    def update_sales_data(self):
        '''Scrapes all sold homes in the last 5 years and combines with current data'''
        logging.info('Scraping Sales Data')

        #Save Backup
        backup_name = 'Sales_Data-' + \
            str(dt.datetime.now().strftime('%m-%d-%Y')) + \
            '_'+str(randrange(1, 99999))+'.csv'
        self.sales_data.to_csv(os.path.join(self.dir, 'backup', backup_name),index=False)

        lead = self.zipcodes

        self.redfin_login()
        scraper = self.scraper

        filename_list = []

        for i in tqdm(range(lead.shape[0])):
            rf_house_path = 'https://www.redfin.com/zipcode/' + \
                str(lead['Zip Code'][i])+'/filter/include=sold-5yr'
            result = scraper.Sales_Data(
                rf_house_path=rf_house_path, City=lead['City'][i], ZipCode=lead['Zip Code'][i])

            if result != False:
                filename_list.append(result)

                data = pd.read_csv(result)
                records = data.shape[0]
                sold = (data.shape[0]-sum(data['SOLD DATE'].isna()))

                logging.debug('Records: %(records)s, Sold: %(sold)s', {
                              'records': records, 'sold': sold})

        for file_name in filename_list:
            shutil.move(os.path.join(self.dir, file_name),
                        os.path.join(self.dir, 'data_files'))

        #Usin
        files = [file for file in glob.glob(
            os.path.join(self.dir, 'data_files', '*')+'.csv')]

        combined_sales = pd.concat([pd.read_csv(f) for f in files])
        combined_sales = combined_sales.reset_index(drop=True).sort_values('DAYS ON MARKET', ascending=False).drop_duplicates([
            'SOLD DATE', 'ADDRESS', 'PROPERTY TYPE', 'SALE TYPE', 'PRICE'])

        combined_sales.to_csv(os.path.join(self.dir, 'Sales_Data.csv'), index=False)
        self.sales_data = combined_sales.copy()

    def update_mls_data(self):
        '''Scrapes MLS data for ONLY new URLs (since it's very slow)'''

        logging.info('Scraping MLS Data (slow)')
        #Save Backup
        backup_name = 'MLS_Data-' + \
            str(dt.datetime.now().strftime('%m-%d-%Y')) + \
            '_'+str(randrange(1, 99999))+'.csv'

        self.mls_data.to_csv(os.path.join(self.dir, 'backup', backup_name),index=False)

        self.redfin_login()
        scraper = self.scraper

        #Initialize Variables
        sales_data = self.sales_data.copy()
        mls_data = self.mls_data.copy()
        urls = sales_data['URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)'].unique()

        if 'url' in mls_data.columns:
            mls_data = mls_data.loc[mls_data.count(1)[::-1].groupby(mls_data['url'][::-1]).idxmax()]
            completed = mls_data.dropna(how='all', subset=list(
                mls_data.columns)[1:], inplace=False)['url'].unique()
            urls = list(set(urls)-set(completed))  # Unscraped urls

        try: 
            for url in tqdm(urls):
                try:
                    mls_parsed = scraper.MLS_Data(url)
                    mls_data = pd.concat([mls_data, mls_parsed])

                except:
                    logging.debug(f'Error with URL: {url}')
                    pass
        except (KeyboardInterrupt, SystemExit):
            logging.info('Interrupted. Saving Data...')


        mls_data.drop_duplicates(inplace=True, ignore_index=True)
        mls_data.to_csv(os.path.join(self.dir, 'MLS_Data.csv'),index=False)
        self.mls_data = mls_data.copy()

    def update_model_data(self):
        '''Cleans and combines the mls data and the sales data for model development'''
        logging.info('Cleaning Model Data')

        #Save Backup
        backup_name = 'Model_Data-' + \
            str(dt.datetime.now().strftime('%m-%d-%Y')) + \
            '_'+str(randrange(1, 99999))+'.csv'
        self.model_data.to_csv(os.path.join(self.dir, 'backup', backup_name),index=False)

        #Clean Data
        clean_sales_data = sales_clean(self.sales_data.copy())
        clean_mls_data = mls_clean(self.mls_data.copy())

        #Combine
        model_data = merge_data(clean_sales_data,clean_mls_data)

        model_data.to_csv(os.path.join(self.dir, 'Model_Data.csv'),index=False)
        self.model_data = model_data.copy()

    def complete_update(self):
        '''runs through all updates'''
        self.update_sales_data()
        self.update_mls_data()
        self.update_mls_data()

    def exit_browser(self):
        '''quits all browsers'''
        if self.logged_in == True:
            logging.info('Exiting Browser...')
            self.scraper.driver.quit()


#%%
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info('Running Updated...')

    print("Choose Update Method (Use VPN if banned):")
    print("[1] Update Sales Data")
    print("[2] Update MLS Data")
    print("[3] Update Model Data")
    print("[4] Update ALL Data")
    response = None
    while response not in {'1', '2', '3', '4'}:
        response = input("Please select from above: ")

    if response == '1':
        updater = RedfinData()
        updater.update_sales_data()
        updater.exit_browser()

    elif response == '2':
        updater = RedfinData()
        updater.update_mls_data()
        updater.exit_browser()

    elif response == '3':
        updater = RedfinData()
        updater.update_model_data()

    elif response == '4':
        updater = RedfinData()
        updater.complete_update()
        updater.exit_browser()

    else:
        print('Error Invalid Response')

    logging.info('Finished')


#%%
