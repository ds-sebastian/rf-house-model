
#%%
from random import choice, randint, randrange
from tqdm import tqdm
import pandas as pd
import os, shutil, glob
#from functions.logger-setup import logger
import logging
import datetime as dt
from functions.selenium_funcs import RedfinScraper
from resources.redfin_login import rf_username, rf_password
from functions.cleaning import sales_clean, mls_clean

class RedfinData():
    
    def __init__(self):
        #Logging
        logging.info('Initializing...')

        #Current Directory
        self.dir = os.path.dirname(__file__)

        self.scraper = RedfinScraper(headless=True)
        self.logged_in = False

        #Data 
        self.user_agent_list = open(os.path.join(self.dir, 'resources','user-agents.txt'), "r").read().splitlines() #User Agent List for Headers
        self.zipcodes = pd.read_csv(os.path.join(self.dir, 'resources','zipcodes.csv'), low_memory=False) #Zipcodes for data pulls
        try:
            self.sales_data = pd.read_csv(os.path.join(self.dir,'Sales_Data.csv'), low_memory=False).convert_dtypes().iloc[:, 1:] #Current Sales Data
        except:
            self.sales_data = pd.DataFrame()
        try:
            self.mls_data = pd.read_csv(os.path.join(self.dir,'MLS_Data.csv'), low_memory=False).convert_dtypes().iloc[:, 1:] #Current MLS Data
        except:
            self.mls_data = pd.DataFrame()
        try:
            self.model_data = pd.read_csv(os.path.join(self.dir,'Model_Data.csv'), low_memory=False).convert_dtypes().iloc[:, 1:] #Current MLS Data
        except:
            self.model_data = pd.DataFrame()
    
    def RedfinLogin(self, username = rf_username, password = rf_password):
        if self.logged_in == False:
            self.scraper.Login(username,password)
            self.logged_in = True
        

    def UpdateSalesData(self):
        logging.info('Scraping Sales Data')

        #Save Backup
        backup_name = 'Sales_Data-'+str(dt.datetime.now().strftime('%m-%d-%Y'))+'_'+str(randrange(1, 99999))+'.csv'
        self.sales_data.to_csv(os.path.join(self.dir,'backup',backup_name))

        lead = self.zipcodes

        self.RedfinLogin()
        scraper = self.scraper

        filename_list = []
    

        for i in tqdm(range(lead.shape[0])):
            rf_house_path = 'https://www.redfin.com/zipcode/'+str(lead['Zip Code'][i])+'/filter/include=sold-5yr'
            result = scraper.Sales_Data(rf_house_path = rf_house_path, City = lead['City'][i], ZipCode = lead['Zip Code'][i])

            if result !=False:
                filename_list.append(result)

                data = pd.read_csv(result)
                records = data.shape[0]
                sold = (data.shape[0]-sum(data['SOLD DATE'].isna()))

                logging.debug('Records: %(records)s, Sold: %(sold)s', { 'records': records, 'sold' : sold })


        for file_name in filename_list:
            shutil.move(os.path.join(self.dir, file_name),
                        os.path.join(self.dir, 'data_files'))


        #Usin
        files = [file for file in glob.glob(os.path.join(self.dir, 'data_files','*')+'.csv')]

        combined_sales = pd.concat([pd.read_csv(f) for f in files])
        combined_sales = combined_sales.reset_index(drop=True).sort_values('DAYS ON MARKET', ascending=False).drop_duplicates(['SOLD DATE','ADDRESS', 'PROPERTY TYPE','SALE TYPE', 'PRICE'])

        combined_sales.to_csv(os.path.join(self.dir, 'Sales_Data.csv'))
        self.sales_data = combined_sales.copy()


    def UpdateMLSData(self):
        logging.info('Scraping MLS Data (slow)')
        #Save Backup
        backup_name = 'MLS_Data-'+str(dt.datetime.now().strftime('%m-%d-%Y'))+'_'+str(randrange(1, 99999))+'.csv'
        self.mls_data.to_csv(os.path.join(self.dir, 'backup',backup_name))


        self.RedfinLogin()
        scraper = self.scraper

        #Initialize Variables
        user_agent_list = self.user_agent_list
        urls = self.sales_data['URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)'].unique()
        completed = self.mls_data.dropna(how='all', subset=list(self.mls_data.columns)[1:], inplace=False)['url'].unique()

        urls = list(set(urls)-set(completed)) #Unscraped urls

        mls_data = self.mls_data.copy()

        

        for url in tqdm(urls):
            try:
                mls_parse = scraper.MLS_Data(url)
                mls_data = pd.concat([mls_data,mls_parse])

            except:
                pass

        mls_data.drop_duplicates(inplace=True, ignore_index = True)
        mls_data.to_csv(os.path.join(self.dir, 'MLS_Data.csv'))
        self.mls_data = mls_data.copy()
    

    def UpdateModelData(self):
        #Save Backup
        backup_name = 'Model_Data-'+str(dt.datetime.now().strftime('%m-%d-%Y'))+'_'+str(randrange(1, 99999))+'.csv'
        self.model_data.to_csv(os.path.join(self.dir, 'backup',backup_name))

        #Clean Data

        clean_sales_data = sales_clean(self.sales_data.copy())
        clean_mls_data = mls_clean(self.mls_data.copy())

        model_data = pd.merge(clean_sales_data, clean_mls_data, on='url', how='left')
        model_data = model_data.replace(',','', regex=True)

        model_data.drop_duplicates(inplace=True, ignore_index = True)
        model_data.to_csv(os.path.join(self.dir, 'Model_Data.csv'))
        self.model_data = model_data.copy()

    def CompleteUpdate(self):
        self.UpdateSalesData()
        self.UpdateMLSData()
        self.UpdateModelData()


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
    while response not in {'1','2','3','4'}:
        response = input("Please select from above: ")

    if response == '1':  
        updater = RedfinData()
        updater.UpdateSalesData()

    elif response == '2':
        updater = RedfinData()
        updater.UpdateMLSData()

    elif response == '3':
        updater = RedfinData()
        updater.UpdateModelData()

    elif response == '4':
        updater = RedfinData()
        updater.CompleteUpdate()

    else:
        print('Error Invalid Response')
    
    logging.info('Finished')




#%%