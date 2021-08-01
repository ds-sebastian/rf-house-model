
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
import datetime as dt
from time import sleep
from random import randrange  # , choice
from urllib.request import urlretrieve
import pandas as pd
import re
import json
import logging


#%%----------------------------------------------------------------


#%%
class RedfinScraper():

    def __init__(self, headless=True):
        logging.debug('Initializing webdriver...')

        #Chrome Options
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)
        if headless == True:
            options.add_argument("--headless")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        options.add_argument("--incognito")
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        #Chrome Driver
        self.driver = webdriver.Chrome(
            executable_path=ChromeDriverManager().install(), options=options)

    #Sign on to Redfin
    def Login(self, username, password):
        driver = self.driver

        logging.info('Signing into Redfin')
        driver.get('https://www.redfin.com/login-v2')

        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '.submit-button'))
        )

        signon = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.NAME, 'emailInput')))
        signon.send_keys(username)
        signon.submit()

        signon = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.NAME, 'passwordInput')))
        signon.send_keys(password)
        signon.submit()

        sleep(5)  # wait to login

    #Get CSVs
    def Sales_Data(self, rf_house_path, City, ZipCode):
        '''Downloads CSVs based on resources/zipcodes.csv'''

        driver = self.driver

        logging.debug(str(ZipCode)+":\n"+str(rf_house_path))
        #time.sleep(2)

        try:
            driver.get(rf_house_path)

            download_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, 'download-and-save'))
            )

            download_link = download_button.get_attribute('href')
            download_link = re.sub(
                'num_homes=350', 'num_homes=10000', download_link)

            logging.debug("Data Link:"+str(download_link))

            driver.get(download_link)
            #time.sleep(2)
            date = dt.datetime.now().strftime('%m-%d-%Y')
            filename = str(date)+'_'+str(City)+'_'+str(ZipCode) + \
                '_'+str(randrange(1, 99999))+'.csv'
            urlretrieve(download_link, filename)
            #time.sleep(2)

            return filename

        except:
            logging.debug('...Error Obtaining Data...')
            return False

    def MLS_Data(self, url):
        '''Uses the URLs from Sales Data to get MLS & School data (from Redfin's API) on each individual page (slow)'''
        driver = self.driver

        #sleep(randint(0, 2))
        property_id = re.search('([^\/]+$)', url)[0]

        api_url = 'https://www.redfin.com/stingray/api/home/details/belowTheFold?propertyId=' + \
            property_id+'&accessLevel=1&pageType=3'

        driver.get(api_url)

        text = driver.find_element(By.TAG_NAME, "body").text

        final = self.mls_parse(url, text)

        return final
    
    @staticmethod
    def mls_parse(url, text):
        url_df = pd.DataFrame({'url': url}, index=[0])
        #######          MLS DATA               #############
        try:
            mls_regex = re.findall(r'{"amenityName":.*?\}', text)
            mls_regex = [json.loads(x) for x in mls_regex]
            mls_data = pd.DataFrame(mls_regex)[
                ['referenceName', 'amenityValues']]
            mls_data['amenityValues'] = mls_data['amenityValues'].map(
                lambda x: x[0])
            mls_data = mls_data.set_index('referenceName').T.reset_index()
        except:
            mls_data = pd.DataFrame()
        ########         BASIC DATA             #############
        try:
            basic_data = pd.DataFrame(json.loads(re.findall(
                '{"basicInfo":({.*?\})', text)[0]), index=[0])
        except:
            basic_data = pd.DataFrame()
        #######          TAX DATA               #############
        try:
            tax_data = pd.DataFrame(json.loads(re.findall(
                '"taxInfo":({.*?\})', text)[0]), index=[0])
        except:
            tax_data = pd.DataFrame()
        #######          SCHOOL DATA            #############
        try:
            school_regex = re.findall(
                r'({"servesHome":.*?),"schoolReviews"', text)
            school_regex = [str(x) + r'}' for x in school_regex]
            school_data = pd.DataFrame([json.loads(x) for x in school_regex])
            try:
                parentRating = school_data['parentRating'].astype(float).mean()
            except:
                parentRating = ''
            try:
                schooldisance = school_data['distanceInMiles'].astype(
                    float).mean()
            except:
                schooldisance = ''
            try:
                servesHome = school_data['servesHome'].sum(
                )/len(school_data.index)
            except:
                servesHome = ''
            try:
                schoolchoice = school_data['isChoice'].sum(
                )/len(school_data.index)
            except:
                schoolchoice = ''
            try:
                numberOfStudents = school_data['numberOfStudents'].astype(
                    float).mean()
            except:
                numberOfStudents = ''
            try:
                school_score = school_data['greatSchoolsRating'].astype(
                    float).mean()
            except:
                school_score = ''
            try:
                student_teacher_ratio = school_data['studentToTeacherRatio'].astype(
                    float).mean()
            except:
                student_teacher_ratio = ''
            try:
                review_nums = school_data['numReviews'].astype(float).mean()
            except:
                review_nums = ''
            school = pd.DataFrame({'parentRating': parentRating,
                                   'schooldisance': schooldisance,
                                   'servesHome': servesHome,
                                   'schoolchoice': schoolchoice,
                                   'numberOfStudents': numberOfStudents,
                                   'school_score': school_score,
                                   'student_teacher_ratio': student_teacher_ratio,
                                   'review_nums': review_nums}, index=[0])
        except:
            school = pd.DataFrame()
        try:
            final = pd.concat(
                [url_df, basic_data, mls_data, school, tax_data], axis=1)
        except:
            final = pd.DataFrame()

        return final
