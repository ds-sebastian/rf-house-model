
# rf-house-model
 Real Estate ML Model with Redfin Data Scraper
![index](https://user-images.githubusercontent.com/69488704/126915391-1bd87d88-694f-4166-9210-8b80c9300bb3.png)
## Getting Started
- Must fill-in username in password in data/resources/redfin_login.py
- Create zipcodes.csv in data/resources with headers as [Region, City, Zip Code], Zip Code being the main values for the scraper.
- ```python run.py``` to start the webapp. Access through http://localhost:45513/

## Predictions 
http://localhost:45513/predictions
- MLS Data is pulled through searching Redfin using the 'Search Redfin' Button. Do not spam or a ban might occur.
  - The app also uses the address to find latitude and longitude (inputs for the model). Searching for the same address breaks this.
- Any current data entered, creates live updates to the predicted sales price at the bottom.
- Current model uses the XGBoost algorithm and has and RMSE of ~40k
![prediction](https://user-images.githubusercontent.com/69488704/126915459-0310f600-bfa9-4841-884d-77388d4145c9.png)

## Update/Scraper 
http://localhost:45513/update
- Ability to scrape both sales data and MLS data.
- Scraping MLS data is very slow
![update](https://user-images.githubusercontent.com/69488704/126915563-81e29f08-aa2d-4587-8654-e39992debfe2.png)
