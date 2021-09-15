
# Real Estate ML Model with Data Scraper
![index](https://user-images.githubusercontent.com/69488704/133361015-56825aa8-8949-45fa-80c5-d6627b640af2.png)
## Getting Started
- Must fill-in username in password in data/resources/redfin_login.py
- Create zipcodes.csv in data/resources with headers as [Region, City, Zip Code], Zip Code being the main values for the scraper.
- ```python run.py``` to start the webapp. Access through http://localhost:45513/

## Predictions 
http://localhost:{port}/predictions
- MLS Data is pulled through searching Redfin using the 'Search Redfin' Button. Do not spam or a ban might occur.
  - The app also uses the address to find latitude and longitude (inputs for the model). Searching for the same address breaks this.
- Any current data entered, creates live updates to the predicted sales price at the bottom.
- Current model uses the XGBoost algorithm and has and RMSE of ~45k
- Model is built in `ml-model.py` and saved as `.joblib` files for use in the predictions.

![prediction](https://user-images.githubusercontent.com/69488704/133361078-69bae561-9f20-4443-9f67-0f7978c47bd7.png)

## Update/Scraper 
- To update the data navigate to /data and run `python updater.py`
- Options will appear to scrape both sales data, MLS data, and Model (combined) data.
- NOTE: Scraping MLS data is very slow (~ 1-2 seconds per address)


## Future Plans
- Include more features and switch to neural network
- List Prices would significantly improve the model
