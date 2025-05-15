import requests
from bs4 import BeautifulSoup as bs
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium import webdriver
import pandas as pd
import time
import re

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------- CONFIGURATION ----------------------

URL = "https://www.footballtransfers.com/us/players/uk-premier-league"
NUM_PAGES = 22
ENCODING = 'utf-8-sig'
CSV_PATH = './Task_1/results.csv'

# ---------------------- DRIVER SETUP ----------------------

def driver_setup():
    """Return an Edge driver that ignores SSL-certificate problems."""
    opts = EdgeOptions()
    # opts.add_argument("--headless")   # Edge ≥118
    # opts.add_argument("--ignore-certificate-errors")
    # opts.add_argument("--ignore-ssl-errors=yes")
    # opts.add_argument("--allow-insecure-localhost")
    opts.set_capability("acceptInsecureCerts", True)   # <-- key line

    service = Service(EdgeChromiumDriverManager().install())
    driver  = webdriver.Edge(service=service, options=opts)
    driver.set_page_load_timeout(60)       # fail fast on bad links
    return driver

# ---------------------- DATA LOADING ----------------------

def load_data():
    try:
        df = pd.read_csv(CSV_PATH, encoding=ENCODING)
        filtered = df[df['minutes'] > 900].copy()
        print(f"✅ Loaded {len(filtered)} players with more than 900 minutes played.")
        return df, filtered
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None


# ---------------------- SCRAPER ----------------------

def scrape_etv(filtered_df):
    player_etv = {name: '' for name in filtered_df['player']}
    driver = driver_setup()
    driver.get(URL)

    names = []
    values = []
    page = 0

    while page < NUM_PAGES:
        time.sleep(1)
        soup = bs(driver.page_source, 'html.parser')
        table = soup.find('table', class_='table table-hover no-cursor table-striped leaguetable mvp-table similar-players-table mb-0')
        if not table:
            print("⚠️ No player table found.")
            break

        name_tags = table.find_all('div', class_='text')
        value_tags = table.find_all('span', class_='player-tag')

        for n in name_tags:
            a = n.find('a')
            if a:
                names.append(a.get('title'))
        for v in value_tags:
            values.append(v.text.strip())

        try:
            page += 1
            next_btn = driver.find_element(By.CLASS_NAME, 'pagination_next_button')
            next_btn.click()
        except:
            print("🔚 Reached the last page.")
            break

    driver.quit()

    for i in range(min(len(names), len(values))):
        if names[i] in player_etv:
            player_etv[names[i]] = values[i]

    return player_etv


# ---------------------- FEATURE HANDLING ----------------------

def clean_and_select_columns(df, feature_list):
    numeric_cols = []
    for col in feature_list:
        if col in df.columns and df[col].notna().sum() > 0:
                numeric_cols.append(col)
    return numeric_cols


# ---------------------- MODEL TRAINING ----------------------

def train_linear_model(df, features):
    X = df[features]
    y = df['ETV']

    # Preprocessing
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict & Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📉 MSE: {mse:.2f}")
    print(f"📈 R² Score: {r2:.2f}")
    print(f"📌 Coefficients: {model.coef_}")
    print(f"📌 Intercept: {model.intercept_}")


# ---------------------- MAIN PROCESS ----------------------

def main():
    df, filtered_df = load_data()
    if df is None:
        return

    print("🕸️  Collecting market value data...")
    etv_dict = scrape_etv(filtered_df)
    filtered_df['ETV'] = filtered_df['player'].map(etv_dict)

    # Save ETVs to file
    filtered_df[['player', 'ETV']].to_csv('./Task_4/etv.csv', index=False, encoding=ENCODING)
    print("✅ Exported ETV data to Task_4/etv.csv")

    # Choose and clean features
    features = ['minutes', 'goals_assists', 'passes_pct', 'progressive_passes_received', 'gca', 'sca']
    usable_cols = clean_and_select_columns(df, features)

    try:
        filtered_df['ETV'] = filtered_df['ETV'].str.replace(r'[^\d.]', '', regex=True)
        filtered_df['ETV'] = pd.to_numeric(filtered_df['ETV'], errors='coerce')
    except:
        print("❌ Failed to convert ETV to float.")
        return
    
    model_ready = filtered_df.dropna(subset=usable_cols + ['ETV']).copy()
    print(f"🧮 Final training size: {len(model_ready)} players")

    train_linear_model(model_ready, usable_cols)

    # ---------------------- COMMENTARY ----------------------
    # print("\nCommentary:")
    # print("Linear Regression was applied to predict Estimated Transfer Values based on selected performance metrics.")
    # print("Players with more minutes and higher goal contribution tend to have higher predicted ETV.")
    # print("The model can be improved with more features or more accurate value data.")

if __name__ == '__main__':
    main()