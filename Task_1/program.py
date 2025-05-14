from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC # Let program waits till the condition is through
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from bs4 import BeautifulSoup as bs
import pandas as pd
import re

# ------------------------------------ DATA COLUMNS IN THE CSV FILE -------------------------------------

stats_columns = ['player', 'nationality', 'position', 'team', 'age', 'games', 'games_starts', 'minutes', 'goals',
                'assists', 'goals_assists', 'cards_yellow', 'cards_red', 'xg', 'xg_assist', 'progressive_carries',
                'progressive_passes', 'progressive_passes_received', 'goals_per90', 'assists_per90', 'xg_per90', 'xg_assist_per90',
                'gk_goals_against_per90', 'gk_save_pct', 'gk_clean_sheets_pct', 'gk_pens_save_pct',
                'shots_on_target_pct', 'shots_on_target_per90', 'goals_per_shot', 'average_shot_distance',
                'passes_completed', 'passes_pct', 'passes_total_distance', 'passes_pct_short', 'passes_pct_medium',
                'passes_pct_long', 'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area', 'crosses_into_penalty_area', 'progressive_passes',
                'sca', 'sca_per90', 'gca', 'gca_per90',
                'tackles', 'tackles_won', 'challenges', 'challenges_lost', 'blocks', 'blocked_shots', 'blocked_passes', 'interceptions',
                'touches', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd', 'touches_att_pen_area', 'take_ons', 'take_ons_won_pct',
                'take_ons_tackled_pct', 'carries', 'carries_progressive_distance', 'progressive_carries', 'carries_into_final_third', 'carries_into_penalty_area',
                'miscontrols', 'dispossessed', 'passes_received', 'progressive_passes_received',
                'fouls', 'fouled', 'offsides', 'crosses', 'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_pct'
]

# ------------------------------------- LINK LIST FOR FETCHING DATA --------------------------------------

URLS = {
    'Standard Stats': ('https://fbref.com/en/comps/9/stats/Premier-League-Stats', 'stats_standard'),
    'Goalkeeping': ('https://fbref.com/en/comps/9/keepers/Premier-League-Stats', 'stats_keeper'),
    'Shooting': ('https://fbref.com/en/comps/9/shooting/Premier-League-Stats', 'stats_shooting'),
    'Passing': ('https://fbref.com/en/comps/9/passing/Premier-League-Stats', 'stats_passing'),
    'Goal and Shot Creation': ('https://fbref.com/en/comps/9/gca/Premier-League-Stats', 'stats_gca'),
    'Defense': ('https://fbref.com/en/comps/9/defense/Premier-League-Stats', 'stats_defense'),
    'Possession': ('https://fbref.com/en/comps/9/possession/Premier-League-Stats', 'stats_possession'),
    'Miscellaneous': ('https://fbref.com/en/comps/9/misc/Premier-League-Stats', 'stats_misc')
}

# Khởi tạo DataFrame results
results = pd.DataFrame(columns=stats_columns)
pd.set_option('future.no_silent_downcasting', True)

def setup_driver():
    """Khởi tạo WebDriver cho Edge"""
    options = EdgeOptions()
    options.add_argument('--headless')  # Running headless (Not presenting interface)
    # options.add_argument('--ignore-certificate-errors') # Ignoring SSL certificate error
    # options.add_argument('--allow-insecure-localhost')  # Allowing insecure localhost
    driver = webdriver.Edge(service=EdgeService(executable_path=EdgeChromiumDriverManager().install()), options=options)
    return driver

def str_to_int(num):
    return int(''.join(re.findall(r'[^,]', num)))

driver = setup_driver()

try:
    for table_name, (link, table_id) in URLS.items():
        data_stats_columns = []
        try:
            driver.get(link)
            target_table = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, table_id))
                )

            # If table can't be found
            if not target_table:
                print(f'Table having ID {table_id} was not found.')
                continue
            
            else:
                if table_name == 'Standard Stats':
                    data_stats_columns = ['player', 'nationality', 'position', 'team', 'age', 'games', 'games_starts', 'minutes', 'goals',
                                        'assists', 'goals_assists', 'cards_yellow', 'cards_red', 'xg', 'xg_assist', 'progressive_carries',
                                        'progressive_passes', 'progressive_passes_received', 'goals_per90', 'assists_per90', 'xg_per90', 'xg_assist_per90'
                                        ]
                
                elif table_name == 'Goalkeeping':
                    data_stats_columns = ['player', 'gk_goals_against_per90', 'gk_save_pct', 'gk_clean_sheets_pct', 'gk_pens_save_pct']
                
                elif table_name == 'Shooting':
                    data_stats_columns = ['player', 'shots_on_target_pct', 'shots_on_target_per90', 'goals_per_shot', 'average_shot_distance']
                
                elif table_name == 'Passing':
                    data_stats_columns = ['player', 'passes_completed', 'passes_pct', 'passes_total_distance', 'passes_pct_short', 'passes_pct_medium',
                                        'passes_pct_long', 'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area', 'crosses_into_penalty_area', 'progressive_passes'
                                        ]
                
                elif table_name == 'Goal and Shot Creation':
                    data_stats_columns = ['player', 'sca', 'sca_per90', 'gca', 'gca_per90']
                
                elif table_name == 'Defense':
                    data_stats_columns = ['player', 'tackles', 'tackles_won', 'challenges', 'challenges_lost', 'blocks', 'blocked_shots', 'blocked_passes', 'interceptions']
                
                elif table_name == 'Possession':
                    data_stats_columns = ['player', 'touches', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd', 'touches_att_pen_area', 'take_ons', 'take_ons_won_pct',
                                        'take_ons_tackled_pct', 'carries', 'carries_progressive_distance', 'progressive_carries', 'carries_into_final_third', 'carries_into_penalty_area',
                                        'miscontrols', 'dispossessed', 'passes_received', 'progressive_passes_received'
                                        ]
                
                elif table_name == 'Miscellaneous':
                    data_stats_columns = ['player', 'fouls', 'fouled', 'offsides', 'crosses', 'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_pct']

                table_html = target_table.get_attribute('outerHTML')
                soup = bs(table_html, 'html.parser')

                target_tbody = soup.find('tbody')
                rows = target_tbody.find_all('tr')
                temp_data = []
                for row in rows:
                    cells = row.find_all(['td'], {'data-stat': True})
                    if len(cells) == 0: continue
                    row_data = {cell['data-stat']: cell.get_text(strip=True) for cell in cells if cell['data-stat'] in data_stats_columns}
                    if table_name == 'Standard Stats':
                        row_data['minutes'] = str_to_int(row_data['minutes'])
                        if row_data['minutes'] > 90:
                            temp_data.append(row_data)
                        else:
                            print(f"❗Player {row_data['player']} hasn't played more than 90mins!!")
                            continue
                    else:
                        if row_data and 'player' in row_data:
                            temp_data.append(row_data)

                # Change data into a DataFrame
                temp_df = pd.DataFrame(temp_data, columns=data_stats_columns)

                for _, row in temp_df.iterrows():
                    if table_name == 'Standard Stats':
                        new_row = pd.Series('N/a', index=results.columns)
                        for col in temp_df.columns:
                            if col in results.columns:
                                new_row[col] = (row[col] if row[col] != '' else 'N/a')
                        results = pd.concat([results, new_row.to_frame().T], ignore_index=True)
                    
                    else:
                        player = row['player']
                        if player in results['player'].values:
                            for col in temp_df.columns:
                                # print(col)
                                if col in results.columns:
                                    results.loc[results['player'] == player, col] = row[col]

        except Exception as e:
            print(f"❌ Error scraping {table_name}: {e}")
            continue

finally:
    driver.quit()

# Print DataFrame
# print(results.head())

# Save DataFrame to CSV file
results['First'] = results['player'].apply(lambda x: x.split()[0])
results.sort_values(by='First', inplace=True)
results.drop(columns=['First'], inplace=True)  # Delete this column

results.index = range(1, len(results) + 1)

results.fillna('N/a', inplace=True)
results.to_csv("results.csv", index=False)
print("✅ Data saved to results.csv")