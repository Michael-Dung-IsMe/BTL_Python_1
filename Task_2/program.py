import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

# ---------------------------- CONFIG ----------------------------
np.random.seed(42)
plt.rcParams.update({'font.size': 12})
HIST_BINS = 20
COLOR_SCHEME = {'attack': '#7ABBD4', 'defense': '#E47460'}

TOP_COUNT = 3
TOP_FILE = './Task_2/top_3.txt'
STATS_FILE = './Task_2/results2.csv'
IMG_OUTPUT_DIR = './Task_2/histograms'
ENCODING = 'utf-8-sig'
INPUT_FILE = './Task_1/results.csv'

# ---------------------------- COLUMN INFO ----------------------------
NON_NUMERIC = ['player', 'nationality', 'position', 'team', 'age']

attack_metrics = {
    'Standard SoT/90': 'shots_on_target_per90',
    'Standard G/Sh': 'goals_per_shot',
    'Standard Dist': 'average_shot_distance'
}

defense_metrics = {
    'Tackles Tkl': 'tackles',
    'Tackles TklW': 'tackles_won',
    'Blocks': 'blocks'
}

# ---------------------------- FUNCTIONS ----------------------------

def read_and_clean_data():
    """Load CSV and convert numeric strings"""
    try:
        df = pd.read_csv(INPUT_FILE, encoding=ENCODING)
        print("📥 File loaded successfully.")
        for col in df.columns:
            if col not in NON_NUMERIC:
                if df[col].dtype == 'object' and df[col].str.contains('%', na=False).any():
                    df[col] = df[col].str.replace('%', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        print("🚫 Input file missing.")
    except Exception as err:
        print(f"⚠️ Error: {err}")


def save_top_bottom(df, num_cols):
    """Save top & bottom 3 players per stat"""
    try:
        with open(TOP_FILE, 'w', encoding=ENCODING) as f:
            for label, col in num_cols.items():
                top = df[['player', col]].dropna().sort_values(by=[col, 'player'], ascending=[False, True]).head(TOP_COUNT)
                bot = df[['player', col]].dropna().sort_values(by=[col, 'player'], ascending=[True, True]).head(TOP_COUNT)
                f.write(f"📊 {label}\n{'='*50}\n⬆️ Top {TOP_COUNT}:\n")
                for _, row in top.iterrows():
                    f.write(f"  {row['player']}: {row[col]}\n")
                f.write(f"⬇️ Bottom {TOP_COUNT}:\n")
                for _, row in bot.iterrows():
                    f.write(f"  {row['player']}: {row[col]}\n")
                f.write("-" * 50 + "\n")
    except Exception as err:
        print(f"❗ Error saving top/bottom: {err}")


def create_histograms(df, cols, tag='all_players'):
    """Create histograms for statistics"""
    out_path = Path(IMG_OUTPUT_DIR) / tag
    out_path.mkdir(parents=True, exist_ok=True)
    for label, field in cols.items():
        color = COLOR_SCHEME['attack'] if label in attack_metrics else COLOR_SCHEME['defense']
        safe_label = re.sub(r'[^\w]', '_', label)
        title = tag.replace('_', ' ')
        plt.figure(figsize=(10, 7), dpi=100)
        plt.hist(df[field].dropna(), bins=HIST_BINS, color=color, edgecolor='black')
        plt.title(f"{title}'s {label} Distribution")
        plt.xlabel(label)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out_path / f"{safe_label}_{tag}.png")
        plt.close()


def summarize_by_team(df, metrics):
    """Calculate mean, median and std for each team"""
    team_data = []
    for team, sub_df in df.groupby('team'):
        team_stats = {'Team': team}
        for label, field in metrics.items():
            team_stats[f'Median of {label}'] = sub_df[field].median()
            team_stats[f'Mean of {label}'] = sub_df[field].mean()
            team_stats[f'Std of {label}'] = sub_df[field].std()
        team_data.append(team_stats)
    return pd.DataFrame(team_data)


def add_overall_average(df_stats, metrics):
    """Add row 'All players' """
    all_row = {'Team': 'All players'}
    for label in metrics:
        for prefix in ['Median', 'Mean', 'Std']:
            col = f'{prefix} of {label}'
            all_row[col] = df_stats[col].mean() if col in df_stats.columns else None
    return pd.concat([pd.DataFrame([all_row]), df_stats], ignore_index=True)


# ---------------------------- MAIN PROCESS ----------------------------
def main():
    df = read_and_clean_data()
    if df is None:
        return

    required = {'Player': 'player', 'Squad': 'team'} | attack_metrics | defense_metrics
    for col in required.values():
        if col not in df.columns:
            print(f"⚠️ Missing column: {col}")
            return

    metrics = attack_metrics | defense_metrics
    save_top_bottom(df, metrics)
    print(f"✅ Top/bottom players saved to {TOP_FILE[2:]}")

    create_histograms(df, metrics)
    print(f"✅ Histograms saved to: {IMG_OUTPUT_DIR[2:]}")

    for team, subset in df.groupby('team'):
        team_name = re.sub(r'[^\w]', '_', team)
        create_histograms(subset, metrics, team_name)

    team_stats_df = summarize_by_team(df, metrics)
    final_stats = add_overall_average(team_stats_df, metrics)
    final_stats.to_csv(STATS_FILE, index=False, encoding=ENCODING)
    print(f"📁 Team stats exported to {STATS_FILE[2:]}")


if __name__ == '__main__':
    main()