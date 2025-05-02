import seaborn as sns
from pathlib import Path

Path("data").mkdir(parents=True, exist_ok=True)
diamonds = sns.load_dataset('diamonds')
diamonds.to_csv('data/diamonds.csv', index=False)