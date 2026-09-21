"""Part 1: regenerate the 30 per-market-seed figures ({market}_metrics_heatmap.png, {market}_rank_comparison.png) from the S4 metrics CSVs with the notebook's own plotting
functions. The CSVs are read, never written. The pre-fix flag column is dropped and rows are ordered as the pipeline writes them (generators by composite, control last)."""
import sys, io, contextlib, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd
import embedding_distance as ed
ns = ed.load_notebook(); R = ns['RESULTS']; n = 0
for m in ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']:
    for s in (42, 43, 44):
        d = R / m / f'seed{s}'; df = pd.read_csv(d / f'{m}_metrics.csv').drop(columns=['discriminative_auc_pre_scaler_fix'], errors='ignore')
        ctrl = df.model.str.contains('CONTROL'); df = pd.concat([df[~ctrl].sort_values('composite_rank'), df[ctrl]], ignore_index=True)
        with contextlib.redirect_stdout(io.StringIO()):
            f1 = ns['plot_metrics_heatmap'](df, save_path=str(d / f'{m}_metrics_heatmap.png')); plt.close(f1)
            f2 = ns['plot_rank_comparison'](df, save_path=str(d / f'{m}_rank_comparison.png')); plt.close(f2)
        n += 2
print('FIGURES_DONE', n, 'figures regenerated')
