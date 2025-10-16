import pandas as pd
import json
import os
from pathlib import Path

# Semboller
SYMBOLS = ['BTCUSDT', 'SOLUSDT', 'ETHUSDT', 'AVAXUSDT', 'PEPEUSDT']
TF = '15m'
BUILD = 'S5-F1.1-soft-k2'

# Config listesi
CONFIGS = ['Baseline', 'DECAY', 'DRAIN', 'RVGATES', 'DECAY+DRAIN', 'FULL']

# Skor ağırlıkları
W_OVERSHOOT = 0.5
W_FLIP = 0.3
W_TRADES = 0.2

def process_symbol(symbol):
    """Tek bir sembol için tüm post-processing"""
    print(f"\n{'='*60}")
    print(f"Processing: {symbol}")
    print(f"{'='*60}")
    
    # Dosya yolları
    summary_file = f"metrics_{symbol}_{TF}_{BUILD}_grid_summary.csv"
    alerts_file = f"alerts_{symbol}_{TF}_{BUILD}_grid.jsonl"
    
    # Output klasörü
    out_dir = Path(f"reports/{symbol}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary dosyasını oku
    df_summary = pd.read_csv(summary_file)
    
    # Config bazlı ortalamalar (zaten summary'de var, sadece yeniden kaydet)
    df_summary.to_csv(out_dir / f"grid_summary_per_config_{symbol}.csv", index=False)
    print(f"✓ Saved: grid_summary_per_config_{symbol}.csv")
    
    # 2. Baseline'a göre deltalar
    baseline = df_summary[df_summary['config'] == 'Baseline'].iloc[0]
    
    deltas = []
    for _, row in df_summary.iterrows():
        if row['config'] == 'Baseline':
            continue
        
        delta_row = {'config': row['config']}
        for col in df_summary.columns:
            if col in ['config', 'block']:
                continue
            delta_row[f"Δ_{col}"] = row[col] - baseline[col]
        deltas.append(delta_row)
    
    df_deltas = pd.DataFrame(deltas)
    df_deltas.to_csv(out_dir / f"grid_deltas_vs_baseline_{symbol}.csv", index=False)
    print(f"✓ Saved: grid_deltas_vs_baseline_{symbol}.csv")
    
    # 3. Alert stats
    alert_stats = []
    with open(alerts_file, 'r') as f:
        for line in f:
            alert = json.loads(line)
            alert_stats.append({
                'config': alert['config'],
                'block': alert['block'],
                'rv': alert['rv'],
                'rv_band': alert['rv_band'],
                'gate_bias': alert['gate_bias'],
                'cooldown_mult': alert['cooldown_mult'],
                'i_decay_evt': alert['i_decay_evt'],
                'i_drain_evt': alert['i_drain_evt']
            })
    
    df_alerts = pd.DataFrame(alert_stats)
    
    # Config bazlı aggregation
    alert_summary = df_alerts.groupby('config').agg({
        'rv': ['count', 'mean', 'std'],
        'gate_bias': 'mean',
        'cooldown_mult': 'mean',
        'i_decay_evt': 'sum',
        'i_drain_evt': 'sum'
    }).reset_index()
    
    alert_summary.columns = ['config', 'samples', 'rv_mean', 'rv_std', 
                             'gate_bias_mean', 'cd_mult_mean', 
                             'i_decay_sum', 'i_drain_sum']
    
    # RV bandları oranı
    rv_bands = df_alerts.groupby('config')['rv_band'].value_counts(normalize=True).unstack(fill_value=0)
    if 'mid' in rv_bands.columns:
        alert_summary['rv_mid_pct'] = rv_bands['mid'].values * 100
    else:
        alert_summary['rv_mid_pct'] = 0.0
    
    alert_summary.to_csv(out_dir / f"alert_sample_stats_{symbol}.csv", index=False)
    print(f"✓ Saved: alert_sample_stats_{symbol}.csv")
    
    # 4. Config ranking
    rankings = []
    for _, row in df_deltas.iterrows():
        score = (
            W_OVERSHOOT * (-row['Δ_overshoot_p95_midRV']) +
            W_FLIP * (-row['Δ_flip_rate_pct']) +
            W_TRADES * (row['Δ_trades_per_day'])
        )
        rankings.append({
            'config': row['config'],
            'score': score,
            'Δ_overshoot_p95': row['Δ_overshoot_p95_midRV'],
            'Δ_flip_rate': row['Δ_flip_rate_pct'],
            'Δ_trades_per_day': row['Δ_trades_per_day']
        })
    
    df_ranking = pd.DataFrame(rankings).sort_values('score', ascending=False)
    df_ranking.to_csv(out_dir / f"ranking_{symbol}.csv", index=False)
    print(f"✓ Saved: ranking_{symbol}.csv")
    
    return df_ranking.iloc[0]['config'], df_ranking.iloc[0]['score']

def main():
    print("🚀 Atlas Apex 2.3 WFA Grid Post-Processing")
    print("="*60)
    
    recommendations = []
    
    for symbol in SYMBOLS:
        try:
            best_config, score = process_symbol(symbol)
            recommendations.append({
                'symbol': symbol,
                'best_config': best_config,
                'score': round(score, 4)
            })
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            continue
    
    # Final recommendations
    df_rec = pd.DataFrame(recommendations)
    df_rec.to_csv("reports/grid_config_recommendations.csv", index=False)
    print(f"\n{'='*60}")
    print("✅ Post-processing COMPLETE!")
    print(f"{'='*60}\n")
    print("📦 Final Recommendations:")
    print(df_rec.to_string(index=False))

if __name__ == "__main__":
    main()