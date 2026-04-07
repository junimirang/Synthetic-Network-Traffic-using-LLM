"""
SSH Traffic t-SNE Comparison
Original vs WGAN-GP vs LLM Synthesized

사용법:
    # 세 파일 모두 있는 경우
    python tsne_comparison.py \
        --original "training_dataset_week4_normalized.csv" \
        --wgan_gp  "fake_ssh_1_2026-03-10-15.csv" \
        --llm_csv  "llm_studio_ssh.csv" \
        --n_samples 1000

    # 파일 없는 경우 시뮬레이션으로 대체
    python tsne_comparison.py \
        --original "training_dataset_week4_normalized.csv"
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ── 피처 정의 ────────────────────────────────────────────────────────────────
FEATURES = [
    'log_count_total_connect',
    'log_cs_byte',
    'log_transmit_speed_BPS',
    'log_count_connect_IP',
    'log_avg_count_connect',
    'log_time_taken',
    'log_time_interval_mean',
    'log_time_interval_var',
    'log_ratio_trans_receive'
]

FEATURE_SHORT = [
    'Total\nConnect',
    'CS\nByte',
    'Speed\nBPS',
    'Connect\nIP',
    'Avg\nConnect',
    'Time\nTaken',
    'IPA\nMean',
    'IPA\nVar',
    'T/R\nRatio'
]

# 그룹별 스타일
STYLE = {
    'Original SSH':    dict(color='#1565C0', marker='o', s=12, alpha=0.70, zorder=3, lw=0),
    'WGAN-GP':         dict(color='#2E7D32', marker='^', s=12, alpha=0.55, zorder=2, lw=0),
    'LLM Synthesized': dict(color='#C62828', marker='x', s=18, alpha=0.50, zorder=1, lw=0.8),
}


# ── 시간 포맷 헬퍼 ────────────────────────────────────────────────────────────
def _fmt_time(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds//60}m {seconds%60}s"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}h {m}m {s}s"


# ── 1. 원본 SSH 데이터 로딩 ──────────────────────────────────────────────────
def load_original(filepath, n_samples=1000, seed=42):
    t0 = time.time()
    print(f"[1] 원본 SSH 데이터 로딩")
    print(f"    파일: {filepath}")
    df  = pd.read_csv(filepath, index_col=0)
    ssh = df[df['LABEL'] == 'ssh'][FEATURES].dropna()
    print(f"    전체 SSH: {len(ssh)}건")
    if len(ssh) >= n_samples:
        ssh = ssh.sample(n=n_samples, random_state=seed)
    print(f"    선택:     {len(ssh)}건  ✅  ({_fmt_time(time.time()-t0)})")
    return ssh.reset_index(drop=True)


# ── 2. WGAN-GP 데이터 로딩 ──────────────────────────────────────────────────
def load_wgan_gp(filepath, n_samples=1000, seed=42):
    t0 = time.time()
    print(f"\n[2] WGAN-GP 데이터 로딩")
    if not filepath or not os.path.exists(filepath):
        print(f"    파일 없음 → 시뮬레이션으로 대체  ⚠")
        return None, True

    print(f"    파일: {filepath}")
    df = pd.read_csv(filepath, index_col=0)

    # LABEL 필터
    for col in ['LABEL with GAP and Softmax', 'LABEL']:
        if col in df.columns:
            before = len(df)
            df = df[df[col] == 'ssh']
            print(f"    '{col}' ssh 필터: {before}→{len(df)}건")
            break

    # 사용 가능한 피처 선택
    avail = [f for f in FEATURES if f in df.columns]
    if len(avail) < len(FEATURES):
        print(f"    ⚠ 누락 피처: {set(FEATURES)-set(avail)}")
    df = df[avail].dropna()

    # 스케일 자동 감지 (/100 된 경우 복원)
    if df[avail].max().max() <= 2.0:
        df[avail] = df[avail] * 100
        print(f"    스케일 ×100 복원 적용")

    if len(df) >= n_samples:
        df = df.sample(n=n_samples, random_state=seed)

    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    print(f"    선택: {len(df)}건  ✅  ({_fmt_time(time.time()-t0)})")
    return df[FEATURES].reset_index(drop=True), False


# ── 3. LLM 합성 데이터 로딩 ─────────────────────────────────────────────────
def load_llm(filepath, n_samples=1000, seed=42):
    t0 = time.time()
    print(f"\n[3] LLM 합성 데이터 로딩")
    if not filepath or not os.path.exists(filepath):
        print(f"    파일 없음 → 시뮬레이션으로 대체  ⚠")
        return None, True

    print(f"    파일: {filepath}")
    df    = pd.read_csv(filepath)
    avail = [f for f in FEATURES if f in df.columns]
    df    = df[avail].dropna()

    if len(df) >= n_samples:
        df = df.sample(n=n_samples, random_state=seed)

    print(f"    선택: {len(df)}건  ✅  ({_fmt_time(time.time()-t0)})")
    return df[FEATURES].reset_index(drop=True), False


# ── 시뮬레이션 데이터 생성 ───────────────────────────────────────────────────
def simulate(df_orig, n_samples=500, seed=42, noise_scale=1.05):
    np.random.seed(seed)
    stats  = df_orig[FEATURES].describe()
    result = {}
    biases = np.random.uniform(-1.0, 1.0, len(FEATURES))
    for i, f in enumerate(FEATURES):
        mu  = stats.loc['mean', f]
        sig = stats.loc['std',  f]
        mn  = stats.loc['min',  f]
        mx  = stats.loc['max',  f]
        s   = np.random.normal(mu + biases[i], sig * noise_scale, n_samples)
        result[f] = np.clip(s, mn, mx)
    return pd.DataFrame(result)


# ── JSD 계산 ─────────────────────────────────────────────────────────────────
def compute_jsd(p_df, q_df, bins=30):
    result = {}
    for f in FEATURES:
        p = p_df[f].values
        q = q_df[f].values
        lo = min(p.min(), q.min())
        hi = max(p.max(), q.max())
        if hi == lo:
            result[f] = 0.0
            continue
        ph, _ = np.histogram(p, bins=bins, range=(lo, hi), density=True)
        qh, _ = np.histogram(q, bins=bins, range=(lo, hi), density=True)
        ph = ph + 1e-10;  ph /= ph.sum()
        qh = qh + 1e-10;  qh /= qh.sum()
        m  = 0.5 * (ph + qh)
        jsd = 0.5 * (np.sum(ph * np.log(ph / m)) +
                     np.sum(qh * np.log(qh / m)))
        result[f] = round(float(jsd), 5)
    return result


# ── 메인 시각화 함수 ─────────────────────────────────────────────────────────
def run_tsne_and_plot(
    df_orig, df_wgan, df_llm,
    output_path, perplexity=50,
    wgan_sim=False, llm_sim=False
):
    n_orig = len(df_orig)
    n_wgan = len(df_wgan)
    n_llm  = len(df_llm)
    total  = n_orig + n_wgan + n_llm

    print(f"\n[4] t-SNE 실행")
    print(f"    총 샘플: {total}건  |  perplexity={perplexity}")

    combined = pd.concat(
        [df_orig[FEATURES], df_wgan[FEATURES], df_llm[FEATURES]],
        ignore_index=True
    )
    labels = np.array(
        ['Original SSH']    * n_orig +
        ['WGAN-GP']         * n_wgan +
        ['LLM Synthesized'] * n_llm
    )

    # ── 표준화 ───────────────────────────────────────────────
    t_scale = time.time()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(combined)
    print(f"    표준화 완료  ({_fmt_time(time.time()-t_scale)})")

    # ── t-SNE ────────────────────────────────────────────────
    t_tsne = time.time()
    print(f"    t-SNE 연산 시작...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    R = tsne.fit_transform(X_scaled)
    print(f"    t-SNE 완료  ✅  ({_fmt_time(time.time()-t_tsne)})")

    # ── JSD 계산 ─────────────────────────────────────────────
    t_jsd = time.time()
    jsd_wgan = compute_jsd(df_orig, df_wgan)
    jsd_llm  = compute_jsd(df_orig, df_llm)
    avg_jsd_wgan = np.mean(list(jsd_wgan.values()))
    avg_jsd_llm  = np.mean(list(jsd_llm.values()))
    print(f"    JSD 계산 완료  ({_fmt_time(time.time()-t_jsd)})")

    # ── 레이아웃 ─────────────────────────────────────────────
    t_plot = time.time()
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('#F7F7F7')
    gs  = gridspec.GridSpec(
        2, 2,
        height_ratios=[1.8, 1],
        hspace=0.35, wspace=0.3
    )

    # ── [상단 왼쪽] t-SNE 산점도 ────────────────────────────
    ax_tsne = fig.add_subplot(gs[0, 0])
    ax_tsne.set_facecolor('#FAFAFA')

    for grp, st in STYLE.items():
        mask = labels == grp
        n    = mask.sum()
        lbl  = f"{grp}  (n={n:,})"
        if grp == 'WGAN-GP'         and wgan_sim: lbl += ' *sim'
        if grp == 'LLM Synthesized' and llm_sim:  lbl += ' *sim'
        ax_tsne.scatter(
            R[mask, 0], R[mask, 1],
            c=st['color'], marker=st['marker'],
            s=st['s'], alpha=st['alpha'],
            zorder=st['zorder'],
            linewidths=st['lw'],
            label=lbl
        )

    title_tsne = ('t-SNE: SSH Traffic Feature Distribution\n'
                  'Original  vs  WGAN-GP  vs  LLM Synthesized')
    if wgan_sim or llm_sim:
        title_tsne += '\n(* = simulation — provide real CSV for actual results)'
    ax_tsne.set_title(title_tsne, fontsize=12, fontweight='bold', pad=8)
    ax_tsne.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax_tsne.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax_tsne.legend(fontsize=9.5, framealpha=0.92,
                   edgecolor='#CCCCCC', loc='upper right')
    ax_tsne.grid(True, alpha=0.18, linestyle='--')
    ax_tsne.tick_params(labelsize=8)

    # ── [상단 오른쪽] 밀도 맵 ───────────────────────────────
    ax_hex = fig.add_subplot(gs[0, 1])
    ax_hex.set_facecolor('#0A0A0A')

    colors_hex = {
        'Original SSH':    '#4FC3F7',
        'WGAN-GP':         '#81C784',
        'LLM Synthesized': '#EF9A9A',
    }
    for grp, col in colors_hex.items():
        mask = labels == grp
        ax_hex.hexbin(
            R[mask, 0], R[mask, 1],
            gridsize=35, cmap=None,
            alpha=0.55, linewidths=0.2,
            mincnt=1,
            facecolors=col
        )

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=colors_hex[g], alpha=0.7, label=g)
        for g in colors_hex
    ]
    ax_hex.legend(handles=legend_els, fontsize=9, framealpha=0.85,
                  edgecolor='#555555', loc='upper right',
                  labelcolor='white', facecolor='#1A1A1A')
    ax_hex.set_title('t-SNE Density Map\n(Hexagonal Binning)',
                     fontsize=12, fontweight='bold', pad=8, color='white')
    ax_hex.set_xlabel('t-SNE Dimension 1', fontsize=10, color='#CCCCCC')
    ax_hex.set_ylabel('t-SNE Dimension 2', fontsize=10, color='#CCCCCC')
    ax_hex.tick_params(colors='#AAAAAA', labelsize=8)
    for sp in ax_hex.spines.values():
        sp.set_edgecolor('#333333')

    # ── [하단 왼쪽] JSD 바 차트 ─────────────────────────────
    ax_jsd = fig.add_subplot(gs[1, 0])
    ax_jsd.set_facecolor('#FAFAFA')

    x  = np.arange(len(FEATURES))
    w  = 0.35
    vw = [jsd_wgan[f] for f in FEATURES]
    vl = [jsd_llm[f]  for f in FEATURES]

    bars_w = ax_jsd.bar(x - w/2, vw, w,
                        label=f'WGAN-GP  (avg={avg_jsd_wgan:.4f})',
                        color='#2E7D32', alpha=0.78)
    bars_l = ax_jsd.bar(x + w/2, vl, w,
                        label=f'LLM       (avg={avg_jsd_llm:.4f})',
                        color='#C62828', alpha=0.78)

    for bar in bars_w:
        h = bar.get_height()
        if h > 0.001:
            ax_jsd.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                        f'{h:.3f}', ha='center', va='bottom',
                        fontsize=6.5, color='#2E7D32')
    for bar in bars_l:
        h = bar.get_height()
        if h > 0.001:
            ax_jsd.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                        f'{h:.3f}', ha='center', va='bottom',
                        fontsize=6.5, color='#C62828')

    ax_jsd.set_title('Jensen-Shannon Divergence per Feature\n'
                     '(vs Original SSH — Lower = More Similar)',
                     fontsize=12, fontweight='bold', pad=8)
    ax_jsd.set_ylabel('JSD Score', fontsize=10)
    ax_jsd.set_xticks(x)
    ax_jsd.set_xticklabels(FEATURE_SHORT, fontsize=8.5)
    ax_jsd.legend(fontsize=9.5, framealpha=0.9)
    ax_jsd.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax_jsd.set_ylim(0, max(max(vw), max(vl)) * 1.35 + 0.005)

    # ── [하단 오른쪽] 박스플롯 ──────────────────────────────
    ax_box = fig.add_subplot(gs[1, 1])
    ax_box.set_facecolor('#FAFAFA')

    KEY_FEATURES = [
        'log_time_interval_mean',
        'log_time_interval_var',
        'log_ratio_trans_receive',
        'log_time_taken'
    ]
    KEY_LABELS = ['IPA Mean', 'IPA Variance', 'T/R Ratio', 'Time Taken']

    n_kf    = len(KEY_FEATURES)
    x_box   = np.arange(n_kf)
    width_b = 0.22
    offsets = [-width_b, 0, width_b]
    groups  = [
        ('Original SSH',    df_orig, '#1565C0'),
        ('WGAN-GP',         df_wgan, '#2E7D32'),
        ('LLM Synthesized', df_llm,  '#C62828'),
    ]

    for idx, (grp, df_g, col) in enumerate(groups):
        data = [df_g[f].dropna().values for f in KEY_FEATURES]
        pos  = x_box + offsets[idx]
        ax_box.boxplot(
            data,
            positions=pos,
            widths=width_b * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='white', linewidth=1.5),
            boxprops=dict(facecolor=col, alpha=0.65, linewidth=0.5),
            whiskerprops=dict(color=col, linewidth=0.8),
            capprops=dict(color=col, linewidth=0.8)
        )
        ax_box.plot([], [], color=col, linewidth=5, alpha=0.65, label=grp)

    ax_box.set_title('Feature Distribution Comparison\n'
                     '(Key Features — Box Plot)',
                     fontsize=12, fontweight='bold', pad=8)
    ax_box.set_ylabel('Value (log-scaled, 0-100)', fontsize=10)
    ax_box.set_xticks(x_box)
    ax_box.set_xticklabels(KEY_LABELS, fontsize=9)
    ax_box.legend(fontsize=9, framealpha=0.9)
    ax_box.grid(True, alpha=0.2, axis='y', linestyle='--')

    # ── 전체 제목 ────────────────────────────────────────────
    subtitle = f'n=1,000 each  |  perplexity={perplexity}  |  DARPA99 SSH Session Dataset'
    fig.suptitle(
        'SSH Traffic Synthesis Quality Analysis\n' + subtitle,
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"    그래프 저장: {output_path}  ✅  ({_fmt_time(time.time()-t_plot)})")
    plt.close()

    return jsd_wgan, jsd_llm


# ── 통계 요약 출력 ───────────────────────────────────────────────────────────
def print_summary(df_orig, df_wgan, df_llm, jsd_wgan, jsd_llm,
                  wgan_sim, llm_sim):
    sep = "=" * 68
    print(f"\n{sep}")
    print("  피처별 평균 비교")
    print(f"  {'Feature':<26} {'Original':>10} {'WGAN-GP':>10} {'LLM':>10}")
    print("-" * 68)
    for f, s in zip(FEATURES, FEATURE_SHORT):
        lbl = s.replace('\n', ' ')
        print(f"  {lbl:<26} "
              f"{df_orig[f].mean():>10.3f} "
              f"{df_wgan[f].mean():>10.3f} "
              f"{df_llm[f].mean():>10.3f}")

    print(f"\n{sep}")
    print("  피처별 표준편차 (다양성 지표)")
    print(f"  {'Feature':<26} {'Original':>10} {'WGAN-GP':>10} {'LLM':>10}")
    print("-" * 68)
    for f, s in zip(FEATURES, FEATURE_SHORT):
        lbl = s.replace('\n', ' ')
        print(f"  {lbl:<26} "
              f"{df_orig[f].std():>10.3f} "
              f"{df_wgan[f].std():>10.3f} "
              f"{df_llm[f].std():>10.3f}")

    print(f"\n{sep}")
    print("  Jensen-Shannon Divergence (vs Original, 낮을수록 유사)")
    print(f"  {'Feature':<26} {'WGAN-GP':>10} {'LLM':>10}  {'우위':>8}")
    print("-" * 68)
    for f, s in zip(FEATURES, FEATURE_SHORT):
        lbl    = s.replace('\n', ' ')
        w      = jsd_wgan[f]
        l      = jsd_llm[f]
        winner = 'WGAN-GP' if w < l else ('LLM' if l < w else 'tie')
        print(f"  {lbl:<26} {w:>10.4f} {l:>10.4f}  {winner:>8}")
    print("-" * 68)
    avg_w      = np.mean(list(jsd_wgan.values()))
    avg_l      = np.mean(list(jsd_llm.values()))
    winner_avg = 'WGAN-GP' if avg_w < avg_l else 'LLM'
    print(f"  {'평균 JSD':<26} {avg_w:>10.4f} {avg_l:>10.4f}  {winner_avg:>8}")

    print(f"\n{sep}")
    print("  데이터 출처")
    print(f"  원본  SSH  : {'실제 데이터':>20}")
    print(f"  WGAN-GP    : {'시뮬레이션' if wgan_sim else '실제 파일':>20}"
          + (" *" if wgan_sim else ""))
    print(f"  LLM        : {'시뮬레이션' if llm_sim else '실제 파일':>20}"
          + (" *" if llm_sim else ""))
    print(sep)


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='SSH Traffic t-SNE Comparison: Original vs WGAN-GP vs LLM'
    )
    parser.add_argument('--original',
                        default='t-sne_data/training_dataset_week4_normalized.csv')
    parser.add_argument('--wgan_gp',
                        default='t-sne_data/fake_ssh_1_2026-03-10-15.csv')
    parser.add_argument('--llm_csv',
                        default='t-sne_data/llm_studio_ssh_2000.csv')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--perplexity',type=int, default=50)
    parser.add_argument('--output',    default='tsne_ssh_comparison.png')
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()   # ── 전체 시작 시간

    print("=" * 60)
    print("  SSH Traffic Synthesis Quality Analysis")
    print("  Original  vs  WGAN-GP  vs  LLM Synthesized")
    print("=" * 60)

    # ── 데이터 로딩 ─────────────────────────────────────────
    df_orig = load_original(args.original, args.n_samples, args.seed)

    df_wgan, wgan_sim = load_wgan_gp(args.wgan_gp, args.n_samples, args.seed)
    if wgan_sim:
        df_wgan = simulate(df_orig, args.n_samples,
                           seed=args.seed + 1, noise_scale=1.03)

    df_llm, llm_sim = load_llm(args.llm_csv, args.n_samples, args.seed)
    if llm_sim:
        df_llm = simulate(df_orig, args.n_samples,
                          seed=args.seed + 2, noise_scale=1.08)

    t_load = time.time() - t_start
    print(f"\n  데이터 로딩 완료  →  {_fmt_time(t_load)}")

    # ── t-SNE + 시각화 ──────────────────────────────────────
    jsd_wgan, jsd_llm = run_tsne_and_plot(
        df_orig, df_wgan, df_llm,
        output_path=args.output,
        perplexity=args.perplexity,
        wgan_sim=wgan_sim,
        llm_sim=llm_sim
    )

    # ── 통계 요약 ────────────────────────────────────────────
    print_summary(df_orig, df_wgan, df_llm,
                  jsd_wgan, jsd_llm, wgan_sim, llm_sim)

    # ── 최종 소요시간 요약 ───────────────────────────────────
    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  출력 파일     : {args.output}")
    print(f"  총 소요시간   : {_fmt_time(t_total)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()