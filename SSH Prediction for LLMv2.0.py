# -*- coding: utf-8 -*-
"""
train_detector_comparison.py  v1.1
==================================
데이터셋별 성능 비교: Original vs WGAN-GP vs LLM

[수정 사항]
  - 실험 모드를 3가지(original, wgan_gp, llm)로 분리
  - 각 데이터셋별로 n_repeat만큼 반복 실험 후 결과 집계
  - 최종 비교 요약 테이블 출력 기능 추가
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from common_config import (DATASET_CONFIG, MODEL_DIR, SSH_LABEL, NON_SSH_LABEL)
from input_helper import ask, section

# GPU 가속 여부 체크
try:
    from cuml.ensemble import RandomForestClassifier as cuRF

    USE_CUML = True
except ImportError:
    USE_CUML = False

N_ESTIMATORS = 100
DUMMY_IP_PORT = "255.255.255.255:1"

# PCA 및 모델 피처 설정
PCA_COLS_DARPA = ["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS", "log_count_connect_IP",
                  "log_avg_count_connect"]
MODEL_COLS_DARPA = ["PC", "log_ratio_trans_receive", "log_time_taken", "log_time_interval_mean",
                    "log_time_interval_var"]
MODEL_COLS_NB15 = ["log_mean_of_IPA_time", "log_ratio_trans_receive", "log_speed_transmit_connect", "log_dur",
                   "log_sbytes", "log_ct_src_dport_ltm", "log_ct_dst_ltm", "log_ct_src_ltm"]


def get_params():
    section("성능 비교 실험 파라미터 입력")
    dataset = ask("데이터셋", choices=["darpa99", "nb15"], default="darpa99")
    cfg = DATASET_CONFIG[dataset]
    test_choices = list(cfg["test_files"].keys())
    test_week = ask(f"테스트셋 {test_choices}", choices=test_choices, default=test_choices[0])
    n_repeat = ask("반복 실험 횟수", cast=int, default=10)  # 비교 실험이므로 기본값 조정
    n_fake_ssh = ask("합성 SSH 수 (-1=전체)", cast=int, default=-1)
    n_fake_nonssh = ask("합성 non-SSH 수 (0=미포함)", cast=int, default=0)

    class P: pass

    p = P()
    p.dataset = dataset;
    p.test_week = test_week;
    p.n_repeat = n_repeat
    p.n_fake_ssh = n_fake_ssh;
    p.n_fake_nonssh = n_fake_nonssh  # 이 부분이 저장되어야 합니다.
    return p


def load_csv(fpath, cfg):
    df = pd.read_csv(fpath).replace([float('inf'), float('-inf')], float('nan'))
    return df.dropna(subset=cfg["feature_cols"]).reset_index(drop=True)


def load_specific_dataset(cfg, opt, target_mode):
    """
    지정된 모드에 따라 데이터를 로드합니다.
    LLM 모드일 경우 교수님이 지정하신 특정 파일명(llm_studio_ssh_1000.csv 등)을 로드합니다.
    """
    label_col = cfg["label_col"]  # common_config의 'LABEL'
    df_real = load_csv(os.path.join(cfg["real_dir"], cfg["train_file"]), cfg)

    # 전파(Propagation) 로직을 위한 기본 목적지 정보 부여
    if "Destination" not in df_real.columns:
        df_real["Destination"] = "0.0.0.0"
        df_real["Destination Port"] = 0

    if target_mode == "original":
        print(f"  [ORIGINAL] 원본 데이터만 사용: {len(df_real):,}행")
        return df_real

    # 합성 데이터 경로 및 파일명 설정
    mode_dir = MODEL_DIR[target_mode]  # common_config에서 'LLM' 혹은 'wgan-gp'

    # LLM 모드일 때 교수님이 말씀하신 파일명 우선 탐색
    if target_mode == "llm":
        # 여러 파일 중 존재하는 첫 번째 파일을 선택하거나, 규칙에 따라 지정 가능
        possible_files = ["llm_studio_ssh_1000.csv", "llm_studio_ssh_2000.csv", "synthetic_ssh.csv"]
        ssh_file = None
        for f in possible_files:
            if os.path.exists(os.path.join(cfg["fake_dir"], mode_dir, f)):
                ssh_file = f
                break

        if ssh_file is None:
            print(f"  ※ 에러: {mode_dir} 폴더 내에 LLM 합성 데이터 파일이 없습니다.")
            return df_real
    else:
        ssh_file = "synthetic_ssh.csv"

    ssh_path = os.path.join(cfg["fake_dir"], mode_dir, ssh_file)
    parts = [df_real]

    if opt.n_fake_ssh != 0 and os.path.exists(ssh_path):
        df_fake = pd.read_csv(ssh_path)

        # [수정] label 또는 LABEL 컬럼 유연하게 대응
        target_col = next((c for c in ['label', 'LABEL'] if c in df_fake.columns), None)

        if target_col:
            # [수정] 'ssh'인 것만 골라내기
            df_ssh_only = df_fake[df_fake[target_col].astype(str).str.lower() == "ssh"].copy()

            if len(df_ssh_only) > 0:
                # 사용자가 입력한 n_fake_ssh만큼 샘플링
                if opt.n_fake_ssh > 0 and len(df_ssh_only) > opt.n_fake_ssh:
                    df_ssh_only = df_ssh_only.sample(n=opt.n_fake_ssh, random_state=42)

                # 테스트셋과 동일한 레이블로 통일 및 목적지 더미값 부여
                df_ssh_only[label_col] = SSH_LABEL
                df_ssh_only["Destination"] = "255.255.255.255"
                df_ssh_only["Destination Port"] = 1

                parts.append(df_ssh_only)
                print(f"  [{target_mode}] {ssh_file}에서 {len(df_ssh_only):,}행 결합 완료")
            else:
                print(f"  ※ 경고: {ssh_file} 내에 'ssh' 레이블 데이터가 없습니다.")
        else:
            print(f"  ※ 에러: {ssh_file} 내에 label/LABEL 컬럼이 없습니다.")

    return pd.concat(parts, ignore_index=True)


# [기존 apply_pca, propagate_majority, compute_metrics 함수는 동일하게 유지]
def apply_pca(df_train, df_test, pca_cols):
    X_tr = df_train[pca_cols].values.astype(np.float32)
    X_te = df_test[pca_cols].values.astype(np.float32)
    cov = np.cov(X_tr.T);
    eigvals, eigvecs = np.linalg.eigh(cov)
    ev = eigvecs[:, np.argmax(eigvals)]
    pc_tr = X_tr @ ev;
    pc_te = X_te @ ev
    mn, mx = pc_tr.min(), pc_tr.max()
    denom = mx - mn if mx > mn else 1.0
    df_train = df_train.copy();
    df_test = df_test.copy()
    df_train["PC"] = 100 * (pc_tr - mn) / denom
    df_test["PC"] = 100 * (pc_te - mn) / denom
    return df_train, df_test


def propagate_majority(pred, dst_df):
    pred = pred.copy().astype(str)
    ip_port = (dst_df["Destination"].astype(str) + ":" + dst_df["Destination Port"].astype(str)).values
    df_tmp = pd.DataFrame({"ip": ip_port, "pred": pred})
    for ip, grp in df_tmp.groupby("ip"):
        if ip == DUMMY_IP_PORT: continue
        majority = SSH_LABEL if (grp["pred"] == SSH_LABEL).sum() > (
                    grp["pred"] == NON_SSH_LABEL).sum() else NON_SSH_LABEL
        pred[grp.index.values] = majority
    return pred


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true);
    y_pred = np.array(y_pred)
    tp = ((y_true == SSH_LABEL) & (y_pred == SSH_LABEL)).sum()
    fp = ((y_true != SSH_LABEL) & (y_pred == SSH_LABEL)).sum()
    fn = ((y_true == SSH_LABEL) & (y_pred != SSH_LABEL)).sum()
    tn = ((y_true != SSH_LABEL) & (y_pred != SSH_LABEL)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def run_evaluation_cycle(df_train, df_test, model_cols, label_col, dst_test, n_repeat):
    """특정 데이터셋에 대해 n_repeat만큼 수행 후 평균 반환"""
    f1_list = []
    prec_list = []
    rec_list = []

    X_tr = df_train[model_cols].values.astype(np.float32)
    y_tr = df_train[label_col].values
    X_te = df_test[model_cols].values.astype(np.float32)
    y_te = df_test[label_col].values

    for _ in range(n_repeat):
        rf = cuRF(n_estimators=N_ESTIMATORS) if USE_CUML else \
            RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        if hasattr(y_pred, "to_numpy"): y_pred = y_pred.to_numpy()

        y_final = propagate_majority(y_pred.astype(str), dst_test)
        m = compute_metrics(y_te, y_final)
        f1_list.append(m['f1']);
        prec_list.append(m['precision']);
        rec_list.append(m['recall'])

    return np.mean(prec_list), np.mean(rec_list), np.mean(f1_list)


def main():
    opt = get_params()
    cfg = DATASET_CONFIG[opt.dataset]
    label_col = cfg["label_col"]

    # 공통 테스트 데이터 로드
    test_file = cfg["test_files"][opt.test_week]
    df_test_raw = load_csv(os.path.join(cfg["real_dir"], test_file), cfg)
    if "Destination" not in df_test_raw.columns:
        df_test_raw["Destination"] = "0.0.0.0";
        df_test_raw["Destination Port"] = 0

    modes = ["original", "wgan_gp", "llm"]
    final_results = []

    for mode in modes:
        print(f"\n>>> [{mode.upper()}] 실험 시작")
        df_train_raw = load_specific_dataset(cfg, opt, mode)

        # 피처 처리 (DARPA일 경우 PCA)
        if opt.dataset == "darpa99":
            df_train, df_test = apply_pca(df_train_raw, df_test_raw, PCA_COLS_DARPA)
            model_cols = MODEL_COLS_DARPA
        else:
            df_train, df_test = df_train_raw, df_test_raw
            model_cols = MODEL_COLS_NB15

        dst_test = df_test[["Destination", "Destination Port"]].copy()

        p, r, f = run_evaluation_cycle(df_train, df_test, model_cols, label_col, dst_test, opt.n_repeat)
        final_results.append({"Mode": mode, "Precision": p, "Recall": r, "F1-Score": f})

    # 최종 결과 출력
    df_res = pd.DataFrame(final_results)
    print(f"\n\n{'=' * 60}")
    print(f"  최종 성능 비교 요약 (Dataset: {opt.dataset.upper()})")
    print(f"{'=' * 60}")
    print(df_res.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()