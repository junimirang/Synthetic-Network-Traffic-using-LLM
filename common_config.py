# -*- coding: utf-8 -*-
"""
공통 설정 — dataset_config.py
모든 프로그램에서 import하여 사용
"""

# ══════════════════════════════════════════════════════
# 데이터셋별 설정
# ══════════════════════════════════════════════════════

DATASET_CONFIG = {
    "darpa99": {
        "real_dir":       "dataset-DARPA99",
        "fake_dir":       "fake dataset-DARPA99",
        "ckpt_dir":       "checkpoints/dataset-DARPA99",
        "result_dir":     "result/dataset-DARPA99",
        "train_file":     "training dataset_week4_normalized.csv",
        "test_files": {
            "week13": "test dataset_week1-3_normalized.csv",
            "week5":  "test dataset_week5_normalized.csv",
        },
        "feature_cols": [
            "log_time_interval_mean",
            "log_time_interval_var",
            "log_time_taken",
            "log_cs_byte",
            "log_ratio_trans_receive",
            "log_count_connect_IP",
            "log_count_total_connect",
            "log_avg_count_connect",
            "log_transmit_speed_BPS",
        ],
        "label_col":  "LABEL",
        "N_FEATURES": 9,
        "N_TOTAL":    10,   # 피처 9 + label 1
    },
    "nb15": {
        "real_dir":       "dataset-NB15",
        "fake_dir":       "fake dataset-NB15",
        "ckpt_dir":       "checkpoints/NB15",
        "result_dir":     "result/NB15",
        "train_file":     "UNSW_NB15_new_training_log_normalized.csv",
        "test_files": {
            "test": "UNSW_NB15_new_testing_log_normalized.csv",
        },
        "feature_cols": [
            "log_mean_of_IPA_time",
            "log_ratio_trans_receive",
            "log_speed_transmit_connect",
            "log_dur",
            "log_sbytes",
            "log_ct_src_dport_ltm",
            "log_ct_dst_ltm",
            "log_ct_src_ltm",
        ],
        "label_col":  "new_label",
        "N_FEATURES": 8,
        "N_TOTAL":    9,    # 피처 8 + label 1
    },
}

# ══════════════════════════════════════════════════════
# 모델별 하위 폴더명 (수정됨)
# ══════════════════════════════════════════════════════
# common_config.py 내부
MODEL_DIR = {
    "wgan_gp":         "wgan-gp",
    "vae_wgan_gp":     "vae-wgan-gp",
    "vae_wgan_gp_lcl": "vae-wgan-gp-lcl",
    "llm":             "LLM"  # <-- 이 부분이 'LLM'이어야 'fake dataset-DARPA99/LLM'을 찾습니다.
}

# 레이블 값
SSH_LABEL     = "ssh"
NON_SSH_LABEL = "non-ssh"

# 공통 하이퍼파라미터
LAMBDA_GP  = 10
LATENT_DIM = 10
N_CRITIC   = 5
BATCH_SIZE = 512
LR         = 0.0002
N_BINS     = 10      # VEE / JSD 히스토그램 bin 수