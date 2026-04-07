"""
LM Studio 기반 Zero-Shot SSH 트래픽 합성
Algorithm 1 구현 (Local LLM Version)

사용법:
    python lm_studio_synthesis.py \
        --original "training dataset_week4.csv" \
        --model    "lmstudio-community/Meta-Llama-3.2-3B-Instruct-GGUF" \
        --batch_size 20 \
        --output   llm_studio_ssh_1000.csv
"""

import json
import time
import argparse
import pandas as pd
import numpy as np
from openai import OpenAI

# ── LM Studio 로컬 서버 연결 ────────────────────────────────────────────────
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

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

# 원본 피처명 ↔ 로그 변환 피처명 매핑 (프롬프트 설명용)
FEATURE_DESC = {
    'log_count_total_connect': (
        'count_total_connect (log-scaled)',
        'Number of connections to the same destination IP'
    ),
    'log_cs_byte': (
        'byte_send (log-scaled)',
        'Total transmitted data size per session'
    ),
    'log_transmit_speed_BPS': (
        'speed_transmit_BPS (log-scaled)',
        'Average data transmission speed per session [byte_send / time_taken]'
    ),
    'log_count_connect_IP': (
        'count_connect_IP (log-scaled)',
        'Number of source IPs connected to the same destination IP'
    ),
    'log_avg_count_connect': (
        'count_avg_connect (log-scaled)',
        'Average connections per source IP [count_total_connect / count_connect_IP]'
    ),
    'log_time_taken': (
        'time_taken (log-scaled)',
        'Total session duration ≈ (N-1) × mean_of_IPA_time'
    ),
    'log_time_interval_mean': (
        'mean_of_IPA_time μ (log-scaled)',
        'Mean of inter-packet arrival time: μ = (1/N-1) × Σ IPA_i'
    ),
    'log_time_interval_var': (
        'var_of_IPA_time σ² (log-scaled)',
        'Variance of inter-packet arrival time: σ² = (1/N-1) × Σ(IPA_i - μ)²'
    ),
    'log_ratio_trans_receive': (
        'ratio_trans_receive (log-scaled)',
        'Transmit-to-receive byte ratio [byte_send / byte_receive]'
    ),
}


# ── Phase 1: 통계 프로파일 추출 ──────────────────────────────────────────────
def extract_stats(df_ssh):
    stats = df_ssh[FEATURES].describe()
    print("\n[Phase 1] 통계 프로파일 추출 완료")
    print(stats.loc[['mean', 'std', 'min', 'max']].round(4).to_string())
    return stats


# ── Phase 2: 피처 간 관계 제약 (R1~R5) ──────────────────────────────────────
def get_feature_relations():
    """
    Algorithm 1 Phase 2
    논문 수식 (1)~(6) 및 SSH 행동 패턴을 자연어로 인코딩
    실제 샘플 없이 관계 지식만 제공 → Zero-Shot
    """
    return """
================================================================================
FEATURE DEFINITIONS AND INTER-FEATURE CONSTRAINTS
(All values are log-scaled to [0, 100] range)
================================================================================

────────────────────────────────────────────────────────────────────────────────
[1] CONNECTION STATISTICS
────────────────────────────────────────────────────────────────────────────────

  Constraint R1:
    count_avg_connect = count_total_connect / count_connect_IP

  Behavioral pattern:
    - HIGH count_total_connect + LOW count_connect_IP
      → Concentrated access to few IPs (potential attack or automation)
    - LOW count_total_connect + HIGH count_connect_IP
      → Distributed scanning behavior

  Therefore in log-scaled features:
    log_count_total_connect and log_count_connect_IP are inversely
    correlated in attack scenarios, and
    log_avg_count_connect must be consistent with their ratio.

────────────────────────────────────────────────────────────────────────────────
[2] DATA TRANSFER AND SPEED STATISTICS
────────────────────────────────────────────────────────────────────────────────

  Constraint R2 (Speed):
    speed_transmit_BPS = byte_send / time_taken

    - A record with LARGE byte_send and SHORT time_taken
      MUST have proportionally HIGH speed_transmit_BPS
    - A record with SMALL byte_send and LONG time_taken
      MUST have LOW speed_transmit_BPS
    - log_transmit_speed_BPS must be coherent with
      log_cs_byte and log_time_taken values

  Constraint R3 (Transmit/Receive Ratio):
    ratio_trans_receive = byte_send / byte_receive

    SSH session type patterns:
    - Interactive session (human typing):
        ratio_trans_receive ∈ [0.3, 0.7]
        (user sends commands, receives more data back)
    - Bulk file upload session:
        ratio_trans_receive > 1.0
        (more data sent than received)
    - Remote execution / scripted session:
        ratio_trans_receive ≈ 0.1 ~ 0.3
        (short commands, large output received)

────────────────────────────────────────────────────────────────────────────────
[3] TIME AND INTER-PACKET ARRIVAL (IPA) STATISTICS
────────────────────────────────────────────────────────────────────────────────

  Definitions (N = total number of packets in session):

    IPA_i = arrival time of packet (i+1) - arrival time of packet i

    Constraint R4 (IPA Mean):
      mean_of_IPA_time (μ) = (1 / N-1) × Σ IPA_i  for i=1 to N-1

    Constraint R5 (IPA Variance):
      var_of_IPA_time (σ²) = (1 / N-1) × Σ (IPA_i - μ)²  for i=1 to N-1

    Constraint R6 (Session Time):
      time_taken ≈ (N-1) × mean_of_IPA_time
      → Session duration is approximately the sum of all packet intervals
      → LONGER sessions MUST have HIGHER mean_of_IPA_time or more packets

  SSH session behavior patterns:

    - Human interactive SSH (manual typing):
        HIGH log_time_interval_mean  (slow, human-paced input)
        HIGH log_time_interval_var   (irregular typing rhythm)
        MODERATE log_time_taken      (session lasts minutes)

    - Automated / scripted SSH session:
        LOW log_time_interval_mean   (fast, machine-paced packets)
        LOW log_time_interval_var    (regular, consistent intervals)
        SHORT log_time_taken         (rapid execution and exit)

    - File transfer over SSH (SCP/SFTP):
        LOW log_time_interval_mean   (continuous data stream)
        LOW log_time_interval_var    (steady transfer rate)
        HIGH log_cs_byte             (large data volume)
        HIGH log_transmit_speed_BPS  (sustained high speed)

================================================================================
COHERENCE REQUIREMENT:
  Generated records MUST satisfy all constraints R1~R6 simultaneously.
  Inconsistent records (e.g., high speed with low bytes and long duration)
  are INVALID and must NOT be generated.
================================================================================
"""


# ── Phase 3: 시스템 프롬프트 ────────────────────────────────────────────────
def build_system_prompt():
    """Algorithm 1 Phase 3"""
    return """You are a network security expert and synthetic dataset generator \
specializing in SSH encrypted traffic analysis.

Your task is to generate synthetic SSH network traffic feature records that are \
statistically consistent with real SSH traffic from the DARPA99 dataset.

STRICT OUTPUT RULES:
1. Output ONLY a valid JSON array. No explanation, no markdown, no code blocks.
2. The output must start with '[' and end with ']'.
3. Each JSON object must contain EXACTLY these 9 numeric keys:
     log_count_total_connect
     log_cs_byte
     log_transmit_speed_BPS
     log_count_connect_IP
     log_avg_count_connect
     log_time_taken
     log_time_interval_mean
     log_time_interval_var
     log_ratio_trans_receive
4. All values must be positive floating-point numbers.
5. All values must fall within the [min, max] range provided.
6. All inter-feature constraints (R1~R6) must be satisfied.
7. Generate diverse records — do NOT repeat identical values.
8. Do NOT include IP addresses, hostnames, or any raw packet content."""


# ── Phase 4: 사용자 프롬프트 생성 (S2T 변환) ────────────────────────────────
def build_user_prompt(stats, batch_size, batch_num, total_batches):
    """
    Algorithm 1 Phase 4
    실제 샘플 없이 통계 + 관계 제약만 제공 → Zero-Shot S2T
    """
    lines = []

    # ── 생성 요청 ─────────────────────────────────────────────
    lines.append(
        f"Generate exactly {batch_size} synthetic SSH network traffic records "
        f"(batch {batch_num} of {total_batches}).\n"
    )

    # ── 통계 프로파일 (Zero-Shot: 실제 샘플 없음) ─────────────
    lines.append("=" * 72)
    lines.append("STATISTICAL PROFILE OF REAL SSH TRAFFIC (DARPA99 session dataset)")
    lines.append("NOTE: No actual traffic samples are provided. Use statistics only.")
    lines.append("=" * 72)
    lines.append("")

    for f in FEATURES:
        orig_name, desc = FEATURE_DESC[f]
        mu  = stats.loc['mean', f]
        sig = stats.loc['std',  f]
        mn  = stats.loc['min',  f]
        mx  = stats.loc['max',  f]
        q1  = stats.loc['25%',  f]
        q2  = stats.loc['50%',  f]
        q3  = stats.loc['75%',  f]

        lines.append(f"  Feature key : {f}")
        lines.append(f"  Represents  : {orig_name}")
        lines.append(f"  Description : {desc}")
        lines.append(f"  Mean ± Std  : {mu:.4f} ± {sig:.4f}")
        lines.append(f"  Range       : [{mn:.4f}, {mx:.4f}]")
        lines.append(f"  Q1/Q2/Q3    : {q1:.4f} / {q2:.4f} / {q3:.4f}")
        lines.append("")

    # ── 피처 관계 제약 삽입 ───────────────────────────────────
    lines.append(get_feature_relations())

    # ── 출력 지시 ─────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append(f"Now output a JSON array of exactly {batch_size} objects.")
    lines.append("Ensure statistical consistency and inter-feature coherence.")
    lines.append("Output the JSON array only:")

    return "\n".join(lines)


# ── LM Studio API 호출 ───────────────────────────────────────────────────────
def call_lm_studio(system_prompt, user_prompt, model_name, temperature=0.7):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()


# ── JSON 파싱 (로컬 LLM 불완전 출력 방어) ───────────────────────────────────
def safe_parse_json(text, batch_num):
    text = text.strip()

    # 마크다운 코드블록 제거
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                text = part
                break

    # JSON 배열 범위 추출
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start:end+1]

    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        # 불완전한 마지막 객체 제거 후 재시도
        last_comma = text.rfind(",", 0, len(text) - 5)
        if last_comma > 0:
            try:
                return json.loads(text[:last_comma] + "]")
            except Exception:
                pass
        print(f"  ⚠  배치 {batch_num}: JSON 파싱 실패")
        return []


# ── 메인 합성 함수 ───────────────────────────────────────────────────────────
def synthesize_ssh_traffic(original_csv="training dataset_week4.csv", output_csv="llm_studio_ssh_1000.csv", n_samples=1000, batch_size=20,
                           model_name="lLlama 3.2 3B Instruct GGUF Q4_K_M", temperature=0.7, seed=42, print_prompt=False):
    print("=" * 60)
    print("  LM Studio Zero-Shot SSH Traffic Synthesis")
    print("  Algorithm 1  |  S2T Prompt (No sample data)")
    print("=" * 60)

    # ── 원본 데이터 6020237
    # ─────────────────────────────────────
    #print(f"\n원본 데이터 로딩: {original_csv}")
    df       = pd.read_csv(original_csv, index_col=0)
    df_ssh   = df[df['LABEL'] == 'ssh'][FEATURES].dropna()
    if len(df_ssh) >= n_samples:
        df_ssh = df_ssh.sample(n=n_samples, random_state=seed)
    print(f"SSH 샘플: {len(df_ssh)}건 선택\n")

    # Phase 1
    stats = extract_stats(df_ssh)

    # Phase 3
    system_prompt = build_system_prompt()

    # 프롬프트 미리보기
    if print_prompt:
        print("\n" + "=" * 60)
        print("  [SYSTEM PROMPT 미리보기]")
        print("=" * 60)
        print(system_prompt)
        print("\n" + "=" * 60)
        print("  [USER PROMPT Batch 1 미리보기]")
        print("=" * 60)
        print(build_user_prompt(stats, batch_size, 1,
              (n_samples + batch_size - 1) // batch_size))

    # Phase 4: 배치 루프
    n_batches   = (n_samples + batch_size - 1) // batch_size
    all_records = []
    failed      = 0

    print(f"\n[Phase 4] 배치 생성: {batch_size}건 × {n_batches}회")
    print("-" * 60)

    for k in range(1, n_batches + 1):
        current_batch = min(batch_size, n_samples - len(all_records))
        if current_batch <= 0:
            break

        user_prompt = build_user_prompt(stats, current_batch, k, n_batches)

        try:
            raw  = call_lm_studio(system_prompt, user_prompt,
                                   model_name, temperature)
            recs = safe_parse_json(raw, k)

            if recs:
                all_records.extend(recs[:current_batch])
                print(f"  배치 {k:2d}/{n_batches}: {len(recs):3d}건 생성 "
                      f"→ 누적 {len(all_records):4d}건")
            else:
                print(f"  배치 {k:2d}/{n_batches}: ⚠  파싱 실패")
                failed += 1

        except Exception as e:
            print(f"  배치 {k:2d}/{n_batches}: ❌ 오류 → {e}")
            failed += 1

        time.sleep(0.3)

    if not all_records:
        print("\n❌ 생성 실패: 레코드 없음")
        return None

    # ── 후처리 ───────────────────────────────────────────────
    df_llm  = pd.DataFrame(all_records)
    missing = [f for f in FEATURES if f not in df_llm.columns]
    if missing:
        print(f"⚠  누락 피처: {missing}")
        return None

    df_llm = df_llm[FEATURES].dropna()
    s      = df_ssh[FEATURES].describe()
    for f in FEATURES:
        df_llm[f] = df_llm[f].clip(
            lower=s.loc['min', f] * 0.95,
            upper=s.loc['max', f] * 1.05
        )

    if len(df_llm) > n_samples:
        df_llm = df_llm.sample(n=n_samples, random_state=seed)

    df_llm.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  완료: {len(df_llm)}건 저장 → {output_csv}")
    print(f"  실패 배치: {failed}회 / {n_batches}회")
    print(f"{'='*60}\n")

    print("다음 단계: t-SNE 비교 실행")
    print(f"  python tsne_comparison.py \\")
    print(f"    --original  \"{original_csv}\" \\")
    print(f"    --wgan_gp   \"fake_ssh_1_2026-03-10-15.csv\" \\")
    print(f"    --llm_csv   \"{output_csv}\"")

    return df_llm


# ── 실행 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original',      default='training dataset_week4.csv')
    parser.add_argument('--output',        default='llm_studio_ssh_1000.csv')
    parser.add_argument('--n_samples',     type=int,   default=1000)
    parser.add_argument('--batch_size',    type=int,   default=20)
    parser.add_argument('--model',         default='lLlama 3.2 3B Instruct GGUF Q4_K_M',
                        help='LM Studio에 로드된 모델명')
    parser.add_argument('--temperature',   type=float, default=0.7)
    parser.add_argument('--print_prompt',  action='store_true',
                        help='배치 1 프롬프트 미리보기 출력')
    args = parser.parse_args()

    synthesize_ssh_traffic(
        original_csv = args.original,
        output_csv   = args.output,
        n_samples    = args.n_samples,
        batch_size   = args.batch_size,
        model_name   = args.model,
        temperature  = args.temperature,
        print_prompt = args.print_prompt
    )