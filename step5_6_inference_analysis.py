import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import matplotlib

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

MODEL_DIR   = "models/mobilebert_ott"
INPUT_PATH  = "data/processed_labeled.csv"
OUTPUT_PATH = "data/predictions.csv"
RESULTS_DIR = "results"
BATCH_SIZE  = 64
MAX_LEN     = 128

os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLATFORM_COLORS = {"Netflix": "#E50914", "DisneyPlus": "#113CCF"}

NEGATIVE_KEYWORDS = {
    "Netflix":    ["price", "expensive", "cost", "cancel", "ads", "slow", "buffer", "account", "share", "increase"],
    "DisneyPlus": ["crash", "lag", "bug", "error", "login", "load", "freeze", "glitch", "broken", "fix"],
}
POSITIVE_KEYWORDS = {
    "Netflix":    ["variety", "content", "quality", "original", "4k", "recommend", "series", "movie", "love", "best"],
    "DisneyPlus": ["marvel", "star wars", "disney", "pixar", "classic", "family", "kids", "exclusive", "love", "amazing"],
}

KEYWORD_TRANSLATION = {
    "price": "요금/가격", "expensive": "비씀", "cost": "비용", "cancel": "해지/취소", "ads": "광고",
    "slow": "느림", "buffer": "버퍼링", "account": "계정", "share": "공유 제한", "increase": "가격 인상",
    "crash": "앱 튕김", "lag": "렉/지연", "bug": "버그", "error": "오류/에러", "login": "로그인 실패",
    "load": "로딩 지연", "freeze": "화면 멈춤", "glitch": "시스템 오류", "broken": "작동 안됨", "fix": "수정 요구",
    "variety": "다양성", "content": "콘텐츠", "quality": "화질/질", "original": "오리지널", "4k": "4K 화질",
    "recommend": "추천 시스템", "series": "시리즈물", "movie": "영화", "love": "매우 만족", "best": "최고의 앱",
    "marvel": "마블(Marvel)", "star wars": "스타워즈", "disney": "디즈니 감성", "pixar": "픽사(Pixar)",
    "classic": "고전 명작", "family": "가족 유저", "kids": "키즈/아이들", "exclusive": "독점 공개", "amazing": "훌륭함"
}


def run_inference(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("  STEP 5: MobileBERT 감성 추론 시작")
    print(f"{'='*55}")

    tokenizer = MobileBertTokenizer.from_pretrained(MODEL_DIR)
    model     = MobileBertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    all_preds, all_probs = [], []
    texts = df["text"].tolist()

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  추론 및 감정 분석 진행"):
        batch_texts = texts[i : i + BATCH_SIZE]
        enc = tokenizer(batch_texts, max_length=MAX_LEN, padding="max_length",
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(enc["input_ids"].to(DEVICE),
                            attention_mask=enc["attention_mask"].to(DEVICE))
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(probs.argmax(axis=1).tolist())
        all_probs.extend(probs[:, 1].tolist())

    df["pred_label"] = all_preds
    df["pos_prob"]   = all_probs
    df["sentiment"]  = df["pred_label"].map({1: "Positive", 0: "Negative"})

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n  추론 완료 및 결과 파일 저장: {OUTPUT_PATH}")
    print(f"  긍정 판정 (Positive): {(df['pred_label'] == 1).sum():,}건")
    print(f"  부정 판정 (Negative): {(df['pred_label'] == 0).sum():,}건")
    return df


def analyze(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print("  STEP 6: 비즈니스 인사이트 시각화 및 분석 시작")
    print(f"{'='*55}")

    df["at"]   = pd.to_datetime(df["at"], errors="coerce")
    df["year"] = df["at"].dt.year

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle("연도별 긍정 감성 리뷰 비율 추이", fontsize=14, fontweight="bold")
    for ax, (platform, group) in zip(axes, df.groupby("platform")):
        yearly = group.groupby("year")["pred_label"].mean() * 100
        bars = ax.bar(yearly.index.astype(str), yearly.values,
                      color=PLATFORM_COLORS[platform], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, yearly.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
        korean_name = "넷플릭스" if platform == "Netflix" else "디즈니플러스"
        ax.set_title(korean_name, color=PLATFORM_COLORS[platform], fontweight="bold", fontsize=12)
        ax.set_xlabel("연도 (Year)"); ax.set_ylabel("긍정 리뷰 비율 (%)")
        ax.set_ylim(0, 100)
        ax.axhline(50, linestyle="--", color="gray", linewidth=0.8, alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/insight_01_sentiment_trend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  저장 완료: insight_01_sentiment_trend.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("플랫폼별 종합 감성 점유율 비교", fontsize=14, fontweight="bold")
    for ax, (platform, group) in zip(axes, df.groupby("platform")):
        pos = (group["pred_label"] == 1).sum()
        neg = (group["pred_label"] == 0).sum()
        ax.pie([pos, neg], labels=["긍정 (Positive)", "부정 (Negative)"],
               colors=[PLATFORM_COLORS[platform], "#d9d9d9"],
               autopct="%1.1f%%", startangle=90,
               wedgeprops={"width": 0.5, "edgecolor": "white"})
        korean_name = "넷플릭스" if platform == "Netflix" else "디즈니플러스"
        ax.set_title(korean_name, color=PLATFORM_COLORS[platform], fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/insight_02_sentiment_donut.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  저장 완료: insight_02_sentiment_donut.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("부정 리뷰의 플랫폼별 주요 불만 키워드 TOP 10", fontsize=14, fontweight="bold")
    for ax, platform in zip(axes, ["Netflix", "DisneyPlus"]):
        neg_texts = df[(df["platform"] == platform) & (df["pred_label"] == 0)]["text"]
        combined  = " ".join(neg_texts.fillna("")).lower()
        kw_counts = {KEYWORD_TRANSLATION[kw]: combined.count(kw) for kw in NEGATIVE_KEYWORDS[platform]}
        kw_df = pd.Series(kw_counts).sort_values(ascending=True)
        bars = ax.barh(kw_df.index, kw_df.values, color=PLATFORM_COLORS[platform], alpha=0.85)
        for bar, val in zip(bars, kw_df.values):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f"{val:,}", va="center", fontsize=9)
        korean_name = "넷플릭스" if platform == "Netflix" else "디즈니플러스"
        ax.set_title(korean_name, color=PLATFORM_COLORS[platform], fontweight="bold", fontsize=12)
        ax.set_xlabel("부정 리뷰 내 언급 빈도수")
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/insight_03_negative_keywords.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  저장 완료: insight_03_negative_keywords.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("긍정 리뷰의 플랫폼별 주요 칭찬 키워드 TOP 10", fontsize=14, fontweight="bold")
    for ax, platform in zip(axes, ["Netflix", "DisneyPlus"]):
        pos_texts = df[(df["platform"] == platform) & (df["pred_label"] == 1)]["text"]
        combined  = " ".join(pos_texts.fillna("")).lower()
        kw_counts = {KEYWORD_TRANSLATION[kw]: combined.count(kw) for kw in POSITIVE_KEYWORDS[platform]}
        kw_df = pd.Series(kw_counts).sort_values(ascending=True)
        bars = ax.barh(kw_df.index, kw_df.values, color=PLATFORM_COLORS[platform], alpha=0.85)
        for bar, val in zip(bars, kw_df.values):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f"{val:,}", va="center", fontsize=9)
        korean_name = "넷플릭스" if platform == "Netflix" else "디즈니플러스"
        ax.set_title(korean_name, color=PLATFORM_COLORS[platform], fontweight="bold", fontsize=12)
        ax.set_xlabel("긍정 리뷰 내 언급 빈도수")
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/insight_04_positive_keywords.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  저장 완료: insight_04_positive_keywords.png")

    print(f"\n  모든 인사이트 시각화 이미지 → results/ 폴더 저장 완료")


def main():
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df = run_inference(df)
    analyze(df)


if __name__ == "__main__":
    main()