import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nltk
from nltk.corpus import stopwords
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PLATFORM_COLORS = {
    "Netflix": "#E50914",
    "DisneyPlus": "#113CCF",
}


def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def get_top_words(text_stream, top_n=10):
    words = [w for w in text_stream.split() if len(w) > 2]
    counts = Counter(words)
    return [f"{word} ({count}번)" for word, count in counts.most_common(top_n)]


def main():
    df = pd.read_csv("data/combined_reviews_raw.csv", parse_dates=["at"], low_memory=False)
    df = df.dropna(subset=["content", "score", "platform"])
    df["score"] = df["score"].astype(int)
    df["text_len"] = df["content"].str.len()
    df["at"] = df["at"].dt.tz_localize(None)
    df["year"] = df["at"].dt.year
    df["year_month"] = df["at"].dt.to_period("M").astype(str)

    print(f"총 리뷰 수: {len(df):,}건")
    stats = df.groupby("platform")["score"].describe().round(2)
    stats.columns = ["개수(count)", "평균(mean)", "표준편차(std)", "최솟값(min)", "25%", "50%", "75%", "최댓값(max)"]
    korean_map = {"DisneyPlus": "디즈니플러스", "Netflix": "넷플릭스"}
    stats.index = [korean_map.get(i, i) for i in stats.index]
    print(stats)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("플랫폼별 별점 리뷰 분포 (2020–2025)", fontsize=14, fontweight="bold")

    for ax, (platform, group) in zip(axes, df.groupby("platform")):
        counts = group["score"].value_counts().sort_index()
        bars = ax.bar(
            counts.index, counts.values,
            color=PLATFORM_COLORS[platform], alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 100,
                f"{val:,}", ha="center", va="bottom", fontsize=9,
            )
        korean_name = "넷플릭스" if platform == "Netflix" else "디즈니플러스"
        ax.set_title(korean_name, fontsize=12, fontweight="bold", color=PLATFORM_COLORS[platform])
        ax.set_xlabel("별점 평점 (1–5점)")
        ax.set_ylabel("리뷰 개수")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/eda_01_rating_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("저장: eda_01_rating_distribution.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    unique_years = sorted(df["year"].unique())

    for platform, group in df.groupby("platform"):
        yearly = group.groupby("year").size()
        korean_name = "넷플릭스" if platform == "Netflix" else "디즈니플러스"
        ax.plot(
            yearly.index, yearly.values,
            marker="o", markersize=6, linewidth=2.2,
            color=PLATFORM_COLORS[platform], label=korean_name,
        )
        for x, y in zip(yearly.index, yearly.values):
            ax.text(
                x, y + (max(df.groupby(["platform", "year"]).size()) * 0.02),
                f"{y:,}", ha="center", fontsize=9, fontweight="bold",
                color=PLATFORM_COLORS[platform],
            )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xticks(unique_years)
    min_yr, max_yr = df["year"].min(), df["year"].max()
    ax.set_title(f"플랫폼별 연도별 리뷰 수 추이 ({min_yr}–{max_yr})", fontsize=13, fontweight="bold")
    ax.set_xlabel("연도 (Year)")
    ax.set_ylabel("리뷰 등록 수")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(min_yr - 0.3, max_yr + 0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/eda_02_yearly_trend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: eda_02_yearly_trend.png (적용 연도: {min_yr}~{max_yr})")

    fig, ax = plt.subplots(figsize=(10, 5))
    for platform, group in df.groupby("platform"):
        korean_name = "넷플릭스" if platform == "Netflix" else "디즈니플러스"
        sns.kdeplot(
            group["text_len"].clip(0, 1000), ax=ax, label=korean_name,
            color=PLATFORM_COLORS[platform], fill=True, alpha=0.25, linewidth=2,
        )
    ax.set_title("리뷰 텍스트 글자 수 분포", fontsize=13, fontweight="bold")
    ax.set_xlabel("글자 수 (1,000자 이상은 제한 처리)")
    ax.set_ylabel("밀도 (Density)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/eda_03_text_length.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("저장: eda_03_text_length.png")

    print("\n" + "=" * 55)
    print("  플랫폼별 리뷰 주요 키워드 TOP 10 (표)")
    print("=" * 55)

    summary_data = {}
    for platform in ["Netflix", "DisneyPlus"]:
        pdata = df[df["platform"] == platform]
        pos_text = " ".join(pdata[pdata["score"] >= 4]["content"].dropna().apply(clean_text))
        neg_text = " ".join(pdata[pdata["score"] <= 2]["content"].dropna().apply(clean_text))
        korean_name = "넷플릭스" if platform == "Netflix" else "디즈니플러스"
        summary_data[f"{korean_name}_긍정(4-5★)"] = get_top_words(pos_text, top_n=10)
        summary_data[f"{korean_name}_부정(1-2★)"] = get_top_words(neg_text, top_n=10)

    keyword_df = pd.DataFrame(summary_data)
    keyword_df.index = [f"{i}위" for i in range(1, 11)]
    print(keyword_df.to_string())
    keyword_df.to_csv(f"{RESULTS_DIR}/eda_04_top_keywords.csv", encoding="utf-8-sig")
    print(f"\n저장 완료: {RESULTS_DIR}/eda_04_top_keywords.csv")

    print("\n" + "=" * 55)
    print("  EDA 요약 통계")
    print("=" * 55)
    for platform, group in df.groupby("platform"):
        korean_name = "넷플릭스" if platform == "Netflix" else "디즈니플러스"
        print(f"\n  [{korean_name}]")
        print(f"    리뷰 수       : {len(group):,}건")
        print(f"    평균 별점     : {group['score'].mean():.2f}")
        print(f"    긍정(4-5★)   : {(group['score'] >= 4).sum():,}건 ({(group['score'] >= 4).mean()*100:.1f}%)")
        print(f"    부정(1-2★)   : {(group['score'] <= 2).sum():,}건 ({(group['score'] <= 2).mean()*100:.1f}%)")
        print(f"    평균 텍스트 길이: {group['text_len'].mean():.0f}자")
    print("\n  모든 EDA 데이터 및 이미지 → results/ 폴더 저장 완료")


if __name__ == "__main__":
    main()