import re
import pandas as pd
from tqdm import tqdm

INPUT_PATH  = "data/combined_reviews_raw.csv"
OUTPUT_PATH = "data/processed_labeled.csv"

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

def clean_text(text: str) -> str:
    text = str(text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s!?.,']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def assign_label(score: int):
    if score <= 2:
        return 0
    elif score >= 4:
        return 1
    return None

def main():
    print("="*55)
    print("  STEP 3: 전처리 및 라벨링 시작")
    print("="*55)

    df = pd.read_csv(INPUT_PATH, parse_dates=["at"], low_memory=False)
    print(f"\n  원본 데이터: {len(df):,}건")

    df = df.drop_duplicates(subset=["userName", "content"])
    print(f"  중복 제거 후: {len(df):,}건")

    df = df.dropna(subset=["content", "score"])
    df = df[df["content"].str.strip().ne("")]
    print(f"  결측치 제거 후: {len(df):,}건")

    tqdm.pandas(desc="  텍스트 클리닝")
    df["text_clean"] = df["content"].progress_apply(clean_text)

    df = df[df["text_clean"].str.len() >= 15]
    print(f"  단문 제거 후: {len(df):,}건")

    def assign_label(score: int):
        if score <= 2:
            return 0
        elif score >= 4:
            return 1
        return None

    df["label"] = df["score"].astype(int).apply(assign_label)

    before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    print(f"  중립(3★) 제거 후: {len(df):,}건 (제거: {before - len(df):,}건)")

    df = df[["text_clean", "label", "score", "platform", "at"]].reset_index(drop=True)
    df = df.rename(columns={"text_clean": "text"})

    for platform, group in df.groupby("platform"):
        pos = (group["label"] == 1).sum()
        neg = (group["label"] == 0).sum()
        print(f"\n  [{platform}]  Positive: {pos:,}  Negative: {neg:,}  비율: {pos/neg:.2f}")

    balanced_parts = []
    for platform, group in df.groupby("platform"):
        pos = group[group["label"] == 1]
        neg = group[group["label"] == 0]
        max_neg = min(len(neg), len(pos) * 3)
        neg_sampled = neg.sample(n=max_neg, random_state=42)
        balanced_parts.extend([pos, neg_sampled])

    df = pd.concat(balanced_parts, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  최종 데이터셋: {len(df):,}건")
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()