import os
import pandas as pd

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    print("=" * 55)
    print("  STEP 1: 데이터 로드 시작")
    print("=" * 55)

    print("\n  [Netflix] 데이터 로드 중...")
    netflix_path = "data/netflix_reviews_raw.csv"
    netflix_df = pd.read_csv(netflix_path, encoding="utf-8-sig", low_memory=False)
    netflix_df["platform"] = "Netflix"
    print(f"  -> Netflix: {len(netflix_df):,}건 로드 완료")

    print("\n  [DisneyPlus] 데이터 로드 중...")
    disney_path = "data/disneyplus_reviews_raw.csv"
    disney_df = pd.read_csv(disney_path, encoding="utf-8-sig", low_memory=False)
    disney_df["platform"] = "DisneyPlus"
    print(f"  -> DisneyPlus: {len(disney_df):,}건 로드 완료")

    combined = pd.concat([netflix_df, disney_df], ignore_index=True)
    combined.to_csv(f"{SAVE_DIR}/combined_reviews_raw.csv", index=False, encoding="utf-8-sig")

    print(f"\n  최종 통합 데이터: {len(combined):,}건")
    print(f"  저장 완료: {SAVE_DIR}/combined_reviews_raw.csv")
    print("=" * 55)


if __name__ == "__main__":
    main()