import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    MobileBertTokenizer,
    MobileBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG = {
    "model_name"       : "google/mobilebert-uncased",
    "max_len"          : 128,
    "batch_size"       : 16,
    "epochs"           : 4,
    "lr"               : 2e-5,
    "warmup_ratio"     : 0.1,
    "test_size"        : 0.15,
    "val_size"         : 0.15,
    "seed"             : 42,
    "save_dir"         : "models/mobilebert_ott",
    "results_dir"      : "results",
    "sample_per_group" : 700,
}
os.makedirs(CONFIG["save_dir"],    exist_ok=True)
os.makedirs(CONFIG["results_dir"], exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {DEVICE}")


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "token_type_ids" : encoding.get(
                "token_type_ids",
                torch.zeros(self.max_len, dtype=torch.long)
            ).squeeze(0),
            "label"          : torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(DEVICE)
            mask   = batch["attention_mask"].to(DEVICE)
            ttype  = batch["token_type_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(ids, attention_mask=mask, token_type_ids=ttype, labels=labels)
            total_loss += outputs.loss.item()
            preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
            trues.extend(labels.cpu().numpy())

    acc  = np.mean(np.array(preds) == np.array(trues))
    loss = total_loss / len(loader)
    return acc, loss, preds, trues


def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    df_full = pd.read_csv("data/processed_labeled.csv", low_memory=False)
    print(f"\n  전처리 완료 데이터 전체: {len(df_full):,}건")

    print(f"\n{'='*55}")
    print("  STEP 4-0: 학습용 샘플 추출")
    print(f"{'='*55}")

    parts = []
    for plat in ["Netflix", "DisneyPlus"]:
        for lbl in [0, 1]:
            subset = df_full[(df_full["platform"] == plat) & (df_full["label"] == lbl)]
            parts.append(subset.sample(min(len(subset), CONFIG["sample_per_group"]), random_state=CONFIG["seed"]))

    df_sample = pd.concat(parts).sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)
    df_sample.to_csv("data/train_sample.csv", index=False, encoding="utf-8-sig")

    print(f"\n  [플랫폼 x 감성별 샘플 분포]")
    for plat in ["Netflix", "DisneyPlus"]:
        pos = len(df_sample[(df_sample["platform"]==plat) & (df_sample["label"]==1)])
        neg = len(df_sample[(df_sample["platform"]==plat) & (df_sample["label"]==0)])
        print(f"  {plat}: 긍정={pos}건, 부정={neg}건")

    print(f"\n  총 학습 샘플: {len(df_sample):,}건")
    print(f"  긍정 비율  : {(df_sample['label']==1).mean()*100:.1f}%")
    print(f"  부정 비율  : {(df_sample['label']==0).mean()*100:.1f}%")
    print(f"  저장 완료  : data/train_sample.csv")

    texts  = df_sample["text"].tolist()
    labels = df_sample["label"].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels,
        test_size=CONFIG["test_size"] + CONFIG["val_size"],
        stratify=labels,
        random_state=CONFIG["seed"],
    )
    val_ratio = CONFIG["val_size"] / (CONFIG["test_size"] + CONFIG["val_size"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_ratio,
        stratify=y_temp,
        random_state=CONFIG["seed"],
    )
    print(f"\n  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    print(f"\n  MobileBERT 로드 중: {CONFIG['model_name']}")
    tokenizer = MobileBertTokenizer.from_pretrained(CONFIG["model_name"])
    model     = MobileBertForSequenceClassification.from_pretrained(
        CONFIG["model_name"], num_labels=2
    ).to(DEVICE)

    train_ds = ReviewDataset(X_train, y_train, tokenizer, CONFIG["max_len"])
    val_ds   = ReviewDataset(X_val,   y_val,   tokenizer, CONFIG["max_len"])
    test_ds  = ReviewDataset(X_test,  y_test,  tokenizer, CONFIG["max_len"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],   shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"]*2, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"]*2, shuffle=False, num_workers=2)

    optimizer    = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
    total_steps  = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} [Train]")

        for batch in pbar:
            ids    = batch["input_ids"].to(DEVICE)
            mask   = batch["attention_mask"].to(DEVICE)
            ttype  = batch["token_type_ids"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(ids, attention_mask=mask, token_type_ids=ttype, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += outputs.loss.item()
            pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        val_acc, val_loss, _, _ = evaluate(model, val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\n  Epoch {epoch}: train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(CONFIG["save_dir"])
            tokenizer.save_pretrained(CONFIG["save_dir"])
            print(f"  ★ Best 모델 저장 (val_acc={best_val_acc:.4f})")

    print(f"\n{'='*55}")
    print("  최종 테스트 평가")
    print(f"{'='*55}")
    model = MobileBertForSequenceClassification.from_pretrained(CONFIG["save_dir"]).to(DEVICE)
    test_acc, test_loss, preds, trues = evaluate(model, test_loader)
    print(f"  Test Accuracy : {test_acc:.4f}  {'✓ 목표 달성!' if test_acc >= 0.85 else '✗ 목표 미달성'}")
    print("\n  분류 리포트 (상세 성적표):")

    report_dict = classification_report(
        trues, preds,
        target_names=["부정 (Negative)", "긍정 (Positive)"],
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose().rename(columns={
        "precision": "정밀도 (Precision)",
        "recall"   : "재현율 (Recall)",
        "f1-score" : "F1-스코어",
        "support"  : "데이터 개수 (Support)",
    })
    print(report_df.round(4).to_string())

    cm = confusion_matrix(trues, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"], ax=ax)
    ax.set_title(f"Confusion Matrix (Test Acc: {test_acc:.4f})", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{CONFIG['results_dir']}/confusion_matrix.png", dpi=150)
    plt.close()

    epochs_x = range(1, CONFIG["epochs"] + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_x, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs_x, history["val_loss"],   "r-o", label="Val Loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs_x, history["val_acc"], "g-o", label="Val Accuracy")
    ax2.axhline(0.85, linestyle="--", color="red", label="Target (0.85)")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{CONFIG['results_dir']}/training_curves.png", dpi=150)
    plt.close()

    print(f"\n  시각화 저장 완료 -> results/")


if __name__ == "__main__":
    main()