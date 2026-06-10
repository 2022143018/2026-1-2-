import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def main():
    print("step7_ott_subscriber 시작")

    years = ["2020", "2021", "2022", "2023", "2024", "2025"]
    netflix_changes = [3657, 1818, 891, 2953, 4172, 2300]
    disney_changes = [3350, 8460, 4610, -1460, -2430, 630]

    df = pd.DataFrame(
        {"Year": years, "Netflix": netflix_changes, "Disney+": disney_changes}
    )

    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

    x = np.arange(len(years))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    rects1 = ax.bar(x - width / 2, df["Netflix"], width, label="Netflix", color="#E50914")
    rects2 = ax.bar(x + width / 2, df["Disney+"], width, label="Disney+", color="#113CCF")

    ax.set_title("Netflix vs Disney+ Annual Subscriber Net Additions (2020-2025)", fontsize=16, pad=20)
    ax.set_xlabel("Year", fontsize=12, labelpad=10)
    ax.set_ylabel("Subscriber Change (Ten Thousand / 만 명)", fontsize=12, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=11)
    ax.legend(fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            va = "bottom" if height >= 0 else "top"
            xytext = (0, 3 if height >= 0 else -14)

            ax.annotate(
                f"{height:+,}만",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=xytext,
                textcoords="offset points",
                ha="center", va=va, fontsize=10, weight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, 'ott_subscriber_chart.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"그래프'{save_path}'저장.")

if __name__ == "__main__":
    main()