"""Draw docs/architecture.png (system architecture diagram).
    python -m src.make_architecture_diagram
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.config import PROJECT_ROOT


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=color, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#333"))


def main():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(-0.5, 6.2); ax.axis("off")
    ax.text(6, 5.95, "System Architecture", ha="center", fontsize=14, weight="bold")

    ax.text(0.2, 5.45, "TRAINING PIPELINE  (python -m src.train)", fontsize=9, weight="bold")
    t = [("Raw CSV\n(6,335 articles)", "#e9ecef"),
         ("Data cleaning\n(drop blank, dupes,\nconflicts)", "#dbeafe"),
         ("Train/Test split\n80/20 stratified\nseed=42", "#fde68a"),
         ("Preprocessing\nlowercase, no URLs,\nstop-words out", "#dbeafe"),
         ("TF-IDF (1-2 grams)\nfit on TRAIN only", "#dcfce7"),
         ("NB / Logistic Reg.\n(+ VADER variant)", "#dcfce7")]
    for i, (txt, col) in enumerate(t):
        box(ax, 0.2 + i * 1.95, 4.15, 1.6, 1.05, txt, col)
        if i: arrow(ax, 0.2 + i * 1.95 - 0.33, 4.67, 0.2 + i * 1.95, 4.67)

    box(ax, 7.3, 2.75, 2.2, 0.95, "Evaluation on TEST set\naccuracy, P/R/F1,\nconfusion matrix, CV", "#fecaca")
    box(ax, 9.9, 2.75, 1.9, 0.95, "Saved artefacts\nmodels/*.joblib\nreports/*", "#e9ecef")
    arrow(ax, 11.0, 4.15, 8.6, 3.72); arrow(ax, 9.5, 3.22, 9.9, 3.22)

    ax.text(0.2, 2.35, "PREDICTION PIPELINE  (Streamlit app)", fontsize=9, weight="bold")
    p = [("User input\n(headline/article)", "#e9ecef"),
         ("Validation\n(empty, short,\ninvalid)", "#fecaca"),
         ("Same preprocessing\nfunction as training", "#dbeafe"),
         ("Saved TF-IDF\n+ saved model", "#dcfce7"),
         ("Prediction\nFAKE / REAL\n+ probability", "#fde68a")]
    for i, (txt, col) in enumerate(p):
        box(ax, 0.2 + i * 1.95, 0.95, 1.6, 1.05, txt, col)
        if i: arrow(ax, 0.2 + i * 1.95 - 0.33, 1.47, 0.2 + i * 1.95, 1.47)
    box(ax, 1.4, -0.35, 5.0, 0.75, "VADER sentiment on RAW text\n(separate module; does not decide FAKE/REAL)", "#f3e8ff")
    arrow(ax, 1.5, 0.95, 2.6, 0.42)
    box(ax, 9.0, -0.35, 2.9, 1.25, "Results page\nverdict + confidence\nsentiment + score\ndisclaimer", "#fde68a")
    arrow(ax, 8.8, 0.95, 9.9, 0.92); arrow(ax, 6.4, 0.0, 9.0, 0.2)
    arrow(ax, 10.2, 2.75, 7.0, 2.02)  # artefacts -> saved model
    ax.text(9.0, 2.3, "load once", fontsize=8, style="italic")

    out = PROJECT_ROOT / "docs" / "architecture.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
