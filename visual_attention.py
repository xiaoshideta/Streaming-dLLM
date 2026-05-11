import matplotlib.pyplot as plt
import torch
import glob
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

colors = ["#0076BA", "#AFC0D6", "#E7EDF4", "#FAEFD9", "#DF8F79"]
files = sorted(glob.glob("hum_attn_records_31/sample_*.pt"))[:100]
attn_list = [torch.load(f) for f in files]

attn_all = torch.cat(attn_list, dim=0)   # [100*32, 712]

mean_attn = attn_all.mean(dim=0)
q25 = attn_all.quantile(0.25, dim=0)
q75 = attn_all.quantile(0.75, dim=0)

K_len = mean_attn.shape[0]
key_idx = torch.arange(K_len)

# ===== Figure =====
plt.rcParams.update({
    "font.size": 9,
    "axes.linewidth": 1.2,
})

plt.figure(figsize=(4.5, 3.0))

# Uncertainty band (IQR)
plt.fill_between(
    key_idx,
    q25,
    q75,
    color="#AEB7CF", 
    alpha=0.55,
    linewidth=0
)

# Mean attention
plt.plot(
    key_idx,
    mean_attn,
    color="#2F3A5F",
    linewidth=1.5
)

# Focus window
plt.ylim(0, 0.015)
plt.yticks([0.0, 0.005, 0.010, 0.015])
plt.xticks([0, 150, 300, 450, 600])



# Method anchors
current_pos = 100
suffix_offset = 131

legend_handles = [
    Line2D(
        [0], [0],
        color="#2F3A5F",
        linewidth=1.5,
        label="Mean attention over heads"
    ),
    Patch(
        facecolor="#AEB7CF",
        edgecolor="none",
        alpha=0.55,
        label="Interquartile range (25–75%)"
    ),
]

plt.legend(
    handles=legend_handles,
    frameon=True,
    edgecolor="0.9",
    # frameon=False,
    fontsize=8,
    loc="upper right"
)

plt.axvline(
    suffix_offset,
    linestyle="--",
    color="gray",
    alpha=0.3, 
    linewidth=0.8,
    zorder=5
)

plt.axvline(
    0,
    linestyle="--",
    color="gray",
    alpha=0.3, 
    linewidth=0.8,
    zorder=5
)

plt.axvline(
    100,
    linestyle="--",
    color="gray",
    alpha=0.3, 
    linewidth=0.8,
    zorder=5
)

plt.axvline(
    612,
    linestyle="--",
    color="gray",
    alpha=0.3, 
    linewidth=0.8,
    zorder=5
)


y_text = 0.0078
plt.text(
    current_pos / 2,
    y_text,
    "Prefix Area",
    ha="center",
    va="bottom",
    fontsize=4.5,
    color="#555555"
)

plt.text(
    (suffix_offset + K_len) / 2,
    y_text,
    "Suffix Area",
    ha="center",
    va="bottom",
    fontsize=4.5,
    color="#555555"
)

# Labels
plt.xlabel("Key index")
plt.ylabel("Attention weight")

ax = plt.gca()
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(1.2)
ax.tick_params(axis="both", labelsize=9)

plt.tight_layout()
plt.savefig("hum_fig_attention_31.pdf", dpi=300, bbox_inches="tight")
plt.savefig("hum_fig_attention_31.png", dpi=300, bbox_inches="tight")
plt.close()
