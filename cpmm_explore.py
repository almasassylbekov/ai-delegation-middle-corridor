"""First look at CPMM trade-facilitation indicators for Kazakhstan rail (2010-2024)."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

b = pd.read_csv("CPMM_BCP_Indicators.csv")
c = pd.read_csv("CPMM_Country_Indicators.csv")
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9), constrained_layout=True)
for name, col, mk in [("Dostyk", BLUE, "o"), ("Altynkol", ORANGE, "s")]:
    d = b[(b.country_code == "KAZ") & (b.transport_mode == "rail") & (b.bcp_name == name)]
    d = d.dropna(subset=["tfi1_inbound_average"]).sort_values("year")
    a1.plot(d.year, d.tfi1_inbound_average, color=col, lw=2, marker=mk, ms=5, label=f"{name} (inbound)")
reg = c[(c.country_code == "REG") & (c.transport_mode == "rail")].dropna(subset=["tfi1_overall_average"])
a1.plot(reg.year, reg.tfi1_overall_average, color=GREY, lw=2, ls="--", label="CAREC average (rail)")
a1.axvspan(2020.5, 2023.5, color="#f0efec", zorder=0)
a1.text(2022, 5, "2021–2023", ha="center", color=INK2, fontsize=8)
a1.set_title("(a) Time to clear rail border crossing (hours)", loc="left", color=INK, fontsize=10)
a1.legend(frameon=False, fontsize=8, loc="upper left")

k = c[(c.country_code == "KAZ") & (c.transport_mode == "rail")].sort_values("year")
share = 100 * (1 - k.tfi4_overall_average / k.swod_overall_average)
a2.plot(k.year, share, color=AQUA, lw=2, marker="^", ms=5)
a2.axvspan(2020.5, 2023.5, color="#f0efec", zorder=0)
for yr in (2019, 2022, 2024):
    v = share[k.year == yr].iloc[0]
    a2.annotate(f"{v:.0f}%", (yr, v), textcoords="offset points", xytext=(0, 7), ha="center", color=INK, fontsize=8)
a2.set_ylim(0, 100)
a2.set_title("(b) Kazakhstan rail: share of transit time lost to delays (%)", loc="left", color=INK, fontsize=10)
for ax in (a1, a2):
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)
fig.text(0.01, -0.04, "Source: CAREC CPMM trade facilitation indicators (TFI1, TFI4, SWOD), own calculation. "
         "(b) = 1 − speed with delay / speed without delay.", color=INK2, fontsize=7.5)
fig.savefig("cpmm_kazakhstan_rail.png", dpi=170, bbox_inches="tight")
print("saved")
