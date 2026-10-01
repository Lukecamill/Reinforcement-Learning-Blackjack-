import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import os

# Config

ALGOS = ["monte_carlo", "sarsa", "qlearning", "double_qlearning"]

ALGO_LABELS = {
    "monte_carlo":       "Monte Carlo",
    "sarsa":             "SARSA",
    "qlearning":         "Q-Learning",
    "double_qlearning":  "Double Q-Learning",
}

EPSILON_TYPES = [
    "fixed_0.1",
    "decay_linear",
    "decay_exp1000",
    "decay_exp10000",
]

EPSILON_LABELS = {
    "fixed_0.1":      "Fixed ε = 0.1",
    "decay_linear":   "Decay ε = 1/k",
    "decay_exp1000":  "Decay ε = e^(−k/1000)",
    "decay_exp10000": "Decay ε = e^(−k/10000)",
}

ALGO_COLORS = {
    "monte_carlo":      "#2196F3",
    "sarsa":            "#4CAF50",
    "qlearning":        "#FF9800",
    "double_qlearning": "#9C27B0",
}

os.makedirs("plots", exist_ok=True)

# Helper

def load_history(algo, eps_type):
    path = f"results_{algo}_{eps_type}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Derive per-1k win rate from cumulative columns
    df["WinRate"] = df["CumWins"] / df["Episode"]
    return df


def load_policy(algo, eps_type):
    path = f"policy_{algo}_{eps_type}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


def load_counts(algo, eps_type):
    path = f"counts_{algo}_{eps_type}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None

# Plot 1: Wins / Losses / Draws per run (cumulative, one chart per run)

def plot_win_loss_draw(algo, eps_type):
    df = load_history(algo, eps_type)
    if df is None:
        print(f"  [skip] results_{algo}_{eps_type}.csv not found")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Episode"], df["CumWins"],   label="Wins",   color="#2196F3", linewidth=1.5)
    ax.plot(df["Episode"], df["CumLosses"], label="Losses", color="#F44336", linewidth=1.5)
    ax.plot(df["Episode"], df["CumDraws"],  label="Draws",  color="#9E9E9E", linewidth=1.5)

    ax.set_title(
        f"Cumulative Wins / Losses / Draws\n"
        f"{ALGO_LABELS[algo]}  |  {EPSILON_LABELS[eps_type]}",
        fontsize=13,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = f"plots/plot1_wld_{algo}_{eps_type}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")

# Plot 2: State-action visit counts bar chart (one chart per run)

def plot_visit_counts(algo, eps_type):
    df = load_counts(algo, eps_type)
    if df is None:
        print(f"  [skip] counts_{algo}_{eps_type}.csv not found")
        return

    df_sorted = df.sort_values("Count", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2196F3" if a == "Hit" else "#FF9800" for a in df_sorted["Action"]]
    ax.bar(range(len(df_sorted)), df_sorted["Count"],
           color=colors, width=1.0, edgecolor="none")

    ax.set_title(
        f"State-Action Visit Counts (sorted)\n"
        f"{ALGO_LABELS[algo]}  |  {EPSILON_LABELS[eps_type]}",
        fontsize=13,
    )
    ax.set_xlabel("State-Action pair (sorted by count)")
    ax.set_ylabel("Visit count")
    ax.set_xticks([])

    hit_patch   = mpatches.Patch(color="#2196F3", label="Hit")
    stand_patch = mpatches.Patch(color="#FF9800", label="Stand")
    ax.legend(handles=[hit_patch, stand_patch])
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = f"plots/plot2_counts_{algo}_{eps_type}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")

# Plot 3: Unique state-action pairs — 4-bar chart per algorithm

def plot_unique_pairs():
    fig, axes = plt.subplots(1, len(ALGOS), figsize=(16, 5), sharey=False)

    for ax, algo in zip(axes, ALGOS):
        unique_counts = []
        labels        = []
        for eps_type in EPSILON_TYPES:
            df = load_counts(algo, eps_type)
            if df is None:
                unique_counts.append(0)
            else:
                unique_counts.append(len(df[df["Count"] > 0]))
            labels.append(EPSILON_LABELS[eps_type])

        colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
        bars   = ax.bar(labels, unique_counts, color=colors,
                        edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, unique_counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    str(val), ha="center", va="bottom", fontsize=9)

        ax.set_title(ALGO_LABELS[algo], fontsize=12)
        ax.set_ylabel("Unique ⟨s, a⟩ pairs")
        ax.set_ylim(0, max(unique_counts) * 1.2 if max(unique_counts) > 0 else 10)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Unique State-Action Pairs Explored per Configuration",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    out = "plots/plot3_unique_pairs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

# Plot 4: Strategy heatmap (Player Sum vs Dealer Card)

def plot_strategy_heatmap(algo, eps_type):
    df = load_policy(algo, eps_type)
    if df is None:
        print(f"  [skip] policy_{algo}_{eps_type}.csv not found")
        return

    # Convert 'S' (Stand) to 1, 'H' (Hit) to 0 for coloring
    df["ActionNum"] = df["BestAction"].map({"S": 1, "H": 0})

    for usable_ace in [True, False]:
        subset = df[df["UsableAce"] == usable_ace]
        
        pivot = subset.pivot(index="PlayerSum", columns="DealerCard", values="ActionNum")
        pivot = pivot.reindex(index=np.arange(12, 22), columns=np.arange(2, 12))
        
        fig, ax = plt.subplots(figsize=(8, 6)) # Slightly wider to fit the legend comfortably
        
        # 1. FIX DEPRECATION & IMPROVE COLORS: Use explicit, solid hex colors
        # Red (#d62728) for Hit (0), Blue (#1f77b4) for Stand (1)
        cmap = mcolors.ListedColormap(['#d62728', '#1f77b4'])
        cmap.set_bad(color='#e0e0e0') # Solid light grey for unvisited states
        
        cax = ax.imshow(
            pivot.values, 
            cmap=cmap, 
            vmin=0, vmax=1, 
            origin="lower", 
            extent=[1.5, 11.5, 11.5, 21.5],
            aspect="auto"
        )

        # 2. ADD ANNOTATIONS ('H' or 'S') INSIDE THE CELLS
        for player_sum in range(12, 22):
            for dealer_card in range(2, 12):
                val = pivot.loc[player_sum, dealer_card]
                if not np.isnan(val):
                    # Determine letter based on the value
                    text = "S" if val == 1 else "H"
                    # Plot text exactly in the center of the cell coordinates
                    ax.text(dealer_card, player_sum, text, 
                            ha="center", va="center", 
                            color="white", fontweight="bold", fontsize=10)

        # Axis ticks and labels
        ax.set_xticks(np.arange(2, 12))
        ax.set_xticklabels(["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"])
        ax.set_yticks(np.arange(12, 22))
        
        ax.set_xlabel("Dealer Showing", fontweight="bold")
        ax.set_ylabel("Player Sum", fontweight="bold")
        
        ace_str = "Usable Ace" if usable_ace else "No Usable Ace"
        ax.set_title(f"Learned Strategy: {ALGO_LABELS[algo]}\n{EPSILON_LABELS[eps_type]} ({ace_str})", pad=15)

        # Custom Legend matching our new solid colors
        hit_patch = mpatches.Patch(color='#d62728', label='Hit (H)')
        stand_patch = mpatches.Patch(color='#1f77b4', label='Stand (S)')
        unvisited_patch = mpatches.Patch(color='#e0e0e0', label='Unvisited')
        ax.legend(handles=[stand_patch, hit_patch, unvisited_patch], bbox_to_anchor=(1.05, 1), loc='upper left')

        # bbox_inches='tight' ensures the external legend doesn't get cut off when saving
        fig.tight_layout()
        
        ace_file_str = "ace" if usable_ace else "noace"
        out = f"plots/plot4_strategy_{algo}_{eps_type}_{ace_file_str}.png"
        fig.savefig(out, dpi=150, bbox_inches='tight') 
        plt.close(fig)
        print(f"  Saved {out}")

# Plot 5: Dealer advantage — all 16 configs on one grouped bar chart

def plot_dealer_advantage():
    path = "dealer_advantages_all.csv"
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return

    df = pd.read_csv(path)

    x      = np.arange(len(EPSILON_TYPES))
    width  = 0.18
    offset = np.linspace(-1.5, 1.5, len(ALGOS)) * width

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, algo in enumerate(ALGOS):
        values = []
        for eps_type in EPSILON_TYPES:
            key = f"{algo}_{eps_type}"
            # match on combined Algorithm+EpsilonType key
            mask = (df["Algorithm"] + "_" + df["EpsilonType"]) == key
            row  = df[mask]
            values.append(float(row["DealerAdvantage"].values[0]) if len(row) > 0 else 0.0)

        bars = ax.bar(
            x + offset[i], values, width,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.004 if val >= 0 else -0.012),
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([EPSILON_LABELS[e] for e in EPSILON_TYPES],
                       rotation=15, ha="right")
    ax.set_ylabel("Dealer Advantage  (L − W) / (L + W)")
    ax.set_title(
        "Dealer Advantage per Algorithm & Configuration\n"
        "(last 10,000 episodes)",
        fontsize=13,
    )
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = "plots/plot5_dealer_advantage_all.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")

# Plot 6: Comparative win-rate line chart — all 4 algos, one per epsilon

def plot_comparative_winrate():
    for eps_type in EPSILON_TYPES:
        fig, ax = plt.subplots(figsize=(11, 5))
        any_data = False

        for algo in ALGOS:
            df = load_history(algo, eps_type)
            if df is None:
                continue
            ax.plot(
                df["Episode"], df["WinRate"],
                label=ALGO_LABELS[algo],
                color=ALGO_COLORS[algo],
                linewidth=1.5,
            )
            any_data = True

        if not any_data:
            plt.close(fig)
            continue

        ax.set_title(
            f"Cumulative Win Rate — All Algorithms\n{EPSILON_LABELS[eps_type]}",
            fontsize=13,
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative win rate")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = f"plots/plot6_winrate_compare_{eps_type}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")

# Run all plots

if __name__ == "__main__":
    print("=== Plot 1: Cumulative Wins / Losses / Draws ===")
    for algo in ALGOS:
        for eps_type in EPSILON_TYPES:
            plot_win_loss_draw(algo, eps_type)

    print("\n=== Plot 2: State-Action Visit Counts ===")
    for algo in ALGOS:
        for eps_type in EPSILON_TYPES:
            plot_visit_counts(algo, eps_type)

    print("\n=== Plot 3: Unique State-Action Pairs ===")
    plot_unique_pairs()

    print("\n=== Plot 4: Strategy Heatmaps ===")
    for algo in ALGOS:
        for eps_type in EPSILON_TYPES:
            plot_strategy_heatmap(algo, eps_type)

    print("\n=== Plot 5: Dealer Advantage (grouped bar) ===")
    plot_dealer_advantage()

    print("\n=== Plot 6: Comparative Win Rate per Epsilon Strategy ===")
    plot_comparative_winrate()

    print("\nAll plots saved to /plots/")