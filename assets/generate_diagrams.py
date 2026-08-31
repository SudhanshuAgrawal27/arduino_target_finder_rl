"""Regenerates the non-slide diagrams under assets/ used by README.md.

assets/grid-image.png is a screenshot of the problem-setup board from
slides/slides.html (kept as a static image since the slide deck itself
isn't part of the published repo) and is not touched by this script.

Run:
    python3 assets/generate_diagrams.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG = "#eef0f4"
CARD = "#ffffff"
INK = "#171a21"
INK_SOFT = "#565f6f"
LINE = "#d8dbe2"
ACCENT = "#a6690f"
ACCENT_SOFT = "#f4ece0"


def _card(ax, xy, width, height):
    ax.add_patch(FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0,rounding_size=0.15",
        linewidth=1, edgecolor=LINE, facecolor=CARD, zorder=1,
    ))


def _box(ax, xy, width, height, title, lines):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0,rounding_size=0.08",
        linewidth=1.4, edgecolor=INK, facecolor=ACCENT_SOFT, zorder=2,
    ))
    ax.text(x + width / 2, y + height - 0.35, title,
            ha="center", va="top", fontsize=13.5, fontweight="bold", color=INK)
    for i, line in enumerate(lines):
        ax.text(x + 0.3, y + height - 0.85 - i * 0.42, line,
                ha="left", va="top", fontsize=10.5, color=INK_SOFT, family="monospace")


def make_system_diagram(out_file="assets/system-diagram.png"):
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    _card(ax, (0.15, 0.15), 10.7, 4.5)

    host_xy, host_w, host_h = (0.7, 0.7), 4.1, 3.3
    ard_xy, ard_w, ard_h = (6.2, 0.7), 4.1, 3.3

    _box(ax, host_xy, host_w, host_h, "HOST MACHINE (python)", [
        "GridEnvironment",
        "  simulator.py",
        "ActorCritic policy",
        "  network.py",
        "Serial client",
        "  led_board_client.py",
    ])
    _box(ax, ard_xy, ard_w, ard_h, "ARDUINO", [
        "LED matrix display",
        "  WS2812B / MAX7219",
        "Photoresistor (LDR)",
        "  proximity sensor",
        "Serial listener",
        "  led_serial_listener.ino",
    ])

    mid_y_top = host_xy[1] + host_h * 0.68
    mid_y_bot = host_xy[1] + host_h * 0.32
    x0 = host_xy[0] + host_w
    x1 = ard_xy[0]

    ax.annotate("", xy=(x1, mid_y_top), xytext=(x0, mid_y_top),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))
    ax.text((x0 + x1) / 2, mid_y_top + 0.18, "agent / target position",
            ha="center", va="bottom", fontsize=9.5, color=ACCENT)

    ax.annotate("", xy=(x0, mid_y_bot), xytext=(x1, mid_y_bot),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, lw=2))
    ax.text((x0 + x1) / 2, mid_y_bot - 0.32, "LDR brightness reading",
            ha="center", va="top", fontsize=9.5, color=INK_SOFT)

    ax.text((x0 + x1) / 2, host_xy[1] + host_h + 0.22, "USB serial · 115200 baud",
            ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")

    ax.text(5.5, 0.35,
            "Training runs fully in simulation. This link is used only during evaluation,\n"
            "to mirror an episode onto real LEDs and, optionally, close the loop with a live LDR reading.",
            ha="center", va="bottom", fontsize=9, color=INK_SOFT, style="italic")

    fig.tight_layout()
    fig.savefig(out_file, facecolor=BG)
    print(f"wrote {out_file}")


if __name__ == "__main__":
    make_system_diagram()
