"""Chat-bubble drawing for the prototype figures.

The axes is set up in inches (0..W, 0..H) so layout maths stays simple.
Answer text keeps the model's own **markdown bold** (drawn bold) and gets a
coloured highlight behind any word that matches a keyword pattern.
"""
import re

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: F401 (re-exported)

FONT = ["Inter", "WenQuanYi Zen Hei", "DejaVu Sans"]
INK = "#1d1d1f"
MUTED = "#6e6e73"
GREY_FILL = "#f2f2f4"
GREY_EDGE = "#c7c7cc"

plt.rcParams["font.family"] = FONT
plt.rcParams["svg.fonttype"] = "path"

PUNCT = set(",.;:!?()\"“”")
_CJK = re.compile(r"[　-ヿ㐀-鿿＀-￯]")


def canvas(w, h, dpi=150):
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


class Measure:
    def __init__(self, fig):
        self.fig = fig
        self.r = fig.canvas.get_renderer()
        self.cache = {}

    def width(self, s, size, weight="normal"):
        key = (s, size, weight)
        if key not in self.cache:
            t = self.fig.text(0, 0, s, fontsize=size, fontweight=weight)
            self.cache[key] = t.get_window_extent(self.r).width / self.fig.dpi
            t.remove()
        return self.cache[key]


def shorten(text, limit):
    """Cut an answer at a word boundary near `limit` visible characters.
    Returns the cut text with a trailing ellipsis when anything was removed
    (or when the model's own output stopped mid-sentence at the token cap)."""
    vis = 0
    out = []
    i = 0
    cut = False
    while i < len(text):
        if text.startswith("**", i):
            out.append("**")
            i += 2
            continue
        ch = text[i]
        if vis >= limit and (ch.isspace() or _CJK.match(ch)):
            cut = True
            break
        out.append(ch)
        vis += 1
        i += 1
    s = "".join(out).rstrip()
    if s.count("**") % 2:
        s += "**"
    s = re.sub(r"\s+", " ", s)
    if cut or not re.search(r"[.!?。！？]$", s.replace("**", "")):
        s = s.rstrip(" ,;:—-") + " …"
    return s


def _tokens(text, highlights):
    """Split into tokens: (string, bold, highlight_colour, space_before)."""
    clean, bold = [], []
    on = False
    i = 0
    while i < len(text):
        if text.startswith("**", i):
            on = not on
            i += 2
            continue
        clean.append(text[i])
        bold.append(on)
        i += 1
    s = "".join(clean)
    hl = [None] * len(s)
    for pattern, colour in highlights:
        for m in re.finditer(pattern, s):
            for j in range(m.start(), m.end()):
                hl[j] = hl[j] or colour
    toks = []
    j = 0
    space = False
    while j < len(s):
        ch = s[j]
        if ch.isspace():
            space = True
            j += 1
            continue
        if _CJK.match(ch):
            k = j + 1
            # keep CJK punctuation attached to the preceding character
            while k < len(s) and s[k] in "，。、；：！？）」』》":
                k += 1
        elif s[j] in PUNCT:
            k = j + 1
        else:
            k = j
            while k < len(s) and not s[k].isspace() and not _CJK.match(s[k]) and s[k] not in PUNCT:
                k += 1
            # apostrophes and dots inside a word (Google's, Qwen3.5) stay attached
            while k < len(s) and s[k] in ".'" and k + 1 < len(s) and s[k + 1].isalnum():
                k += 1
                while k < len(s) and not s[k].isspace() and not _CJK.match(s[k]) and s[k] not in PUNCT:
                    k += 1
        idx = range(j, k)
        b = sum(bold[x] for x in idx) * 2 >= len(idx)
        colour = next((hl[x] for x in idx if hl[x]), None)
        toks.append((s[j:k], b, colour, space))
        space = False
        j = k
    return toks


def layout(meas, text, width, size, highlights=()):
    """Return (lines, height). Each line is a list of (x, tok, bold, colour)."""
    lh = size / 72 * 1.42
    sp = meas.width("a a", size) - meas.width("aa", size)
    lines, cur, x = [], [], 0.0
    for tok, b, colour, space in _tokens(text, highlights):
        w = meas.width(tok, size, "bold" if b else "normal")
        gap = sp if (space and cur) else 0.0
        if cur and x + gap + w > width and tok not in PUNCT:
            lines.append(cur)
            cur, x, gap = [], 0.0, 0.0
        cur.append((x + gap, tok, b, colour, w))
        x += gap + w
    if cur:
        lines.append(cur)
    return lines, len(lines) * lh


def draw_text(ax, lines, x0, ytop, size, color=INK):
    lh = size / 72 * 1.42
    pad_x, pad_y = 0.035, size / 72 * 0.14
    for li, line in enumerate(lines):
        yb = ytop - (li + 1) * lh + lh * 0.24
        # merge neighbouring highlighted tokens into one block
        runs = []
        for (x, tok, b, colour, w) in line:
            if colour and runs and runs[-1][2] == colour and x - runs[-1][1] < 0.2:
                runs[-1][1] = x + w
            elif colour:
                runs.append([x, x + w, colour])
        for a, bnd, colour in runs:
            ax.add_patch(FancyBboxPatch(
                (x0 + a - pad_x, yb - pad_y), bnd - a + 2 * pad_x, size / 72 * 1.05 + pad_y,
                boxstyle="round,pad=0,rounding_size=0.05", fc=colour, ec="none", zorder=3))
        for (x, tok, b, colour, w) in line:
            ax.text(x0 + x, yb, tok, fontsize=size, fontweight="bold" if b else "normal",
                    color=color, va="baseline", ha="left", zorder=4)


def bubble(ax, meas, x, ytop, width, text, size=14, highlights=(), side="left",
           fc="white", ec=GREY_EDGE, tc=INK, lw=1.4, pad=0.16, tail=True, ls="-"):
    """Draw a chat bubble whose top-left corner is (x, ytop). Returns its height."""
    lines, th = layout(meas, text, width - 2 * pad - 0.12, size, highlights)
    real_w = max(l[-1][0] + l[-1][4] for l in lines) + 2 * pad if len(lines) == 1 else width
    if side == "right":
        x = x + width - real_w
    h = th + 2 * pad
    ax.add_patch(FancyBboxPatch((x, ytop - h), real_w, h,
                                boxstyle="round,pad=0,rounding_size=0.16",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    if tail and fc == ec:
        if side == "left":
            pts = [(x + 0.05, ytop - h + 0.28), (x - 0.13, ytop - h + 0.02), (x + 0.3, ytop - h + 0.05)]
        else:
            pts = [(x + real_w - 0.05, ytop - h + 0.28), (x + real_w + 0.13, ytop - h + 0.02),
                   (x + real_w - 0.3, ytop - h + 0.05)]
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec="none", zorder=2.5))
    draw_text(ax, lines, x + pad, ytop - pad, size, tc)
    return h


def dots(ax, x, ytop, colours, per_row, r=0.07, gap=0.05, edges=None, marks=None):
    """Grid of small circles, filled in reading order. Returns (width, height)."""
    step = 2 * r + gap
    for i, c in enumerate(colours):
        cx = x + r + (i % per_row) * step
        cy = ytop - r - (i // per_row) * step
        ec = edges[i] if edges else "none"
        ax.add_patch(plt.Circle((cx, cy), r, fc=c, ec=ec, lw=1.0, zorder=3))
        if marks and marks[i]:
            d = r * 0.62
            ax.plot([cx - d, cx + d], [cy - d, cy + d], color=marks[i], lw=1.3, zorder=4)
            ax.plot([cx - d, cx + d], [cy + d, cy - d], color=marks[i], lw=1.3, zorder=4)
    rows = (len(colours) + per_row - 1) // per_row
    return per_row * step - gap, rows * step - gap


def outline(ax, x, y, w, h, color, label=None, size=12, ls=(0, (5, 3))):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                                fc="none", ec=color, lw=2.4, ls=ls, zorder=6))
    if label:
        ax.text(x + 0.18, y + h, f" {label} ", fontsize=size, fontweight="bold", color=color,
                va="center", ha="left", zorder=7,
                bbox=dict(fc="white", ec="none", pad=1.5))


def swatch(ax, x, y, colour, label, size=13):
    ax.add_patch(Rectangle((x, y - 0.09), 0.36, 0.22, fc=colour, ec="none"))
    ax.text(x + 0.46, y, label, fontsize=size, color=INK, va="center")
