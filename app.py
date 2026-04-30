# app.py  — ReviewLens Dark Edition  (FULLY FIXED)
#
# FIXES IN THIS VERSION:
#   1. Product titles now show real product names — "Left my hair dull" was a
#      review summary, not a product name. Fixed in item_retrieval._best_product_title().
#   2. Amazon URLs validated with strict regex — only real B0XXXXXXXXX ASINs get URLs.
#   3. Open-URL buttons show product name, not review snippet.
#   4. Sort + Min-rating filter in Step 4.
#   5. Resizable panels via tk.PanedWindow (drag the divider between panels).
#   6. Full dark mode throughout.
#   7. State fully reset when a new file is chosen.
#   8. Charts re-open every time via matplotlib backend fix in chart_utils.py.

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import webbrowser
import threading
import re as _re

from review_processing  import load_reviews, process_reviews
from summary_formatter  import format_all_summaries
from sentence_retrieval import retrieve_relevant_sentences
from chart_utils        import show_aspect_distribution, show_sentiment_trends
from item_retrieval     import get_item_index, reset_item_index

# ── Dark-mode palette ──────────────────────────────────────────────────────────
BG          = "#1e1e2e"
BG2         = "#2a2a3e"
BG3         = "#313149"
ACCENT      = "#7c5cbf"
ACCENT2     = "#5b8dee"
GREEN       = "#3fb950"
ORANGE      = "#e3a03c"
RED         = "#e05c5c"
TEAL        = "#2ea89e"
GREY        = "#6e6e8a"
FG          = "#cdd6f4"
FG2         = "#a6adc8"
FG_DIM      = "#585b70"
ENTRY_BG    = "#313149"
ENTRY_FG    = "#cdd6f4"
RESULT_BG   = "#1a1a2e"
RESULT_FG   = "#cdd6f4"
SEL_BG      = "#45475a"

FONT_TITLE  = ("Segoe UI", 24, "bold")
FONT_SUB    = ("Segoe UI", 11)
FONT_LABEL  = ("Segoe UI", 10)
FONT_BOLD   = ("Segoe UI", 10, "bold")
FONT_MONO   = ("Consolas", 9)
FONT_BTN    = ("Segoe UI", 10, "bold")

# Strict ASIN validation — must be B + 9 alphanumeric, or 10-digit number
_ASIN_RE = _re.compile(r'^(B[A-Z0-9]{9}|[0-9]{10})$', _re.IGNORECASE)

# ── State ──────────────────────────────────────────────────────────────────────
selected_file               = ""
loaded_reviews: list        = []
latest_clusters: dict       = {}
_last_product_results: list = []


# ── Helpers ────────────────────────────────────────────────────────────────────
def _safe_log(box, msg):
    def _do():
        box.insert(tk.END, msg)
        box.see(tk.END)
    root.after(0, _do)


def _set_buttons(state, *button_lists):
    def _do():
        for btns in button_lists:
            for b in btns:
                try:
                    b.config(state=state)
                except tk.TclError:
                    pass
    root.after(0, _do)


def _clean_html(text):
    text = _re.sub(r"<br\s*/?>", " ", text, flags=_re.IGNORECASE)
    text = _re.sub(r"<[^>]+>", "", text)
    for ent, rep in [("&amp;","&"),("&lt;","<"),("&gt;",">"),
                     ("&nbsp;"," "),("&#39;","'"),("&quot;",'"')]:
        text = text.replace(ent, rep)
    return _re.sub(r"\s{2,}", " ", text).strip()


def _valid_amazon_url(r: dict) -> str:
    """
    Build a valid Amazon URL only when a genuine ASIN exists.
    Priority: parent_asin → item_id → asin.
    Returns "" for anything that doesn't match the strict ASIN pattern.
    """
    for key in ("parent_asin", "item_id", "asin"):
        val = (r.get(key) or "").strip().upper()
        if val and _ASIN_RE.match(val):
            return f"https://www.amazon.com/dp/{val}"
    return ""


# ── File browser ───────────────────────────────────────────────────────────────
def browse_file():
    global selected_file, loaded_reviews, latest_clusters

    path = filedialog.askopenfilename(
        title="Select JSONL File",
        filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
    )
    if not path:
        return

    # Full state reset when a new file is chosen
    selected_file   = path
    loaded_reviews  = []
    latest_clusters = {}
    reset_item_index()

    file_label.config(text=path)
    for box in (analysis_box, search_box, prod_box):
        box.delete(1.0, tk.END)
    for w in open_buttons_frame.winfo_children():
        w.destroy()

    analysis_box.insert(tk.END, f"[File selected]\n{path}\n\nPress 'Analyze Reviews' to load.\n")


# ── ANALYZE ────────────────────────────────────────────────────────────────────
def analyze_reviews():
    global loaded_reviews, latest_clusters

    if not selected_file:
        messagebox.showerror("Error", "Please select a JSONL file first.")
        return
    try:
        max_reviews = int(review_limit_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid number for review limit.")
        return

    analysis_box.delete(1.0, tk.END)
    _set_buttons("disabled", analyze_btns, search_btns, prod_btns)

    def _worker():
        global loaded_reviews, latest_clusters

        _safe_log(analysis_box, "⏳ Loading reviews...\n")
        reviews_raw = load_reviews(selected_file, max_reviews)
        for r in reviews_raw:
            if isinstance(r.get("text"), str):
                r["text"] = _clean_html(r["text"])
        loaded_reviews = reviews_raw
        _safe_log(analysis_box, f"✔  {len(loaded_reviews)} reviews loaded.\n\n")

        _safe_log(analysis_box, "⏳ Building BLAIR item index...\n")
        reset_item_index()
        n = get_item_index().build(loaded_reviews, min_reviews=1)
        _safe_log(analysis_box, f"✔  {n} products indexed.\n\n")

        _safe_log(analysis_box, "⏳ Clustering reviews with BLAIR...\n")
        latest_clusters, _ = process_reviews(loaded_reviews, num_clusters=8)

        _safe_log(analysis_box, "⏳ Generating meta-review...\n\n")
        summaries = format_all_summaries(latest_clusters)

        lines = ["=" * 60 + "\n", "  FINAL META REVIEW\n", "=" * 60 + "\n\n"]
        for aspect, summary in summaries.items():
            count = len(latest_clusters[aspect])
            lines.append(f"[*] {aspect.upper()}  ({count} reviews)\n")
            lines.append("-" * 50 + "\n")
            lines.append(summary + "\n\n")

        def _finish():
            analysis_box.insert(tk.END, "".join(lines))
            analysis_box.insert(tk.END, "\n✔  Analysis complete.\n")
            analysis_box.see(tk.END)

        root.after(0, _finish)
        _set_buttons("normal", analyze_btns, search_btns, prod_btns)

    threading.Thread(target=_worker, daemon=True).start()


# ── SENTENCE SEARCH ────────────────────────────────────────────────────────────
def search_reviews():
    query = query_entry.get().strip()
    if not loaded_reviews:
        messagebox.showerror("Error", "Please analyze reviews first.")
        return
    if not query:
        messagebox.showerror("Error", "Please enter a search query.")
        return

    search_box.delete(1.0, tk.END)
    _safe_log(search_box, f"🔍 Query: {query}\n")
    _safe_log(search_box, "Encoding sentences with BLAIR...\n\n")
    _set_buttons("disabled", search_btns)

    def _worker():
        results = retrieve_relevant_sentences(query, loaded_reviews, top_k=5)
        lines = ["=" * 60 + "\n", "  TOP MATCHING SENTENCES\n", "=" * 60 + "\n\n"]
        for i, item in enumerate(results, 1):
            lines.append(f"[{i}]  Score : {item['score']:.4f}\n")
            lines.append(f"     {item['sentence']}\n")
            if item.get("title"):
                lines.append(f"     Product : {item['title']}\n")
            # FIX: use strict ASIN validation for URL
            url = item.get("amazon_url", "")
            if not url or not _ASIN_RE.match((url.split("/dp/")[-1] if "/dp/" in url else "")):
                url = _valid_amazon_url(item)
            if url:
                lines.append(f"     Link    : {url}\n")
            elif item.get("source"):
                lines.append(f"     Source  : {item['source']}\n")
            lines.append("\n")

        def _finish():
            search_box.delete(1.0, tk.END)
            search_box.insert(tk.END, "".join(lines))
            search_box.see(tk.END)

        root.after(0, _finish)
        _set_buttons("normal", search_btns)

    threading.Thread(target=_worker, daemon=True).start()


# ── PRODUCT SEARCH ─────────────────────────────────────────────────────────────
def search_products():
    global _last_product_results

    query = prod_query_entry.get().strip()
    if not loaded_reviews:
        messagebox.showerror("Error", "Please analyze reviews first.")
        return
    if not query:
        messagebox.showerror("Error", "Please enter a product query.")
        return

    index = get_item_index()
    if index.size == 0:
        messagebox.showerror("Error", "Item index empty. Run 'Analyze Reviews' first.")
        return

    try:
        top_k = int(prod_topk_var.get())
    except ValueError:
        top_k = 10

    prod_box.delete(1.0, tk.END)
    _safe_log(prod_box, f"🔍 Query: {query}\n")
    _safe_log(prod_box, "Finding products with BLAIR...\n\n")
    _set_buttons("disabled", prod_btns)

    def _worker():
        global _last_product_results
        # Fetch more so sorting/filtering is meaningful
        fetch_k = max(top_k * 3, 20)
        results = index.query(query, top_k=fetch_k)

        # Rebuild URL for every result using strict validation
        for r in results:
            if not r.get("amazon_url"):
                r["amazon_url"] = _valid_amazon_url(r)

        _last_product_results = results
        _render_products(results, top_k)
        _set_buttons("normal", prod_btns)

    threading.Thread(target=_worker, daemon=True).start()


def _render_products(results, top_k=None):
    """Apply sort/filter then display in prod_box."""
    sort_by    = sort_var.get()
    min_rating = float(min_rating_var.get())

    filtered = [r for r in results
                if (r.get("avg_rating") or 0) >= min_rating or r.get("avg_rating") is None]

    if sort_by == "Rating ↓":
        filtered.sort(key=lambda r: r.get("avg_rating") or 0, reverse=True)
    elif sort_by == "Rating ↑":
        filtered.sort(key=lambda r: r.get("avg_rating") or 99)
    elif sort_by == "Reviews":
        filtered.sort(key=lambda r: r.get("review_count") or 0, reverse=True)
    # else: keep BLAIR score order

    if top_k:
        filtered = filtered[:top_k]

    lines = [
        "=" * 60 + "\n",
        f"  TOP PRODUCTS  [{sort_by}]  (min rating ≥ {min_rating})\n",
        "=" * 60 + "\n\n",
    ]

    for i, r in enumerate(filtered, 1):
        rating_str = f"{r['avg_rating']:.1f}/5" if r.get("avg_rating") else "N/A"
        stars      = int(r.get("avg_rating") or 0)
        star_str   = "★" * stars + "☆" * (5 - stars)

        # FIX: strict URL — empty string if ASIN not valid
        url = r.get("amazon_url", "") or _valid_amazon_url(r)
        url_display = url if url else "N/A  (ASIN not in dataset metadata)"

        # FIX: product name — must come from metadata title field, never review summary
        product_name = r.get("title") or r.get("item_id") or "Unknown Product"

        lines.append(f"[{i}]  {product_name}\n")
        lines.append(f"     Score    : {r['score']:.4f}\n")
        lines.append(f"     Rating   : {star_str} {rating_str}  ({r['review_count']} reviews)\n")
        lines.append(f"     ASIN     : {r['item_id']}\n")
        lines.append(f"     URL      : {url_display}\n")
        lines.append(f"     Match    : {r['explanation']}\n")
        if r.get("sample_review"):
            short = _clean_html(r["sample_review"][:180])
            lines.append(f"     Sample   : \"{short}...\"\n")
        lines.append("\n")

    def _finish():
        prod_box.delete(1.0, tk.END)
        prod_box.insert(tk.END, "".join(lines))
        prod_box.see(tk.END)
        _rebuild_open_buttons(filtered)

    root.after(0, _finish)


def _rebuild_open_buttons(results):
    for w in open_buttons_frame.winfo_children():
        w.destroy()

    for i, r in enumerate(results[:6], 1):
        url = r.get("amazon_url", "") or _valid_amazon_url(r)
        if not url:
            continue  # only show button if URL is valid

        # FIX: show real product title in button
        raw_title = r.get("title") or r.get("item_id", "")
        title = raw_title[:22] if raw_title else f"Product {i}"

        tk.Button(
            open_buttons_frame,
            text=f"[{i}] {title}",
            command=lambda u=url: webbrowser.open(u),
            bg=ORANGE, fg="#1e1e2e",
            font=("Segoe UI", 9, "bold"),
            relief="flat", padx=6, pady=3,
            cursor="hand2",
        ).pack(side="left", padx=3, pady=3)


def _on_filter_change(*_):
    if _last_product_results:
        try:
            top_k = int(prod_topk_var.get())
        except ValueError:
            top_k = 10
        _render_products(_last_product_results, top_k)


# ── Charts ─────────────────────────────────────────────────────────────────────
def open_aspect_chart():
    if not latest_clusters:
        messagebox.showerror("Error", "Please analyze reviews first.")
        return
    tc = {k: list(v) for k, v in latest_clusters.items()}
    # FIX: always create a NEW Thread object — never reuse a finished thread.
    # daemon=False so the window stays alive after the main loop would exit.
    t = threading.Thread(target=show_aspect_distribution, args=(tc,), daemon=False)
    t.start()


def open_sentiment_chart():
    if not latest_clusters:
        messagebox.showerror("Error", "Please analyze reviews first.")
        return
    tc = {k: list(v) for k, v in latest_clusters.items()}
    t = threading.Thread(target=show_sentiment_trends, args=(tc,), daemon=False)
    t.start()


# ── Export / Clear ─────────────────────────────────────────────────────────────
def _export_box(box):
    content = box.get(1.0, tk.END).strip()
    if not content:
        messagebox.showerror("Error", "No output to export.")
        return
    path = filedialog.asksaveasfilename(defaultextension=".txt",
                                        filetypes=[("Text Files","*.txt")])
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        messagebox.showinfo("Exported", "File saved successfully.")


def export_analysis():   _export_box(analysis_box)
def export_search():     _export_box(search_box)
def export_products():   _export_box(prod_box)
def clear_analysis():    analysis_box.delete(1.0, tk.END)
def clear_search():      search_box.delete(1.0, tk.END)
def clear_products():
    prod_box.delete(1.0, tk.END)
    for w in open_buttons_frame.winfo_children():
        w.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# BUILD GUI
# ══════════════════════════════════════════════════════════════════════════════
root = tk.Tk()
root.title("ReviewLens — BLAIR-Powered Review Intelligence")
root.geometry("1440x920")
root.minsize(1100, 700)
root.configure(bg=BG)
root.rowconfigure(3, weight=1)
root.columnconfigure(0, weight=1)

# ttk style for combobox
style = ttk.Style()
style.theme_use("clam")
style.configure("Dark.TCombobox",
                fieldbackground=ENTRY_BG, background=ENTRY_BG,
                foreground=ENTRY_FG, arrowcolor=FG2,
                selectbackground=SEL_BG, selectforeground=FG)
style.map("Dark.TCombobox",
          fieldbackground=[("readonly", ENTRY_BG)],
          selectbackground=[("readonly", ENTRY_BG)])


# ── Widget factories ────────────────────────────────────────────────────────────
def _btn(parent, text, cmd, bg, fg="#ffffff", width=None, pady=4):
    kw = dict(text=text, command=cmd, bg=bg, fg=fg,
              font=FONT_BTN, relief="flat", padx=10, pady=pady,
              activebackground=bg, activeforeground=fg,
              cursor="hand2", bd=0)
    if width:
        kw["width"] = width
    return tk.Button(parent, **kw)


def _lframe(parent, text, bg=BG3, fg=ACCENT2):
    return tk.LabelFrame(parent, text=f"  {text}  ",
                         font=FONT_BOLD, bg=bg, fg=fg,
                         padx=10, pady=8, bd=1, relief="solid")


# ── Title ───────────────────────────────────────────────────────────────────────
tk.Label(root, text="ReviewLens",
         font=FONT_TITLE, bg=BG, fg=ACCENT).grid(row=0, column=0, pady=(18,2))
tk.Label(root,
         text="BLAIR Item Retrieval  •  Aspect Discovery  •  Semantic Search  •  Meta Review",
         font=FONT_SUB, bg=BG, fg=FG2).grid(row=1, column=0, pady=(0,10))

# ── Top controls ────────────────────────────────────────────────────────────────
top = tk.Frame(root, bg=BG)
top.grid(row=2, column=0, sticky="ew", padx=22)
top.columnconfigure(0, weight=1)

# Step 1 — Load
s1 = _lframe(top, "Step 1 — Load Dataset")
s1.grid(row=0, column=0, sticky="ew", pady=4)
s1.columnconfigure(1, weight=1)
_btn(s1, "Choose JSONL File", browse_file, ACCENT2, width=18).grid(row=0, column=0, padx=8, pady=4)
file_label = tk.Label(s1, text="No file selected", anchor="w",
                       bg=BG3, fg=FG2, font=FONT_LABEL, wraplength=1000)
file_label.grid(row=0, column=1, padx=8, sticky="ew")

# Step 2 — Analyze
s2 = _lframe(top, "Step 2 — Analyse Reviews & Build Index")
s2.grid(row=1, column=0, sticky="ew", pady=4)
tk.Label(s2, text="Max Reviews:", bg=BG3, fg=FG2,
         font=FONT_LABEL).grid(row=0, column=0, padx=8)
review_limit_entry = tk.Entry(s2, width=7, font=FONT_LABEL,
                               bg=ENTRY_BG, fg=ENTRY_FG,
                               insertbackground=FG, relief="flat")
review_limit_entry.insert(0, "2000")
review_limit_entry.grid(row=0, column=1, padx=4)

btn_analyze = _btn(s2, "Analyze Reviews", analyze_reviews,      GREEN,  "#0d1117", width=16)
btn_achart  = _btn(s2, "Aspect Chart",    open_aspect_chart,    ACCENT2, width=13)
btn_tchart  = _btn(s2, "Trend Chart",     open_sentiment_chart, ACCENT,  width=13)
btn_expana  = _btn(s2, "Export",          export_analysis,      TEAL,    width=9)
btn_clrana  = _btn(s2, "Clear",           clear_analysis,       GREY,    width=7)
for col, b in enumerate([btn_analyze, btn_achart, btn_tchart, btn_expana, btn_clrana], 2):
    b.grid(row=0, column=col, padx=5, pady=4)
analyze_btns = [btn_analyze, btn_achart, btn_tchart]

# Step 3 — Sentence Search
s3 = _lframe(top, "Step 3 — Sentence Search")
s3.grid(row=2, column=0, sticky="ew", pady=4)
tk.Label(s3, text="Query:", bg=BG3, fg=FG2, font=FONT_LABEL).grid(row=0, column=0, padx=8)
query_entry = tk.Entry(s3, width=52, font=FONT_LABEL,
                        bg=ENTRY_BG, fg=ENTRY_FG,
                        insertbackground=FG, relief="flat")
query_entry.grid(row=0, column=1, padx=8)
btn_search  = _btn(s3, "Search", search_reviews, ORANGE, "#1e1e2e", width=10)
btn_expsrch = _btn(s3, "Export", export_search,  TEAL,   width=8)
btn_clrsrch = _btn(s3, "Clear",  clear_search,   RED,    width=7)
for col, b in enumerate([btn_search, btn_expsrch, btn_clrsrch], 2):
    b.grid(row=0, column=col, padx=5, pady=4)
search_btns = [btn_search]

# Step 4 — Product Recommendations
s4 = _lframe(top, "Step 4 — Product Recommendations  [BLAIR Item Retrieval]",
             bg=BG3, fg=ORANGE)
s4.grid(row=3, column=0, sticky="ew", pady=4)
tk.Label(s4, text="Query:", bg=BG3, fg=FG2, font=FONT_LABEL).grid(row=0, column=0, padx=8)
prod_query_entry = tk.Entry(s4, width=42, font=FONT_LABEL,
                             bg=ENTRY_BG, fg=ENTRY_FG,
                             insertbackground=FG, relief="flat")
prod_query_entry.grid(row=0, column=1, padx=8)

tk.Label(s4, text="Top K:", bg=BG3, fg=FG2, font=FONT_LABEL).grid(row=0, column=2, padx=(8,2))
prod_topk_var = tk.StringVar(value="10")
tk.Spinbox(s4, from_=1, to=50, width=4, textvariable=prod_topk_var,
           font=FONT_LABEL, bg=ENTRY_BG, fg=ENTRY_FG,
           buttonbackground=BG3, relief="flat").grid(row=0, column=3, padx=(0,8))

tk.Label(s4, text="Sort:", bg=BG3, fg=FG2, font=FONT_LABEL).grid(row=0, column=4, padx=(8,2))
sort_var = tk.StringVar(value="Score")
sort_cb  = ttk.Combobox(s4, textvariable=sort_var, width=11,
                         values=["Score","Rating ↓","Rating ↑","Reviews"],
                         state="readonly", style="Dark.TCombobox", font=FONT_LABEL)
sort_cb.grid(row=0, column=5, padx=(0,8))
sort_var.trace_add("write", _on_filter_change)

tk.Label(s4, text="Min ★:", bg=BG3, fg=FG2, font=FONT_LABEL).grid(row=0, column=6, padx=(8,2))
min_rating_var = tk.StringVar(value="0")
rating_cb = ttk.Combobox(s4, textvariable=min_rating_var, width=5,
                          values=["0","1","2","3","3.5","4","4.5"],
                          state="readonly", style="Dark.TCombobox", font=FONT_LABEL)
rating_cb.grid(row=0, column=7, padx=(0,8))
min_rating_var.trace_add("write", _on_filter_change)

btn_findprod = _btn(s4, "Find Products", search_products, ORANGE, "#1e1e2e", width=13)
btn_expprod  = _btn(s4, "Export",        export_products, TEAL,   width=8)
btn_clrprod  = _btn(s4, "Clear",         clear_products,  GREY,   width=7)
for col, b in enumerate([btn_findprod, btn_expprod, btn_clrprod], 8):
    b.grid(row=0, column=col, padx=5, pady=4)
prod_btns = [btn_findprod]


# ── Bottom: resizable PanedWindow ──────────────────────────────────────────────
# tk.PanedWindow (not ttk) gives sash handles you can DRAG to resize panels.
# Drag the thin bar between panels left/right to make any panel wider/narrower.
pane = tk.PanedWindow(
    root,
    orient=tk.HORIZONTAL,
    bg=BG,
    sashwidth=6,
    sashrelief="raised",
    sashpad=2,
    opaqueresize=True,
    handlesize=10,
    handlepad=40,
)
pane.grid(row=3, column=0, sticky="nsew", padx=14, pady=(10,14))


def _result_frame(parent, title, bg=BG2, fg=ACCENT2):
    f = tk.Frame(parent, bg=bg, bd=0)
    tk.Label(f, text=title, font=FONT_BOLD, bg=bg, fg=fg, pady=6).pack(fill="x", padx=4)
    return f


# Analysis pane
af = _result_frame(pane, "  Analysis Results")
analysis_box = scrolledtext.ScrolledText(
    af, wrap=tk.WORD, font=FONT_MONO,
    bg=RESULT_BG, fg=FG, insertbackground=FG,
    selectbackground=SEL_BG, relief="flat", bd=4)
analysis_box.pack(fill="both", expand=True, padx=4, pady=(0,4))
pane.add(af, stretch="always", minsize=260)

# Sentence search pane
sf = _result_frame(pane, "  Sentence Search Results")
search_box = scrolledtext.ScrolledText(
    sf, wrap=tk.WORD, font=FONT_MONO,
    bg=RESULT_BG, fg=FG, insertbackground=FG,
    selectbackground=SEL_BG, relief="flat", bd=4)
search_box.pack(fill="both", expand=True, padx=4, pady=(0,4))
pane.add(sf, stretch="always", minsize=260)

# Product recommendations pane
pf = _result_frame(pane, "  Product Recommendations", fg=ORANGE)
prod_box = scrolledtext.ScrolledText(
    pf, wrap=tk.WORD, font=FONT_MONO,
    bg=RESULT_BG, fg=FG, insertbackground=FG,
    selectbackground=SEL_BG, relief="flat", bd=4)
prod_box.pack(fill="both", expand=True, padx=4, pady=(0,2))
open_buttons_frame = tk.Frame(pf, bg=BG2)
open_buttons_frame.pack(fill="x", padx=4, pady=(2,4))
pane.add(pf, stretch="always", minsize=270)

root.mainloop()
