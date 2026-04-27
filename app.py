# app.py

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from review_processing import load_reviews, process_reviews
from summary_formatter import format_all_summaries
from sentence_retrieval import retrieve_relevant_sentences
from chart_utils import show_aspect_distribution, show_sentiment_trends

selected_file = ""
loaded_reviews = []      # list of dicts: {text, asin, reviewer, date, rating, summary}
latest_clusters = {}     # {aspect -> [review_dict, ...]}


# -----------------------------
# FILE BROWSER
# -----------------------------
def browse_file():
    global selected_file
    file_path = filedialog.askopenfilename(
        title="Select JSONL File",
        filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
    )
    if file_path:
        selected_file = file_path
        file_label.config(text=file_path)


# -----------------------------
# ANALYZE REVIEWS
# -----------------------------
def analyze_reviews():
    global selected_file, loaded_reviews, latest_clusters

    if not selected_file:
        messagebox.showerror("Error", "Please select a JSONL file first.")
        return

    try:
        max_reviews = int(review_limit_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid number for review limit.")
        return

    analysis_box.delete(1.0, tk.END)
    analysis_box.insert(tk.END, "Loading reviews...\n")
    analysis_box.see(tk.END)
    root.update()

    loaded_reviews = load_reviews(selected_file, max_reviews)

    analysis_box.insert(tk.END, f"Total reviews loaded: {len(loaded_reviews)}\n\n")
    analysis_box.see(tk.END)
    root.update()

    analysis_box.insert(tk.END, "Clustering reviews by aspect...\n")
    analysis_box.see(tk.END)
    root.update()

    latest_clusters, _ = process_reviews(loaded_reviews, num_clusters=8)

    analysis_box.insert(tk.END, "Generating structured summaries...\n\n")
    analysis_box.see(tk.END)
    root.update()

    summaries = format_all_summaries(latest_clusters)

    analysis_box.insert(tk.END, "=" * 60 + "\n")
    analysis_box.insert(tk.END, "FINAL META REVIEW\n")
    analysis_box.insert(tk.END, "=" * 60 + "\n\n")

    for aspect, summary in summaries.items():
        count = len(latest_clusters[aspect])
        # Use plain ASCII marker instead of emoji (avoids Tkinter font encoding issues)
        analysis_box.insert(tk.END, f"[*] {aspect.upper()} ({count} reviews)\n")
        analysis_box.insert(tk.END, "-" * 50 + "\n")
        analysis_box.insert(tk.END, summary + "\n\n")

    analysis_box.see(tk.END)


# -----------------------------
# SEARCH REVIEWS
# -----------------------------
def search_reviews():
    global loaded_reviews

    query = query_entry.get().strip()

    if not loaded_reviews:
        messagebox.showerror("Error", "Please analyze reviews first.")
        return

    if not query:
        messagebox.showerror("Error", "Please enter a search query.")
        return

    search_box.delete(1.0, tk.END)
    search_box.insert(tk.END, f"Search Query: {query}\n\n")
    search_box.see(tk.END)
    root.update()

    results = retrieve_relevant_sentences(query, loaded_reviews, top_k=5)

    search_box.insert(tk.END, "=" * 60 + "\n")
    search_box.insert(tk.END, "TOP MATCHING SENTENCES\n")
    search_box.insert(tk.END, "=" * 60 + "\n\n")

    for i, item in enumerate(results, start=1):
        search_box.insert(tk.END, f"[{i}] Score: {item['score']:.4f}\n")
        search_box.insert(tk.END, f"    {item['sentence']}\n")
        if item.get("source"):
            search_box.insert(tk.END, f"    Source: {item['source']}\n")
        search_box.insert(tk.END, "\n")

    search_box.see(tk.END)


# -----------------------------
# CHARTS
# -----------------------------
def open_aspect_chart():
    global latest_clusters
    if not latest_clusters:
        messagebox.showerror("Error", "Please analyze reviews first.")
        return
    # chart_utils expects {aspect: [texts]} - pass list of texts
    text_clusters = {k: [r["text"] if isinstance(r, dict) else r for r in v]
                     for k, v in latest_clusters.items()}
    show_aspect_distribution(text_clusters)


def open_sentiment_chart():
    global latest_clusters
    if not latest_clusters:
        messagebox.showerror("Error", "Please analyze reviews first.")
        return
    text_clusters = {k: [r["text"] if isinstance(r, dict) else r for r in v]
                     for k, v in latest_clusters.items()}
    show_sentiment_trends(text_clusters)


# -----------------------------
# EXPORT
# -----------------------------
def export_analysis():
    content = analysis_box.get(1.0, tk.END).strip()
    if not content:
        messagebox.showerror("Error", "No analysis output to export.")
        return
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text Files", "*.txt")]
    )
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        messagebox.showinfo("Success", "Analysis exported successfully.")


def export_search():
    content = search_box.get(1.0, tk.END).strip()
    if not content:
        messagebox.showerror("Error", "No search output to export.")
        return
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text Files", "*.txt")]
    )
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        messagebox.showinfo("Success", "Search results exported successfully.")


# -----------------------------
# CLEAR OUTPUTS
# -----------------------------
def clear_analysis():
    analysis_box.delete(1.0, tk.END)


def clear_search():
    search_box.delete(1.0, tk.END)


# -----------------------------
# BUILD GUI
# -----------------------------
root = tk.Tk()
root.title("Software Review Intelligence Dashboard")
root.geometry("1350x850")
root.configure(bg="#f4f6f8")

root.rowconfigure(3, weight=1)
root.columnconfigure(0, weight=1)

# -----------------------------
# TITLE
# -----------------------------
title_label = tk.Label(
    root,
    text="Software Review Intelligence Dashboard",
    font=("Arial", 22, "bold"),
    bg="#f4f6f8",
    fg="#1f2d3d"
)
title_label.grid(row=0, column=0, pady=(15, 5), sticky="n")

subtitle_label = tk.Label(
    root,
    text="Semantic Aspect Analysis + Query-Based Review Retrieval",
    font=("Arial", 11),
    bg="#f4f6f8",
    fg="#5b6b7a"
)
subtitle_label.grid(row=1, column=0, pady=(0, 15), sticky="n")

# -----------------------------
# TOP CONTROL CONTAINER
# -----------------------------
top_container = tk.Frame(root, bg="#f4f6f8")
top_container.grid(row=2, column=0, sticky="ew", padx=20)
top_container.columnconfigure(0, weight=1)

# FILE SECTION
section1 = tk.LabelFrame(
    top_container,
    text=" Step 1: Load Review Dataset ",
    font=("Arial", 11, "bold"),
    padx=10, pady=10,
    bg="#ffffff", fg="#1f2d3d"
)
section1.grid(row=0, column=0, sticky="ew", pady=6)
section1.columnconfigure(1, weight=1)

browse_button = tk.Button(
    section1, text="Choose JSONL File", command=browse_file,
    width=20, bg="#4f81bd", fg="white",
    font=("Arial", 10, "bold"), relief="flat"
)
browse_button.grid(row=0, column=0, padx=10, pady=5)

file_label = tk.Label(
    section1, text="No file selected",
    wraplength=900, anchor="w", justify="left",
    bg="#ffffff", fg="#333333", font=("Arial", 10)
)
file_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")

# ANALYSIS SECTION
section2 = tk.LabelFrame(
    top_container,
    text=" Step 2: Analyze Reviews ",
    font=("Arial", 11, "bold"),
    padx=10, pady=10,
    bg="#ffffff", fg="#1f2d3d"
)
section2.grid(row=1, column=0, sticky="ew", pady=6)

tk.Label(section2, text="Max Reviews to Load:", bg="#ffffff",
         font=("Arial", 10)).grid(row=0, column=0, padx=10, pady=5)

review_limit_entry = tk.Entry(section2, width=10, font=("Arial", 10))
review_limit_entry.insert(0, "2000")
review_limit_entry.grid(row=0, column=1, padx=10, pady=5)

for col, (label, cmd, color, w) in enumerate([
    ("Analyze Reviews", analyze_reviews,   "#5cb85c", 16),
    ("Aspect Chart",    open_aspect_chart, "#337ab7", 13),
    ("Trend Chart",     open_sentiment_chart, "#8e44ad", 13),
    ("Export Analysis", export_analysis,   "#16a085", 13),
    ("Clear Analysis",  clear_analysis,    "#999999", 13),
], start=2):
    tk.Button(section2, text=label, command=cmd, bg=color, fg="white",
              width=w, font=("Arial", 10, "bold"), relief="flat"
              ).grid(row=0, column=col, padx=8, pady=5)

# SEARCH SECTION
section3 = tk.LabelFrame(
    top_container,
    text=" Step 3: Search Reviews by User Query ",
    font=("Arial", 11, "bold"),
    padx=10, pady=10,
    bg="#ffffff", fg="#1f2d3d"
)
section3.grid(row=2, column=0, sticky="ew", pady=6)

tk.Label(section3, text="Search Query:", bg="#ffffff",
         font=("Arial", 10)).grid(row=0, column=0, padx=10, pady=5)

query_entry = tk.Entry(section3, width=50, font=("Arial", 10))
query_entry.grid(row=0, column=1, padx=10, pady=5)

for col, (label, cmd, color, w) in enumerate([
    ("Search Reviews", search_reviews, "#f0ad4e", 16),
    ("Export Search",  export_search,  "#16a085", 13),
    ("Clear Search",   clear_search,   "#d9534f", 13),
], start=2):
    tk.Button(section3, text=label, command=cmd, bg=color, fg="white",
              width=w, font=("Arial", 10, "bold"), relief="flat"
              ).grid(row=0, column=col, padx=8, pady=5)

# -----------------------------
# BOTTOM PANELS
# -----------------------------
bottom_container = tk.Frame(root, bg="#f4f6f8")
bottom_container.grid(row=3, column=0, sticky="nsew", padx=20, pady=(10, 15))
bottom_container.columnconfigure(0, weight=1)
bottom_container.columnconfigure(1, weight=1)
bottom_container.rowconfigure(0, weight=1)

analysis_frame = tk.LabelFrame(
    bottom_container, text=" Analysis Results ",
    font=("Arial", 11, "bold"), padx=10, pady=10,
    bg="#ffffff", fg="#1f2d3d"
)
analysis_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

analysis_box = scrolledtext.ScrolledText(
    analysis_frame, wrap=tk.WORD,
    font=("Courier", 10),        # Courier: safe monospace, no emoji issues
    bg="#fcfcfc", fg="#222222",
    insertbackground="black", relief="flat", borderwidth=1
)
analysis_box.pack(fill="both", expand=True)

search_frame = tk.LabelFrame(
    bottom_container, text=" Search Results ",
    font=("Arial", 11, "bold"), padx=10, pady=10,
    bg="#ffffff", fg="#1f2d3d"
)
search_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

search_box = scrolledtext.ScrolledText(
    search_frame, wrap=tk.WORD,
    font=("Courier", 10),
    bg="#fcfcfc", fg="#222222",
    insertbackground="black", relief="flat", borderwidth=1
)
search_box.pack(fill="both", expand=True)

root.mainloop()