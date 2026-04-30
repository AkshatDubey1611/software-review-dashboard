# ReviewLens — BLAIR Upgrade Change Log

## What changed and why

---

### NEW FILE: `item_retrieval.py`

The biggest addition. Implements full BLAIR-style item-level retrieval
as described in the paper (§2.3 and §3.3).

**What it does:**
1. Groups all loaded reviews by product (`parent_asin` → `asin` fallback).
2. Builds a rich **item profile** per product:
   - Product title
   - Top 5 representative review sentences (scored by keyword frequency)
   - Top 10 aspect keywords
   - This mirrors BLAIR's "(review, item_metadata)" training pairs.
3. Encodes every item profile with the shared BLAIR encoder.
4. At query time, encodes the query and ranks products by dot-product
   similarity (= cosine similarity since embeddings are L2-normalised).
5. Returns results with: title, ASIN, **Amazon URL**, avg rating,
   review count, match explanation, supporting review snippet.

**Why:** Before this, the system only returned review sentences. Now it
returns ranked *products*, which is the core BLAIR use case.

---

### MODIFIED: `review_processing.py`

`load_reviews()` now extracts three additional fields per review:
- `parent_asin` — canonical product grouping key (Amazon's hierarchy)
- `item_id`     — `parent_asin` if available, else `asin`
- `title`       — product title (distinct from review summary/headline)

Tries multiple field names (`title`, `product_title`, `name`) to be
robust across JSONL schema variants.

**Why:** `item_retrieval.ItemIndex` needs these to group reviews by
product and build item profiles.

---

### MODIFIED: `sentence_retrieval.py`

`retrieve_relevant_sentences()` results now include three new fields:
- `amazon_url` — direct `https://www.amazon.com/dp/<asin>` link
- `item_id`    — canonical ASIN for the product
- `title`      — product title

`_build_source_line()` replaced by `_build_source_parts()` which returns
a tuple `(formatted_string, url, item_id, title)` so the GUI can render
clickable links separately.

**Why:** Users should be able to click through to the actual product on
Amazon directly from the sentence search results.

---

### MODIFIED: `app.py`

Major GUI upgrade:

1. **Step 4 section** — new "Product Recommendations [BLAIR Item Retrieval]"
   control strip with:
   - Product query entry box
   - Top-K spinner (1–20)
   - "Find Products" button (Amazon orange)
   - Export / Clear buttons

2. **Third results panel** — "Product Recommendations (BLAIR)" panel
   alongside the existing Analysis and Sentence Search panels.
   Shows per-product: title, BLAIR score, star rating, review count,
   ASIN, Amazon URL, match explanation, sample review snippet.

3. **Dynamic "Open on Amazon" buttons** — up to 5 clickable buttons
   appear below the product results panel, one per top result, opening
   the product's Amazon page in the browser via `webbrowser.open()`.

4. **Sentence search results** now display the product title and Amazon
   URL inline (not just the raw source metadata string).

5. **Analyze step** also calls `reset_item_index()` + `index.build()`
   so the item index is always in sync with the loaded reviews.

---

### UNCHANGED FILES (included for completeness)
- `blair_encoder.py`  — no changes needed
- `chart_utils.py`    — no changes needed
- `config.py`         — no changes needed
- `dynamic_aspects.py`— no changes needed
- `query_understanding.py` — no changes needed
- `retrieval.py`      — no changes needed
- `summary_formatter.py`   — no changes needed
- `utils.py`          — no changes needed
- `code_extractor.py` — no changes needed

---

## How to describe this to your professor

> "Initially we performed review-level aspect analysis using BLAIR embeddings
> for clustering and semantic retrieval of review sentences. To align more
> closely with the BLAIR paper's core contribution, we extended the system
> toward item-level retrieval: we now construct product representations by
> aggregating review text and metadata per ASIN (mirroring BLAIR's training
> pairs of user reviews and item metadata), encode them with the BLAIR
> checkpoint, and rank products against natural-language queries using cosine
> similarity in the BLAIR embedding space. Results include direct Amazon links,
> match explanations, and average ratings — transforming ReviewLens from a
> review analysis tool into a review-based product recommendation system."
