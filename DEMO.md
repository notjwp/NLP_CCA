# Presentation demo: script and checklist

The demo is an interactive app that runs locally in your browser. It sends the same query to three systems:
**TF-IDF cosine**, **semantic search (bge-small)** and a **hybrid** of the two. It also shows, step by
step, how TF-IDF computed its score.

```sh
uv run irsys demo
```

It loads in about 20–30 s, then opens your default browser at **http://127.0.0.1:7860**. If you'd rather
present from Brave, paste that address into Brave. Stop the demo with `Ctrl+C` in the terminal.

---

## Pre-flight checklist

**The day before**
- [ ] Run `uv run irsys demo` once and click through every preset. All models are already downloaded on
      this laptop, and the demo runs **fully offline**: no Wi-Fi needed.
- [ ] Copy `demo_screenshots/` into your slides as a backup (see "If something goes wrong").

**10 minutes before**
- [ ] Plug in the charger. On battery, the GPU may throttle.
- [ ] Close games and other GPU-heavy apps.
- [ ] Run `uv run irsys demo`. Wait for the page to show results for the Bcl2 example.
- [ ] Zoom the browser to **125–150 %** (`Ctrl` + `+`) so the back row can read it.
- [ ] Leave it on **Tab 1** with the Bcl2 example loaded.

---

## Script (about 7 minutes)

### 0 · Set the scene (30 s)
> "This is a search engine over 5,183 real scientific abstracts from the SciFact benchmark. One query goes
> to three systems at once. On the left, the **classical** system I built from scratch: TF-IDF weighting
> plus cosine similarity. In the middle, a **semantic** system using a transformer language model that turns
> text into meaning vectors. On the right, a **hybrid** that merges both rankings. Yellow highlights are
> words that literally match the query."

### 1 · Preset **"Bcl2 vs “BCL-2” · semantic wins"** (already loaded) (1.5 min)
**Tab 1**
- Semantic: **✓ correct abstract at rank 1**. TF-IDF: **▼ only at rank 32**.
> "The claim says 'Bcl2'. The correct paper writes 'BCL-2', with a hyphen. To TF-IDF those are different
> words, so it can't connect them. The language model knows they're the same gene."

**Tab 2 · How TF-IDF scored it**
- Step 1: walk through tokens → stopwords crossed out → stems ("silencing" → `silenc`).
- Step 2: `bcl2` appears in only **7** abstracts, so it gets the **highest idf (7.6)** and dominates the query.
- Step 3: TF-IDF's #1 result wins on `bcl2` **alone**. The correct abstract shows "not in abstract" for
  `bcl2`, because "BCL-2" was split into `bcl` + `2`.
> "This is the vocabulary mismatch problem. Exact matching fails when the same idea is spelled differently."

### 2 · Preset **"Rare term · TF-IDF wins"** (1 min)
**Tab 1**
- TF-IDF: **✓ rank 1**. Semantic: **▼ only at rank 59**.

**Tab 2**
- Step 1: "ivermectin" is **crossed out**: it appears in no abstract at all.
- Step 2: `onchocerciasi` appears in exactly **1** abstract out of 5,183, so its **idf is 9.55**, the
  maximum possible. Its query weight is 0.92 (the maximum is 1).
- Step 3: that single rare term is the whole score.
> "Here exact matching wins. A rare technical term is a precise fingerprint. The language model only has a
> vague idea of this disease name, so it drifts to generic drug-therapy papers."

### 3 · Live query (1 min)
Type in the box and press Enter:

**`heart attack risk in people with diabetes`**

- Semantic #1: *"…unstable angina and non-Q-wave **myocardial infarction**…"*. That's the medical term for a
  heart attack, even though the words "heart attack" never appear.
- TF-IDF's top hit is a questionnaire paper that repeats the words heart, risk, people and diabetes. Its
  next hits are about Alzheimer's risk factors and cholesterol.
> "No hand-written synonyms anywhere: the model learned that 'heart attack' means 'myocardial infarction'."

**Audience suggestion?** Ask for a short **biomedical** claim; the corpus is all medical/biology abstracts.
These also work well:
- `cancer cells hide from the immune system`. Tab 2 shows "hide" is unknown to TF-IDF.
- `children who are overweight`. Semantic also finds "obesity in children".

If the audience suggests a question, there's no ✓ marking. Only the 300 benchmark claims have known answers
(the 🎲 **Random test claim** button picks one of them).

### 4 · Hybrid + the numbers (1.5 min)
- Point at the **hybrid** column: in the Bcl2 example it lifts the correct paper from rank 32 to **rank 7**;
  for the rare term it keeps it at **rank 3** even though semantic search alone ranked it 59th.
- **Tab 3 · Evaluation results** shows the averages over all 300 test claims:

| | nDCG@10 | Latency per query |
|---|---|---|
| TF-IDF (from scratch) | 0.659 | **0.26 ms** on CPU |
| Semantic (bge-small) | 0.713, significantly better | 19 ms on GPU |
| **Hybrid** | **0.722, best** | 21 ms |

> "Semantic search is about 8 % better on average, but TF-IDF is about 70× faster, needs no GPU and can
> explain every score, as Tab 2 just did. They fail on *different* queries, so combining them gives the
> best result."

### 5 · Close (30 s)
> "Classical and semantic retrieval are complementary: exact terms vs. meaning. That's why many modern
> search engines use hybrids."

The other two presets, if there's time:
- **Paraphrase · semantic wins:** TF-IDF rank 11 → semantic rank 1. The claim and the abstract express the
  same idea in different words.
- **Exact terms · TF-IDF wins:** TF-IDF rank 1 → semantic rank 10. "LDL" vs "HDL" cholesterol: the
  embedding blurs a one-letter difference that exact matching keeps.

---

## Likely questions

| Question | Answer |
|---|---|
| Why are the hybrid's scores so small (≈0.03)? | Hybrid uses Reciprocal Rank Fusion, score = Σ 1/(60 + rank). It combines **ranks**, not scores, so cosine values and model scores never need to be on the same scale. |
| What does "lsc.lsc" mean? | SMART notation: **l**og tf (1 + ln tf), **s**klearn-style idf (ln(N/df) + 1), **c**osine normalization, for both documents and query. It was chosen on the *training* queries, not the test set. |
| Is TF-IDF really from scratch? | Yes: own tokenizer, inverted index, weighting and cosine. A test checks it matches scikit-learn's TfidfVectorizer to 9 decimal places. |
| Is the semantic model an LLM? | It's a small transformer language model (33M parameters, BERT family) trained for retrieval. It produces embeddings instead of generating text. |
| Why is the improvement "significant"? | Wilcoxon signed-rank test over the same 300 queries, Holm-corrected: p = 0.008 for bge, p ≈ 10⁻⁷ for hybrid. |
| Does it need a GPU? | Only the semantic side benefits. On CPU it still works, just slower. TF-IDF runs in a fraction of a millisecond on CPU. |
| Why not use the cross-encoder in the demo? | It's in the report (0.690). At ~0.5 s per query it's the slowest system, and the three columns tell the story more clearly. |

---

## If something goes wrong
- **The app doesn't open:** check the terminal for the address and paste http://127.0.0.1:7860 into the
  browser.
- **"Address already in use":** an old demo is still running. Close that terminal, or run
  `uv run python -c "from irsys.demo import launch_demo; launch_demo(server_port=7870)"`.
- **The laptop fails entirely:** use the backup screenshots in `demo_screenshots/`. They're in script order:

| File | Shows |
|---|---|
| `01_bcl2_compare.png`, `02_bcl2_explain.png` | Step 1 |
| `03_paraphrase_compare.png` | Paraphrase preset |
| `04_rare_term_compare.png`, `05_rare_term_explain.png` | Step 2 |
| `06_exact_terms_compare.png` | Exact-terms preset |
| `07_live_heart_attack_compare.png` | Step 3 |
| `08_evaluation_results.png` | Step 4 |
