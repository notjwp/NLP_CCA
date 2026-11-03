# Presentation demo: script and checklist

The demo is a small website that runs on this laptop. One scientific claim goes to three search engines at
once, and the page shows which one finds the correct paper:

| Column | What it is |
|---|---|
| **Word matching · TF-IDF + cosine** | The classical system, built from scratch: TF-IDF weights + cosine similarity |
| **Meaning matching · semantic search** | bge-small, a transformer that turns each text into a "meaning vector" |
| **LLM reranker · Qwen3** | A large language model reads the claim with each of TF-IDF's top 100 papers and answers yes/no |

**What's on the screen, top to bottom (tab 1):**
1. **The claim box and the example buttons** ① to ④, plus 🎲 for a random test claim.
2. **A yellow story card:** one or two sentences on what this example shows. Read it out, or let the judges read it.
3. **One big tile per engine:** where it put the correct paper (green **#1**, orange **#32**) and how long it took.
4. **Two "why?" buttons** that jump to tab 2 (inside TF-IDF) and tab 3 (inside the LLM). Both tabs have a
   "← Back" button.
5. **The three top-5 lists:** the correct paper is shaded green with ✓, and words shared with the claim are
   yellow. Hover over a title to see the start of the paper.

```sh
uv run irsys demo
```

It loads in about 30–40 s, then opens your default browser at **http://127.0.0.1:7860**. To present
from Brave, paste that address into Brave. Stop the demo with `Ctrl+C` in the terminal. Without a GPU, use
`uv run irsys demo --no-llm` (two columns, no tab 3).

**How the page behaves:** the first two tiles and lists appear instantly. The LLM tile shows a spinner
(*"reading the 100 candidate papers …"*) and fills in about 5 seconds later. That wait is part of the story: the LLM is
the most accurate system and by far the slowest.

---

## Pre-flight checklist

**The day before**
- [ ] Run `uv run irsys demo` and click all four examples. Everything is downloaded already and the demo
      runs **fully offline**: no Wi-Fi needed.
- [ ] Copy `demo_screenshots/` into your slides as a backup (see "If something goes wrong").

**10 minutes before**
- [ ] Close other GPU-heavy apps; they take the GPU memory the LLM needs.
- [ ] Plug in the charger. On battery, the GPU may throttle.
- [ ] Run `uv run irsys demo`. Wait until example ① shows all three tiles.
- [ ] Zoom the browser to **125–150 %** (`Ctrl` + `+`) so the back row can read it. The three columns stay
      side by side; if the screen is narrow they wrap onto a second row, which also works.
- [ ] Leave it on **tab 1** with example ① loaded.

---

## Two presenters, one talk

The talk follows the 10 slides in `NLP_CCA_Presentation.pptx` (a local file, not in the repository):

| Part | Who | What |
|---|---|---|
| Slides 1–5 | Presenter 1 | What information retrieval is, the task and data, the three systems and how they are judged, the classical TF-IDF system; then the handover |
| Slide 6 | Presenter 2 | How the AI systems work: semantic search and the LLM's yes/no score |
| Slide 7 | Presenter 2 | **This demo**, steps 0–3 below (about 3 minutes) |
| Slides 8–10 | Presenter 2 | Results (all six metrics), error analysis, conclusions |

The slides already introduce the task and present the results, so inside the talk the demo runs steps 0–3 and
skips steps 4–5. Use steps 4–5 only when demoing without the slides.

## Script

### 0 · Set the scene (15 s; 30 s without slides)
> "Let's see the three systems on real claims. I type a claim, and three search engines look for the paper
> that supports or refutes it. On the left, our **classical** system: TF-IDF with cosine similarity, it
> matches **words**, and rare words count more. In the middle, **semantic search**: an AI model that matches
> **meaning**. On the right, a **large language model** that reads each candidate paper and answers yes or no.
> For these example claims we know the correct paper, so the big tiles show where each engine put it."

### 1 · Example ① "Bcl2 vs BCL-2 (spelling)" (already loaded) (2 min)
**Tab 1 · Compare**
- Point at the tiles: word matching **#32** (orange), meaning matching **#1**. Then the LLM tile fills in:
  **#1**. The yellow story card already says why.
> "The claim writes 'Bcl2'. The correct paper writes 'BCL-2', with a hyphen. To word matching those are
> different words. The AI models know they are the same gene."

**Tab 2 · Inside TF-IDF** (click **Why did word matching rank them like this? →**)
- **1a–1e:** the claim is split into words; each word gets its **part of speech** (noun, verb, adjective);
  the parser groups **noun phrases**; filler words like *of, is, the* are crossed out; words are cut to
  their roots (*silencing* → `silenc`, *tumors* → `tumor`).
- **Step 2:** `bcl2` appears in only **7** of 5,183 papers, so it gets the **highest rarity score (idf 7.6)**:
  the longest bar.
- **Step 3:** read the blue box: *the claim's most important word, "bcl2", is in TF-IDF's #1 paper but not in
  the correct paper, so the correct paper ends up at position 32.* In the correct paper's table, `bcl2` shows
  **✗ missing**, because there it is written "BCL-2", which became `bcl` + `2`.
> "This is the vocabulary mismatch problem: exact matching fails when the same thing is written differently."

**Tab 3 · Inside the LLM** (click **← Back**, then **How did the LLM decide? →**)
- The three boxes at the top are the method: TF-IDF picks 100 candidates → the LLM reads the claim with one
  paper and may only say yes or no → the papers are sorted by P("yes").
- Read the purple box: the LLM says **97.8%** "yes" to the correct paper but only **56.2%** to TF-IDF's #1
  (also about Bcl2, but it doesn't address the claim), so the correct paper moves from position 32 to 1.
  The bars show the same.
- Click **Show the exact prompt…**: "This is exactly what the model reads: an instruction, the claim, one
  paper, and it may only say yes or no. We read the probability of 'yes'. It does that 100 times per search."

### 2 · Example ③ "Rare disease name" (1 min)
**Tab 1**
- Tiles: word matching **#1**, meaning matching **#59** (orange), LLM **#1**.

**Tab 2**
- **1e:** *ivermectin* is faded: it appears in **no** paper at all.
- **Step 2:** `onchocerciasi` appears in exactly **1** paper out of 5,183, so its **idf is 9.55**, the maximum.
- **Step 3:** the blue box says TF-IDF's #1 result is the correct paper ✓ and that it contains
  `onchocerciasi`, in only 1 of 5,183 papers: that one rare word is the whole score.
> "Here word matching wins. A rare technical term is a precise fingerprint. The embedding model only has a
> vague idea of this disease and drifts to general drug-treatment papers. The LLM reads the actual paper,
> so it gets it right too."

### 3 · Live claim (1 min)
Type in the box and press Enter:

**`heart attack risk in people with diabetes`**

- The tiles show "—": a typed claim has no answer key, so point at the lists instead.
- Meaning matching's #1 is *"Impact of diabetes on long-term prognosis in patients with unstable angina and
  non-Q-wave **myocardial infarction**…"*: that is the medical term for a heart attack, although the words
  "heart attack" never appear.
- Word matching's top hit is a questionnaire paper that repeats *heart, risk, people, diabetes*.
> "Nobody wrote a synonym list: the model learned that 'heart attack' means 'myocardial infarction'."

**Audience suggestion?** Ask for a short **biomedical** claim; all the papers are medicine/biology. These
also work well: `cancer cells hide from the immune system` (tab 2 shows *hide* is in no paper),
`children who are overweight` (meaning matching also finds "obesity in children"). Typed claims get no ✓:
only the 300 benchmark claims have known answers (the 🎲 button picks one).

### 4 · Results on all 300 test claims (1.5 min; skip when the slides follow)
**Tab 4** opens with one big tile per system: nDCG@10, time per claim, and whether it is really better than
TF-IDF. The plain-language guide is under *What do these numbers mean?*, then two tables: the six metrics
(nDCG@k, MAP, MRR, Precision@k, Recall@k, F1@k at k = 5 and 10), then speed and significance. Point at nDCG@10, the main grade: 1 = the
correct paper is always first, 0.5 ≈ usually third, 0 = not in the top 10.

| | nDCG@10 | MAP | MRR | Time per claim |
|---|---|---|---|---|
| Word matching · TF-IDF (from scratch) | 0.659 | 0.609 | 0.623 | **0.17 ms** on the CPU |
| Meaning matching · semantic search (bge-small) | 0.713 | 0.674 | 0.687 | 12 ms on the GPU |
| **LLM reranker (Qwen3)** | **0.772** | **0.734** | **0.743** | 5.8 s on the GPU |

> "The LLM reranker is the best on all six metrics: 17% better than our TF-IDF on nDCG@10, and
> significantly better than semantic search too. Semantic search is 8% better than TF-IDF. But TF-IDF is
> about 34,000 times faster than the LLM, needs no GPU, and every score can be explained word by word, as
> tab 2 showed."

### 5 · Close (30 s; skip when the slides follow)
> "Classical and AI retrieval are complementary: exact words versus meaning. The best results come from
> using them together, a fast classical first stage and an AI model on top, which is how modern search
> engines are built."

The other two examples, if there's time:
- **② Same idea, other words:** word matching position 11 → meaning matching 1 → LLM 1. The claim and the
  paper say the same thing in different words.
- **④ LDL vs HDL:** word matching 1 → meaning matching 10 → LLM 1. The embedding blurs the one-letter
  difference between LDL and HDL cholesterol; exact matching and the LLM keep it.

---

## Likely questions

| Question | Answer |
|---|---|
| What is POS tagging / parsing here? | Tab 2, rows 1b and 1c: NLTK's tagger labels each word noun/verb/adjective…, and a chunk parser with the grammar `NP: {<JJ.*\|NN.*>*<NN.*>}` groups noun phrases. We tested using them for lemmatization and phrase terms; on the 809 training claims plain stemming worked best (0.679 vs 0.664 / 0.660), so the final system uses stemming. |
| Why does stemming beat lemmatization? | Stemming merges more word forms (*discontinue / discontinuation*, *activate / activation*). On 228 of 809 training claims it matches more of the claim's words in the correct paper; lemmatization does so on only 27. |
| What does "lsc.lsc" mean? | SMART notation: **l**og tf (1 + ln tf), **s**klearn-style idf (ln(N/df) + 1), **c**osine normalization, for both papers and claim. Chosen on the *training* claims, not the test set. |
| Is TF-IDF really from scratch? | Yes: own tokenizer, inverted index, weighting and cosine. A test checks it matches scikit-learn's TfidfVectorizer to 9 decimal places. |
| How does the LLM give a score if it only says yes or no? | We never let it write. One forward pass gives the probability of every possible next word; we take P("yes") vs P("no"). Tab 3 shows the exact prompt. |
| How often does the LLM say "yes"? | To only 2.3% of the 29,889 candidate papers it judged, but to 75% of the correct ones: it says yes rarely, and mostly to the right papers. |
| Why does the LLM still miss some? | It fails on 31 claims; on 20 of them TF-IDF never put the correct paper in its top 100, so the LLM never saw it. A better first stage would help: semantic search has 13 of those 20 papers in its own top 100. |
| Why not a bigger LLM? | The laptop GPU has 4 GB. A 0.6B model already takes ~5 s per claim to read 100 papers; a 7B model would not fit, and an API model would cost money per query. |
| Which metrics did you use? | Six: nDCG@k, MAP, MRR, Precision@k, Recall@k and F1@k, at k = 5 and 10. nDCG@10 is the main one (the BEIR standard); the others rank the systems in almost the same order. Precision and F1 are low for everyone because most claims have only one correct paper (P@10 can be at most 0.1). |
| Is the improvement significant? | Yes. A statistical test on the same 300 claims (Wilcoxon) gives p ≈ 2×10⁻¹¹ for the LLM reranker and p ≈ 0.003 for semantic search, both far below 0.05, so neither gain is luck. (Details, including the Holm correction for running two tests, are in the report.) |
| Does it need a GPU? | Only the AI side. TF-IDF runs in a fraction of a millisecond on the CPU. `irsys demo --no-llm` runs without the LLM. |

---

## If something goes wrong
- **The app doesn't open:** check the terminal for the address and paste http://127.0.0.1:7860 into the
  browser.
- **"Address already in use":** an old demo is still running. Close that terminal, or run
  `uv run python -c "from irsys.demo import launch_demo; launch_demo(server_port=7870)"`.
- **"CUDA out of memory":** something else is using the GPU (close other heavy apps), or start with
  `uv run irsys demo --no-llm`.
- **The laptop fails entirely:** use the backup screenshots in `demo_screenshots/`, in script order:

| File | Shows |
|---|---|
| `01_bcl2_compare.png`, `02_bcl2_inside_tfidf.png`, `03_bcl2_inside_llm.png` | Step 1 |
| `04_rare_term_compare.png`, `05_rare_term_inside_tfidf.png` | Step 2 |
| `06_live_heart_attack_compare.png` | Step 3 |
| `07_results.png` | Step 4 |
| `08_paraphrase_compare.png`, `09_ldl_hdl_compare.png` | Extra examples |
