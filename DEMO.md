# Presentation demo: script and checklist

The demo is a small website that runs on this laptop. One scientific claim goes to four search engines at
once, and the page shows which one finds the correct paper:

| Column | What it is |
|---|---|
| **Word matching · TF-IDF** | The classical system, built from scratch: TF-IDF weights + cosine similarity |
| **Meaning matching · AI embeddings** | bge-small, a transformer that turns each text into a "meaning vector" |
| **Combined · words + meaning** | The two lists above merged (Reciprocal Rank Fusion) |
| **LLM reranker · Qwen3** | A large language model reads the claim with each of TF-IDF's top 100 papers and answers yes/no |

```sh
uv run irsys demo
```

It loads in about 30–40 s, then opens your default browser at **http://127.0.0.1:7860**. To present
from Brave, paste that address into Brave. Stop the demo with `Ctrl+C` in the terminal. Without a GPU, use
`uv run irsys demo --no-llm` (three columns, no tab 3).

**How the page behaves:** the first three columns appear instantly. The LLM column says *"The LLM is reading
the 100 candidate papers…"* and fills in about 5 seconds later. That wait is part of the story: the LLM is
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
- [ ] Run `uv run irsys demo`. Wait until example ① shows all four columns.
- [ ] Zoom the browser to **125–150 %** (`Ctrl` + `+`) so the back row can read it. The four columns stay
      side by side; if the screen is narrow they wrap onto a second row, which also works.
- [ ] Leave it on **tab 1** with example ① loaded.

---

## Two presenters, one talk

The talk follows the slides in `presentation/slides.pdf` (kept out of git):

| Part | Who | What |
|---|---|---|
| Slides 1–5 | Presenter 1 | The task, the classical TF-IDF system, the NLP steps, how the systems are judged; then the handover |
| Slides 6–7 | Presenter 2 | The AI side and how the LLM scores |
| Slide 8 | Presenter 2 | **This demo**, steps 0–3 below (about 3 minutes) |
| Slides 9–12 | Presenter 2 | Results, error analysis, prompting vs. fine-tuning, conclusions |

The slides already introduce the task and present the results, so inside the talk the demo runs steps 0–3 and
skips steps 4–5. Use steps 4–5 only when demoing without the slides.

## Script

### 0 · Set the scene (15 s; 30 s without slides)
> "Let's see the four systems on real claims. I type a claim, and four search engines look for the paper
> that supports or refutes it. On the left, our **classical** system: it matches **words**, and rare words
> count more. Next, an AI model that matches **meaning**. Then the two combined. On the right, a **large
> language model** that reads each candidate paper and answers yes or no. For these example claims we know
> the correct paper, so the page marks it with a green ✓."

### 1 · Example ① "Bcl2 vs BCL-2 (spelling)" (already loaded) (2 min)
**Tab 1 · Compare**
- Word matching: **▼ correct paper only at position 32**. Meaning matching: **✓ position 1**. Combined:
  **position 7**. Then the LLM column fills in: **✓ position 1**.
> "The claim writes 'Bcl2'. The correct paper writes 'BCL-2', with a hyphen. To word matching those are
> different words. The AI models know they are the same gene."

**Tab 2 · Inside TF-IDF**
- **1a–1e:** the claim is split into words; each word gets its **part of speech** (noun, verb, adjective);
  the parser groups **noun phrases**; filler words like *of, is, the* are crossed out; words are cut to
  their roots (*silencing* → `silenc`, *tumors* → `tumor`).
- **Step 2:** `bcl2` appears in only **7** of 5,183 papers, so it gets the **highest rarity score (idf 7.6)**
  and dominates the claim.
- **Step 3:** TF-IDF's #1 result wins on `bcl2` **alone**. For the correct paper, `bcl2` shows *"not in this
  paper"*, because there it is written "BCL-2", which became `bcl` + `2`.
> "This is the vocabulary mismatch problem: exact matching fails when the same thing is written differently."

**Tab 3 · Inside the LLM**
- Point at the table: the LLM gives the correct paper a **~98%** probability of "yes". TF-IDF's #1 paper,
  which is also about Bcl2 but doesn't address the claim, gets only **~56%**, so the correct one moves to
  the top.
- Point at the prompt: "This is exactly what the model reads: an instruction, the claim, one paper, and it
  may only say yes or no. We read the probability of 'yes'. It does that 100 times per search."

### 2 · Example ③ "Rare disease name" (1 min)
**Tab 1**
- Word matching: **✓ position 1**. Meaning matching: **▼ only at position 59**. Combined: **position 3**.
  LLM: **✓ position 1**.

**Tab 2**
- **1e:** *ivermectin* is faded: it appears in **no** paper at all.
- **Step 2:** `onchocerciasi` appears in exactly **1** paper out of 5,183, so its **idf is 9.55**, the maximum.
- **Step 3:** that one rare word is the whole score.
> "Here word matching wins. A rare technical term is a precise fingerprint. The embedding model only has a
> vague idea of this disease and drifts to general drug-treatment papers. The LLM reads the actual paper,
> so it gets it right too."

### 3 · Live claim (1 min)
Type in the box and press Enter:

**`heart attack risk in people with diabetes`**

- Meaning matching finds *"…unstable angina and non-Q-wave **myocardial infarction**…"*: that is the medical
  term for a heart attack, although the words "heart attack" never appear.
- Word matching's top hit is a questionnaire paper that repeats *heart, risk, people, diabetes*.
> "Nobody wrote a synonym list: the model learned that 'heart attack' means 'myocardial infarction'."

**Audience suggestion?** Ask for a short **biomedical** claim; all the papers are medicine/biology. These
also work well: `cancer cells hide from the immune system` (tab 2 shows *hide* is in no paper),
`children who are overweight` (meaning matching also finds "obesity in children"). Typed claims get no ✓:
only the 300 benchmark claims have known answers (the 🎲 button picks one).

### 4 · Results on all 300 test claims (1.5 min; skip when the slides follow)
**Tab 4** has a plain-language guide above the table. The grade is **nDCG@10**: 1 = the correct paper is
always first, 0.5 ≈ usually third, 0 = not in the top 10.

| | nDCG@10 | Time per claim |
|---|---|---|
| Word matching · TF-IDF (from scratch) | 0.659 | **0.17 ms** on the CPU |
| Meaning matching · bge-small | 0.713 | 12 ms on the GPU |
| Combined (hybrid) | 0.721 | 14 ms |
| LLM zero-shot (general Qwen3, just prompted) | 0.276 | 5.7 s |
| **LLM reranker (Qwen3 trained for this)** | **0.772** | 5.8 s |

> "The LLM reranker is the best: 17% better than our TF-IDF, and significantly better than every other
> system. But TF-IDF is about 34,000 times faster, needs no GPU, and every score can be explained word by
> word, as tab 2 showed. And the same small LLM, only *prompted*, does far worse than TF-IDF: it says 'yes'
> to every single paper. Training on the task matters more than the model being an LLM."

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
| Why is the zero-shot LLM so bad? | It answers "yes" to 100% of the 29,889 candidate papers, so its ranking is close to random. The same model fine-tuned for relevance says "yes" to only 2.3% of candidates but to 75% of the correct ones. A 0.6B model can't follow "answer yes or no" discriminatively from the prompt alone; training on the task fixes that. |
| Why does the LLM still miss some? | It fails on 31 claims; on 20 of them TF-IDF never put the correct paper in its top 100, so the LLM never saw it. A better first stage (the hybrid) would fix many of these. |
| Why not a bigger LLM? | The laptop GPU has 4 GB. A 0.6B model already takes ~5 s per claim to read 100 papers; a 7B model would not fit, and an API model would cost money per query. |
| Why only nDCG@10? | It's the standard BEIR grade and rewards putting the correct paper high. MAP, MRR and precision are in the report and rank the systems in practically the same order. |
| How does the hybrid merge two lists? | Reciprocal Rank Fusion: score = Σ 1/(60 + position). It combines **positions**, not scores, so the two systems' different score scales never need matching. |
| Is the improvement significant? | Yes. A statistical test on the same 300 claims (Wilcoxon) gives p ≈ 5×10⁻¹¹ for the LLM reranker, far below 0.05, so it is not luck. bge and the hybrid are also really better than TF-IDF; MiniLM and the cross-encoder are not. (Details, including the correction for running several tests, are in the report.) |
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
