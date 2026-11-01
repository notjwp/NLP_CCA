# Qualitative examples (SciFact test)

## Where dense retrieval (bge-small) beats TF-IDF most

### Query 1088: “Silencing of Bcl2 is important for the maintenance and progression of tumors.”

- Lexical overlap with relevant abstract(s): 0.33; ΔnDCG@10 (bge − TF-IDF) = +1.000
- Relevant: [37549932] Antiapoptotic BCL-2 is required for maintenance of a model leukemia.
- Rank of first relevant abstract — TF-IDF cosine (lsc.lsc): 32; Dense: all-MiniLM-L6-v2: 1; Dense: bge-small-en-v1.5: 1; Hybrid: TF-IDF + bge (RRF): 7; TF-IDF → cross-encoder: 1; TF-IDF → LLM zero-shot (Qwen3-0.6B): 7; TF-IDF → LLM reranker (Qwen3-Reranker-0.6B): 1
- TF-IDF's top hit instead: [14863011] Bcl2 Regulation by the Melanocyte Master Regulator Mitf Modulates Lineage Survival and Melanoma Cell Viability

### Query 1278: “The treatment of cancer patients with co-IR blockade does not cause any adverse autoimmune events.”

- Lexical overlap with relevant abstract(s): 0.40; ΔnDCG@10 (bge − TF-IDF) = +1.000
- Relevant: [11335781] Is autoimmunity the Achilles' heel of cancer immunotherapy?
- Rank of first relevant abstract — TF-IDF cosine (lsc.lsc): 11; Dense: all-MiniLM-L6-v2: 1; Dense: bge-small-en-v1.5: 1; Hybrid: TF-IDF + bge (RRF): 1; TF-IDF → cross-encoder: 12; TF-IDF → LLM zero-shot (Qwen3-0.6B): 30; TF-IDF → LLM reranker (Qwen3-Reranker-0.6B): 1
- TF-IDF's top hit instead: [31293581] Ionizing radiation-induced long-term expression of senescence markers in mice is independent of p53 and immune status.

### Query 1382: “aPKCz causes tumour enhancement by affecting glutamine metabolism.”

- Lexical overlap with relevant abstract(s): 0.43; ΔnDCG@10 (bge − TF-IDF) = +1.000
- Relevant: [17755060] Control of Nutrient Stress-Induced Metabolic Reprogramming by PKCζ in Tumorigenesis
- Rank of first relevant abstract — TF-IDF cosine (lsc.lsc): 11; Dense: all-MiniLM-L6-v2: 1; Dense: bge-small-en-v1.5: 1; Hybrid: TF-IDF + bge (RRF): 4; TF-IDF → cross-encoder: 1; TF-IDF → LLM zero-shot (Qwen3-0.6B): 3; TF-IDF → LLM reranker (Qwen3-Reranker-0.6B): 2
- TF-IDF's top hit instead: [3831884] Glutamine supports pancreatic cancer growth through a Kras-regulated metabolic pathway

## Where TF-IDF beats dense retrieval (bge-small) most

### Query 660: “Ivermectin is used to treat onchocerciasis.”

- Lexical overlap with relevant abstract(s): 0.25; ΔnDCG@10 (bge − TF-IDF) = -1.000
- Relevant: [1215116] “Rapid-Impact Interventions”: How a Policy of Integrated Control for Africa's Neglected Tropical Diseases Could Benefit the Poor
- Rank of first relevant abstract — TF-IDF cosine (lsc.lsc): 1; Dense: all-MiniLM-L6-v2: >100; Dense: bge-small-en-v1.5: 59; Hybrid: TF-IDF + bge (RRF): 3; TF-IDF → cross-encoder: 57; TF-IDF → LLM zero-shot (Qwen3-0.6B): 51; TF-IDF → LLM reranker (Qwen3-Reranker-0.6B): 1
- bge-small's top hit instead: [2522977] Olmesartan-based therapies: an effective way to improve blood pressure control and cardiovascular protection.

### Query 674: “LDL cholesterol has no involvement in the development of cardiovascular disease.”

- Lexical overlap with relevant abstract(s): 0.83; ΔnDCG@10 (bge − TF-IDF) = -0.711
- Relevant: [2095573] LDL-cholesterol concentrations: a genome-wide association study
- Rank of first relevant abstract — TF-IDF cosine (lsc.lsc): 1; Dense: all-MiniLM-L6-v2: 7; Dense: bge-small-en-v1.5: 10; Hybrid: TF-IDF + bge (RRF): 4; TF-IDF → cross-encoder: 1; TF-IDF → LLM zero-shot (Qwen3-0.6B): 28; TF-IDF → LLM reranker (Qwen3-Reranker-0.6B): 1
- bge-small's top hit instead: [30981192] How to control residual cardiovascular risk despite statin treatment: focusing on HDL-cholesterol.

### Query 295: “Crosstalk between dendritic cells (DCs) and innate lymphoid cells (ILCs) is important in the regulation of intestinal homeostasis.”

- Lexical overlap with relevant abstract(s): 0.91; ΔnDCG@10 (bge − TF-IDF) = -0.685
- Relevant: [20310709] The Transcription Factor T-bet Regulates Intestinal Inflammation Mediated by Interleukin-7 Receptor+ Innate Lymphoid Cells
- Rank of first relevant abstract — TF-IDF cosine (lsc.lsc): 1; Dense: all-MiniLM-L6-v2: 3; Dense: bge-small-en-v1.5: 8; Hybrid: TF-IDF + bge (RRF): 5; TF-IDF → cross-encoder: 1; TF-IDF → LLM zero-shot (Qwen3-0.6B): 23; TF-IDF → LLM reranker (Qwen3-Reranker-0.6B): 1
- bge-small's top hit instead: [24044977] Innate lymphoid cells: emerging insights in development, lineage relationships, and function.
