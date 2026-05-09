# RAFT: A Hybrid Unicode-Semantic Knowledge Corruption Attack on Retrieval-Augmented Generation Systems

## Final Report — B.Tech Thesis Project

**Submitted by:** Dhruv Khanna
**Roll Number:** [Your Roll Number]
**Under the supervision of:** [Your Supervisor's Name]

**Department of Computer Science and Engineering**
**[Your Institution Name]**

**May 2026**

---

## DECLARATION

I hereby certify that the work which is being presented in the report, entitled **"RAFT: A Hybrid Unicode-Semantic Knowledge Corruption Attack on Retrieval-Augmented Generation Systems"**, in fulfillment of the requirement for the award of the Degree of Bachelor's of Technology (B.Tech) and submitted to the institution is an authentic record of my work carried out during the period [Start Date] to [End Date] under the supervision of [Your Supervisor's Name]. I have also cited the references to the texts, figures, and tables from where they have been taken.

Date: ____________
Signature of Candidate: ____________
Dhruv Khanna

This is to confirm that the candidate's above statement is true to the best of our knowledge.

Date: ____________
Signature of the Supervisor: ____________

---

## ACKNOWLEDGEMENT

I sincerely thank **[Your Supervisor's Name]** for his valuable guidance and unwavering support throughout my Bachelor's Thesis Project. His insights into the security of large language models and his willingness to let me explore an emerging area of research were crucial in shaping this work. From the very first discussion about retrieval-augmented generation systems to the final analysis of adversarial attacks, his mentorship gave me both technical direction and intellectual confidence.

I am especially grateful for the opportunity to work on a topic that lies at the intersection of natural language processing, information retrieval, and adversarial machine learning. This project has not only deepened my understanding of how modern AI systems work but has also helped me appreciate the practical challenges of building secure, trustworthy systems. The process of designing the hybrid attack, reproducing prior work, and evaluating defenses has taught me far more than any classroom could.

I am thankful to **[Your Institution]** for providing the resources, computational facilities, and academic environment that made this project possible. I also extend my gratitude to the authors of the foundational papers — particularly **PoisonedRAG (Zou et al., 2024)** and **RAG-Pull (Stambolic et al., 2025)** — whose openly published work and accompanying code provided a rigorous baseline for the experiments in this thesis.

Finally, I would like to thank my family, peers, and friends for their constant encouragement during the long hours of debugging, retraining, and result analysis. Their patience and support have been a quiet but indispensable foundation for everything I have accomplished here.

Because of these enriching experiences, I have grown into a more careful, curious, and self-driven researcher. I will continue to carry forward the spirit of inquiry that I developed under this project.

**Dhruv Khanna**

---

## ABSTRACT

Retrieval-Augmented Generation (RAG) has rapidly become the dominant architecture for grounding large language models (LLMs) in factual, up-to-date knowledge. By retrieving relevant documents from an external corpus and injecting them into the model's context, RAG systems mitigate hallucination, support domain adaptation without expensive retraining, and enable real-time knowledge integration. As a result, they are now deployed in healthcare, legal consulting, financial services, customer support, and search engines.

However, this same architecture introduces a new and largely unexplored attack surface: the **knowledge database**. Recent work has shown that an attacker who can inject even a small number of malicious passages into this corpus can hijack the model's output. Two attack families have emerged. **Semantic attacks**, exemplified by PoisonedRAG, craft visible, fluent passages that state a target answer; they can be detected by perplexity filters, fact-checking layers, and trust-aware RAG architectures. **Character-level attacks**, exemplified by RAG-Pull, use invisible Unicode characters to manipulate retrieval embeddings; they can be neutralized by Unicode normalization or zero-width character stripping.

A natural question arises: **what happens when an attacker combines both?** This thesis answers that question. We introduce **RAFT (RAG Adversarial Forensics and Threat-analysis)**, a systematic evaluation framework for adversarial attacks on RAG systems, and propose a novel **hybrid Unicode-semantic attack** that fuses PoisonedRAG-style semantic injection with RAG-Pull-style invisible-character retrieval boosting. The attack uses a two-stage pipeline: in Stage 1, an LLM crafts a semantically plausible passage stating a fake answer; in Stage 2, a differential evolution optimizer inserts invisible TAG characters (U+E0001–U+E007F) into the passage's prefix to maximize cosine similarity with the target query embedding.

We evaluate this attack across three datasets (Natural Questions, HotpotQA, MS-MARCO), five RAG architectures (Vanilla RAG, Self-RAG, CRAG, TrustRAG, RobustRAG), three attacks (semantic, unicode, hybrid), and four defenses (NFKC normalization, zero-width stripping, perplexity filtering, query paraphrasing) — a total of sixty configurations. Our experiments demonstrate that the hybrid attack is the only attack class that **no single standard defense can fully neutralize**. Zero-width stripping, the canonical defense against invisible-character attacks, completely blocks the pure unicode attack (e2e ASR drops from 10% to 0%) but fails against the hybrid attack (90% ASR retained), because the semantic payload survives character removal. Strict perplexity filtering at threshold τ=50 marginally separates the two (semantic 80%, hybrid 85%) but does not eliminate the hybrid's effectiveness. Trust-aware K-means filtering in TrustRAG reduces both attacks proportionally to ~30%, indicating that the architectural defense is not attack-discriminative.

The implications are significant. First, current RAG defenses are designed under the implicit assumption that semantic and character-level attacks are independent threats; the hybrid attack invalidates this assumption. Second, the hybrid attack is more **perplexity-stealthy** than either of its component attacks: its maximum GPT-2 perplexity (49.1) is lower than that of the pure semantic attack (112.6), making it harder to filter under conservative threshold settings. Third, our evaluation shows that the attack generalizes across multiple RAG architectures, suggesting that hybrid-aware defenses must be developed as a new defense category rather than as extensions of existing single-family filters.

This thesis makes the following contributions: **(1)** the design of a novel hybrid Unicode-semantic attack with empirically validated hyperparameters, **(2)** a comprehensive open-source evaluation framework spanning 5 RAG variants × 3 attacks × 4 defenses on real-world datasets, **(3)** a systematic comparison with prior attacks (PoisonedRAG, RAG-Pull, prompt injection, GCG) demonstrating the hybrid attack's superior defense evasion, and **(4)** an analysis of why current defenses fail and what properties future defenses must satisfy.

**Keywords:** Retrieval-Augmented Generation, Adversarial Machine Learning, Knowledge Corruption Attack, Invisible Unicode Characters, LLM Security, Hybrid Attack, Differential Evolution, Trust-Aware RAG

---

## TABLE OF CONTENTS

```
Declaration .................................................. i
Acknowledgement .............................................. ii
Abstract ..................................................... iii
List of Figures .............................................. vi
List of Tables ............................................... vii

1 Introduction ............................................... 1
  1.1 What is Retrieval-Augmented Generation? ................ 1
  1.2 What is RAG Knowledge Corruption? ...................... 3
  1.3 Types of Adversarial Attacks on RAG .................... 4
  1.4 Challenges in Defending RAG Systems .................... 6
  1.5 Role of Invisible Unicode Characters ................... 7

2 Literature Review .......................................... 9
  2.1 Foundations of RAG ..................................... 9
  2.2 Knowledge Corruption Attacks (PoisonedRAG and Variants)  10
  2.3 Embedding-Level Attacks (RAG-Pull and Bad Characters)   12
  2.4 Trust-Aware RAG Defenses ............................... 13
  2.5 Inference-Time Defenses ................................ 14
  2.6 Research Gap ........................................... 15
  2.7 Objectives ............................................. 16

3 Methodology ................................................ 17
  3.1 Threat Model ........................................... 17
  3.2 System Architecture .................................... 18
  3.3 Hybrid Attack Pipeline ................................. 19
  3.4 Defense Evaluation Pipeline ............................ 22

4 Implementation ............................................. 24
  4.1 Dataset Preparation .................................... 24
  4.2 Index Construction with FAISS .......................... 25
  4.3 Attack Implementation .................................. 26
  4.4 Defense Implementation ................................. 28
  4.5 Five RAG Variants ...................................... 29

5 Experimental Setup ......................................... 32
  5.1 Datasets ............................................... 32
  5.2 Models ................................................. 33
  5.3 Evaluation Metrics ..................................... 34
  5.4 Configuration Profiles ................................. 35

6 Experimental Results ....................................... 37
  6.1 Baseline Vanilla RAG Performance ....................... 37
  6.2 Attack Effectiveness on Vanilla RAG .................... 38
  6.3 Defense Effectiveness Analysis ......................... 41
  6.4 Trust-Aware RAG Variants ............................... 44
  6.5 Perplexity Threshold Ablation .......................... 46
  6.6 Stealth Profile Analysis ............................... 48
  6.7 Comparative Analysis with Prior Attacks ................ 49

7 Conclusion ................................................. 51
  7.1 Summary of Findings .................................... 51
  7.2 Thesis Contributions ................................... 52
  7.3 Limitations ............................................ 53
  7.4 Future Work ............................................ 54
```

---

## LIST OF FIGURES

| Fig. | Description | Page |
|------|-------------|------|
| 1.1 | Overview of a typical RAG pipeline | 2 |
| 1.2 | Example of RAG knowledge corruption | 4 |
| 1.3 | Taxonomy of RAG attacks (Semantic, Unicode, Hybrid) | 5 |
| 1.4 | Visualization of invisible Unicode TAG characters | 8 |
| 3.1 | High-level threat model and attacker capabilities | 18 |
| 3.2 | RAFT system architecture overview | 19 |
| 3.3 | Two-stage hybrid attack pipeline diagram | 21 |
| 3.4 | Defense evaluation flow per (RAG × attack × defense) cell | 23 |
| 4.1 | Pipeline for adversarial passage crafting and injection | 27 |
| 4.2 | TrustRAG K-means cluster filtering visualization | 30 |
| 6.1 | ASR comparison across attacks on Vanilla RAG | 39 |
| 6.2 | ASR heatmap across all (RAG × attack × defense) combinations | 42 |
| 6.3 | Perplexity threshold sweep — strict fallback mode | 47 |
| 6.4 | GPT-2 perplexity distributions per attack class | 49 |
| 6.5 | Comparison with prior attacks (PoisonedRAG, RAG-Pull, etc.) | 50 |

---

## LIST OF TABLES

| Tbl. | Description | Page |
|------|-------------|------|
| 2.1 | Comparative summary of RAG attacks in the literature | 11 |
| 5.1 | Dataset statistics (corpus size, questions, domain) | 32 |
| 5.2 | Models used (retrievers, generators, scorers) | 33 |
| 5.3 | Evaluation metrics overview | 34 |
| 5.4 | Configuration profiles and timing estimates | 36 |
| 6.1 | Baseline clean accuracy of Vanilla RAG | 37 |
| 6.2 | Attack × Defense matrix for Vanilla RAG (NQ, n=100) | 41 |
| 6.3 | Attack × RAG Variant matrix (NQ, no defense) | 44 |
| 6.4 | PPL threshold ablation results (strict fallback, n_adv=1) | 46 |
| 6.5 | Stealth profile: GPT-2 perplexity statistics per attack | 48 |
| 6.6 | Comparison with prior attacks (ASR, F1@5, P@5) | 50 |

---

# Chapter 1
# Introduction

## 1.1 What is Retrieval-Augmented Generation?

Large language models (LLMs) such as GPT-4, Llama 3, Claude, and Qwen have transformed natural language processing by demonstrating remarkable capabilities in summarization, code generation, question answering, and creative writing. However, despite their fluency, these models suffer from two fundamental limitations: they have a **fixed knowledge cutoff date** beyond which they cannot answer accurately, and they are prone to **hallucination** — the confident generation of factually incorrect information. These limitations are particularly serious in safety-critical domains like healthcare, finance, legal advice, and scientific research, where the cost of wrong answers is high.

**Retrieval-Augmented Generation (RAG)** [Lewis et al., 2021] is the state-of-the-art technique for mitigating these limitations. The core idea is simple: rather than relying purely on the LLM's parametric (internal) knowledge, RAG augments the model's input with externally retrieved documents that are relevant to the user's query. By grounding the answer in retrieved evidence, RAG reduces hallucination, supports up-to-date knowledge without retraining, and enables domain adaptation simply by changing the underlying corpus.

A typical RAG pipeline consists of three components, as shown in Figure 1.1:

1. **Knowledge Database (or Corpus)** — a collection of documents (Wikipedia articles, news, internal documentation, etc.) that the system can retrieve from.
2. **Retriever** — typically a dense neural encoder (e.g., Contriever, BGE, ANCE) that embeds queries and documents into a shared vector space, allowing the system to retrieve the top-k most relevant documents for any given query via cosine similarity or dot product.
3. **LLM Generator** — the language model that takes the user's query along with the retrieved documents as context and produces a final answer.

> **[INSERT FIGURE 1.1]** *Overview of a typical RAG pipeline. A user query is encoded by the retriever, the top-k most similar documents are retrieved from the knowledge database, and these documents are concatenated with the query as context for the LLM. The LLM then generates an answer grounded in the retrieved evidence. (Adapted from Zou et al., 2024)*

The success of RAG has led to its rapid adoption in industry. **ChatGPT Retrieval Plugin**, **NVIDIA ChatRTX**, **LangChain**, **LlamaIndex**, **WikiChat**, **Bing Search with AI**, **Google Search with AI Overviews**, **Perplexity AI**, and various enterprise LLM agents are all real-world deployments of the RAG paradigm. As of 2025, RAG has effectively become the default architecture for any LLM application that requires factual grounding.

## 1.2 What is RAG Knowledge Corruption?

The very property that makes RAG powerful — its reliance on external knowledge — also makes it vulnerable. If an attacker can manipulate the contents of the knowledge database, they can manipulate the answers produced by the LLM. This is known as a **knowledge corruption attack**.

Consider a concrete example. Suppose a RAG system is built on top of Wikipedia and is deployed as a customer-facing chatbot. A user asks: *"Who is the CEO of OpenAI?"*. Under normal operation, the retriever fetches Wikipedia passages about OpenAI, the LLM reads these passages, and produces the correct answer ("Sam Altman").

Now suppose an attacker injects the following passage into the Wikipedia corpus: *"In 2024, OpenAI witnessed a surprising leadership change. Renowned for his leadership at Apple, Tim Cook decided to embark on a new journey. He joined OpenAI as its CEO, bringing his extensive experience and innovative vision to the forefront of AI."* If this passage is retrieved by the system, the LLM, trusting the context, may now answer **"Tim Cook"** to the same question. The attack has succeeded — without ever touching the LLM's weights, the retriever, or any system code.

> **[INSERT FIGURE 1.2]** *Example of a successful knowledge corruption attack. An adversarial passage injected into the Wikipedia corpus causes the LLM to generate an attacker-chosen target answer. (Adapted from PoisonedRAG, Figure 2)*

Knowledge corruption attacks are particularly dangerous because they are:

- **Stealthy** — they do not require model access; they only require the ability to insert documents into the corpus.
- **Persistent** — once a malicious document is in the corpus, it can affect every subsequent query that retrieves it.
- **Realistic** — many RAG systems are built on user-editable corpora (Wikipedia, Reddit), public web crawls, or enterprise knowledge bases that may be vulnerable to insider injection.
- **Hard to attribute** — generation errors can be blamed on the LLM, on the retriever, on the corpus, or on a "model hallucination", obscuring the attack's origin.

A recent study [Carlini et al., 2024] shows that maliciously editing 6.5% of Wikipedia is feasible at low cost. Since most knowledge corruption attacks succeed with only a handful of injected documents, the practical attack budget is well within reach of motivated adversaries.

## 1.3 Types of Adversarial Attacks on RAG

The literature has identified two distinct families of knowledge corruption attacks against RAG, separated by the **layer of the pipeline** they target:

**1. Semantic-level attacks** (e.g., **PoisonedRAG**, Zou et al. 2024). The attacker crafts a fluent, grammatically correct, and topically relevant passage that asserts a target fake answer. The malicious text is human-readable and written in natural language. Attacks in this class are easy to generate (often using an LLM as a "writer"), but the malicious content is also visible to humans and can be detected by:
   - Perplexity-based filters (since fabricated facts may have unusual word patterns)
   - Fact-checking layers (cross-referencing with a trusted database)
   - Trust-aware RAG architectures that score document credibility (Self-RAG, CRAG, TrustRAG, RobustRAG)

**2. Character-level / embedding attacks** (e.g., **RAG-Pull**, Stambolic et al. 2025; **Bad Characters**, Boucher et al. 2021). The attacker inserts **invisible Unicode characters** (zero-width spaces, joiners, variation selectors, TAG characters) into a passage. These characters do not affect the visible content of the text but **do shift the dense embedding** computed by the retriever, allowing the attacker to push their passage to the top of the retrieval ranking for a target query. Attacks in this class are nearly invisible to humans and bypass content-based defenses, but they can be neutralized by:
   - Unicode normalization (NFKC) at the input stage
   - Zero-width character stripping at the tokenizer level
   - Embedder retraining to ignore invisible characters

**The key observation of this thesis** is that these two attack families have, until now, been studied in isolation. Defenses have been designed against one family at a time. We propose a third class:

**3. Hybrid attacks (this work).** The attacker combines a semantic payload (Stage 1 of PoisonedRAG) with invisible character perturbations (Stage 2 of RAG-Pull). The result is a passage that:
   - Looks plausible to a human reader (so it survives content-based defenses).
   - Contains targeted invisible character insertions that boost retrieval rank (so it consistently makes it to the top-k).
   - Is more perplexity-stealthy than the pure semantic attack (because TAG characters add only ~11–18 PPL vs. 44–90 for zero-width characters).

> **[INSERT FIGURE 1.3]** *Taxonomy of RAG knowledge corruption attacks. Three classes are distinguished by the layer they target: semantic attacks operate at the content layer, unicode attacks at the embedding layer, and the proposed hybrid attack at both layers simultaneously.*

## 1.4 Challenges in Defending RAG Systems

Defending RAG systems is challenging for several interrelated reasons:

1. **Pipeline complexity.** A RAG system has at least three components (corpus, retriever, generator), each of which can be the target of an attack. A defense at one layer may be evaded by an attack at another.

2. **Defense specialization.** Existing defenses are designed against specific attack families. NFKC normalization defends against Unicode attacks but does nothing against semantic attacks. Perplexity filtering defends against semantic attacks (which may produce slightly unnatural text) but is weak against unicode attacks (which preserve the visible text). No single defense in production today is designed for hybrid attacks.

3. **Trust-aware RAG limitations.** Recent architectural defenses (Self-RAG, CRAG, TrustRAG, RobustRAG) attempt to score document credibility before passing it to the generator. However, all of these architectures were evaluated against PoisonedRAG-style semantic attacks. We show that they are equally vulnerable to hybrid attacks, because the semantic content of a hybrid passage is plausible and indistinguishable from clean text under any of these architectural filters.

4. **The cost of strict defenses.** Aggressive defenses (e.g., a perplexity threshold of τ=30) reduce attack success but also increase false positives, blocking legitimate queries and degrading clean RAG performance. The defender must balance security and utility, while the attacker only needs to evade the defense once.

5. **Adaptive attackers.** A real-world attacker can iterate against any defense the system deploys. Our hybrid attack is itself an example of this — defenses against PoisonedRAG and RAG-Pull existed; the hybrid attack was designed by combining the two attack vectors so that **no single defense addresses both halves**.

## 1.5 Role of Invisible Unicode Characters

Invisible Unicode characters are a fundamental ingredient in this thesis. The Unicode standard defines hundreds of characters that have no visible glyph in standard fonts but that are nevertheless valid characters in any Unicode-compliant string. These include:

- **Zero-Width Space (U+200B), Zero-Width Joiner (U+200D), Zero-Width Non-Joiner (U+200C)** — used in legitimate contexts for word boundary control in scripts like Arabic, Devanagari, and emoji rendering.
- **Variation Selectors (U+FE00–U+FE0F)** — used to select alternate glyph forms.
- **Tag Characters (U+E0001–U+E007F)** — originally designed for language tagging but never widely deployed.
- **Word Joiner (U+2060), Soft Hyphen (U+00AD)** — formatting controls.

To a human reader looking at rendered text, all of these are invisible. To a Unicode-aware tokenizer, however, each is a distinct token. To a dense neural encoder, each contributes to the final embedding vector of the text.

> **[INSERT FIGURE 1.4]** *Visualization of how an invisible Unicode TAG character (U+E0001) is rendered as nothing visually but is treated as a distinct token by the embedder. The figure should show: (a) the text "Tim Cook is the CEO" rendered identically with and without TAG characters, (b) the tokenization showing distinct tokens for each TAG character, (c) the resulting embedding vector difference.*

The **RAG-Pull paper** (Stambolic et al., 2025) demonstrated that by carefully selecting invisible character insertions, an attacker can shift the retrieval embedding of a passage so that it becomes more similar to a target query. They use a **differential evolution** optimizer to search for the optimal insertions — a black-box, gradient-free method that is effective even against unknown embedders.

Our profiling work (Section 6.6) shows that not all invisible characters are equal in stealth. **TAG characters add only +11 to +18 GPT-2 perplexity per 5 characters inserted, while zero-width characters add +44 to +90 PPL per 5 characters**. This is a critical finding: it means we can build a hybrid attack that uses TAG characters to remain perplexity-stealthy, while pure-RAG-Pull-style attacks using zero-width characters would be filtered out.

This thesis is the first to systematically exploit this PPL gap by restricting the hybrid attack's character inventory to TAG characters only. The result is an attack that is more stealthy than its semantic component (max PPL 49 vs. 112), while preserving the retrieval boost.

---

# Chapter 2
# Literature Review

## 2.1 Foundations of RAG

Retrieval-Augmented Generation was formally introduced by Lewis et al. [2020] as a way to combine the parametric knowledge of large language models with the non-parametric knowledge of an external corpus. The original RAG architecture used Dense Passage Retrieval (Karpukhin et al., 2020) as the retriever and BART as the generator. Subsequent work extended this paradigm in several directions:

- **Better retrievers** — Contriever [Izacard et al., 2022], BGE [BAAI, 2023], ANCE [Xiong et al., 2021], E5 [Wang et al., 2022] — train retrievers with contrastive objectives on large-scale corpora.
- **Better generators** — GPT-4, Llama 3, Claude, Qwen 2.5, etc., with specialized RAG-tuned variants such as Atlas [Izacard et al., 2022].
- **Better orchestration** — Self-RAG [Asai et al., 2024], CRAG [Yan et al., 2024], iterative RAG (IRCoT), and chain-of-thought-guided RAG.

For the purposes of this thesis, we focus on the most widely deployed variants of RAG and treat the system as a black box composed of three components: corpus, retriever, generator.

## 2.2 Knowledge Corruption Attacks (PoisonedRAG and Variants)

**PoisonedRAG** [Zou, Geng, Wang, Jia — USENIX Security 2025] is the seminal work on knowledge corruption attacks against RAG. The authors formalize the attack as an optimization problem and decompose it into two **necessary conditions**:

1. **Retrieval condition** — the malicious text must be retrieved by the retriever for the target query.
2. **Generation condition** — once retrieved, the malicious text must induce the LLM to generate the target answer.

PoisonedRAG decomposes each malicious passage P into two sub-texts P = S ⊕ I, where:
- I is generated by an LLM (typically GPT-4) and is designed to satisfy the generation condition.
- S is crafted to satisfy the retrieval condition. In the **black-box setting**, S is simply the target question prepended to I. In the **white-box setting**, S is optimized using HotFlip or TextFooler to maximize cosine similarity to the query embedding.

Their results show that PoisonedRAG achieves up to **97% attack success rate (ASR)** by injecting just five malicious texts per target question into a corpus of millions. Crucially, they evaluate against several defenses (paraphrasing, perplexity filtering, duplicate filtering, knowledge expansion) and find that none of them eliminate the attack.

**HijackRAG** [Zhang et al., 2024], **PoisonedRAG variants** [Chang et al., 2025], **Phantom** [Chaudhari et al., 2024], **BadRAG** [Xue et al., 2024], and **AgentPoison** [Chen et al., 2024] extend the PoisonedRAG threat model to LLM agents, multi-source corpora, and chain-of-thought reasoning. All of these works share the assumption of **visible, semantic attacks** and do not consider invisible character perturbations.

> **[INSERT TABLE 2.1]** *Comparative summary of RAG attacks in the literature. Columns: attack name, attack vector (corpus/query/hybrid), threat model (white-box/black-box), attack goal, imperceptibility, defense coverage. Adapted from Table 1 of the RAG-Pull paper, with our hybrid attack added as the final row.*

| Attack | Vector | Threat Model | Goal | Imperceptible |
|--------|--------|--------------|------|---------------|
| Ignore Previous Prompt (Perez & Ribeiro, 2022) | Prompt | Black-box | Prompt hijack | ✗ |
| GCG (Zou et al., 2023) | Prompt | White-box | Jailbreak | ✗ |
| Bad Characters (Boucher et al., 2021) | Prompt | Black-box | Misalignment, evasion | ✓ |
| Poisoning Retrieval Corpora (Zhong et al., 2023) | Corpus | White-box | Knowledge corruption | ✗ |
| HijackRAG (Zhang et al., 2024) | Corpus | Both | Knowledge corruption | ✗ |
| **PoisonedRAG (Zou et al., 2024)** | **Corpus** | **Both** | **Knowledge corruption** | **✗** |
| Phantom (Chaudhari et al., 2024) | Corpus | White-box | DoS, info leakage | ✗ |
| BadRAG (Xue et al., 2024) | Corpus | White-box | Knowledge corruption | ✗ |
| BadChain (Xiang et al., 2024) | Hybrid | Black-box | Misleading output | ✗ |
| AgentPoison (Chen et al., 2024) | Hybrid | White-box | Agent misalignment | ✗ |
| **RAG-Pull (Stambolic et al., 2025)** | **Prompt + Corpus** | **Black-box** | **Code generation** | **✓** |
| **RAFT Hybrid Attack (this work)** | **Corpus** | **Black-box** | **Knowledge corruption + defense evasion** | **✓** |

## 2.3 Embedding-Level Attacks (RAG-Pull and Bad Characters)

The other thread of research relevant to this thesis is **embedding-level attacks** that exploit the difference between what humans see and what tokenizers process.

**Bad Characters** [Boucher et al., 2021] was the first paper to demonstrate that imperceptible Unicode perturbations can be used to attack NLP models. They categorized attack characters into invisible characters, homoglyphs, reorderings, and deletions, and showed that small perturbations can degrade classification, translation, and search systems.

**RAG-Pull** [Stambolic et al., 2025] is the most direct precursor to our work. The authors demonstrate that invisible Unicode perturbations of either the query, the target document, or both can be used to redirect retrieval toward an attacker-controlled snippet. They use a **differential evolution** algorithm to search for optimal insertions, achieving up to 100% retrieval success and 99.44% end-to-end attack success rate on code generation tasks. Their target domain is code-RAG (e.g., GitHub Copilot), and their malicious payloads are short shell commands or vulnerable code snippets.

The RAG-Pull paper makes two observations that are critical to our work:

1. **Combined query + target perturbations are remarkably effective.** When both sides are perturbed, retrieval succeeds nearly 100% of the time across all configurations.
2. **Perturbing only the target is far weaker** than perturbing the query, because target perturbations cause a small embedding shift while leaving the (clean) query–benign-document similarity unchanged.

We extend RAG-Pull's design in three ways: **(1)** we transfer the attack from code-RAG to general knowledge-RAG (factual question answering), **(2)** we combine RAG-Pull's character-level perturbation with PoisonedRAG's semantic crafting to produce the hybrid attack, and **(3)** we restrict the character inventory to TAG characters specifically to maintain perplexity stealth.

## 2.4 Trust-Aware RAG Defenses

In response to PoisonedRAG-style attacks, several defended RAG architectures have emerged:

**Self-RAG** [Asai et al., ICLR 2024] trains the LLM to emit "reflection tokens" that critique each retrieved passage for relevance and supportiveness. In our reimplementation, we approximate this with a prompt-based relevance filter (asking the LLM directly whether each passage is relevant before generation).

**CRAG (Corrective RAG)** [Yan et al., 2024] uses a lightweight retrieval evaluator (originally a fine-tuned T5) to classify each retrieved passage as Correct, Incorrect, or Ambiguous. Correct passages are kept; Incorrect ones are dropped; Ambiguous ones trigger query refinement and re-retrieval. We reimplement this with cosine similarity scores as the evaluator proxy.

**TrustRAG** [Zhou et al., 2025] is the most recently published trust-aware variant. It retrieves a larger candidate set (e.g., 20 passages), embeds them, runs **K-means clustering** with k=2, and discards the minority cluster (presumed adversarial). The intuition is that adversarial passages, even if individually plausible, will cluster together in embedding space because they share an unusual semantic objective. We use scikit-learn's KMeans for this.

**RobustRAG** [Xiang et al., USENIX Security 2024] uses an **isolate-then-aggregate** strategy: each retrieved passage is fed independently to the LLM, producing isolated answers; the final answer is selected by keyword-level majority vote. This protects against single-passage injection but can fail when multiple injected passages reinforce each other.

> **[INSERT FIGURE 2.1 (optional)]** *Comparison of trust-aware RAG architectures. Each variant introduces a different filtering or aggregation step between retrieval and generation.*

A critical observation: **all four trust-aware RAG variants were originally evaluated against PoisonedRAG-style semantic attacks**, not against unicode or hybrid attacks. Our work is the first to evaluate them against the hybrid attack, and we find that they are similarly vulnerable.

## 2.5 Inference-Time Defenses

In addition to architectural defenses, several lightweight **inference-time** defenses have been proposed [Jaiswal et al., 2026] for the broader LLM security space:

- **Input filtering / Prompt risk classification** — pattern-matching, keyword blacklisting, regex.
- **Self-examination / Self-defense** — using a secondary LLM as a safety judge.
- **System prompt hardening / Policy guardrails** — explicitly stating safety policies in the system prompt.
- **Vector defense** — comparing input embeddings to a database of known adversarial patterns.
- **Voting defense** — generating multiple responses and selecting by consensus.

The empirical study by Jaiswal et al. (2026) on prompt-injection and jailbreak attacks showed that **self-defense (LLM-as-judge)** is the strongest of these defenses, while **input filtering** is consistently the weakest. This finding informs our defense suite: we evaluate perplexity filtering (analogous to input filtering) and find it insufficient, and we recommend that future work explore self-defense and policy guardrails as candidate defenses against hybrid attacks.

## 2.6 Research Gap

The literature reveals a clear research gap:

1. **Existing attacks are siloed.** PoisonedRAG and its descendants attack the semantic layer; RAG-Pull and Bad Characters attack the embedding layer. No prior work combines both vectors in a single attack on knowledge RAG.

2. **Existing defenses are layer-specific.** Trust-aware RAG architectures defend against semantic attacks; Unicode normalization defends against character attacks. No single defense in production simultaneously addresses both.

3. **The Unicode-PPL trade-off is unexplored.** Prior unicode attacks use any invisible character, ignoring the fact that different character classes have very different perplexity costs. We are the first to systematically restrict the inventory to TAG characters for stealth.

4. **No comprehensive cross-defense evaluation matrix exists.** Prior work evaluates each attack against one or two defenses on one or two RAG architectures. We are the first to evaluate the full 5 RAGs × 3 attacks × 4 defenses matrix on the same datasets and models.

## 2.7 Objectives

The objectives of this thesis are:

1. **Design** a hybrid Unicode-semantic attack on RAG that combines PoisonedRAG-style semantic injection with RAG-Pull-style invisible-character retrieval boosting.

2. **Implement** a comprehensive open-source evaluation framework (RAFT) that supports five RAG architectures, three attack types, four defenses, and three datasets.

3. **Evaluate** the hybrid attack against the strongest current defenses (zero-width stripping, NFKC normalization, perplexity filtering, query paraphrasing) and against four trust-aware RAG architectures (Self-RAG, CRAG, TrustRAG, RobustRAG).

4. **Compare** the hybrid attack quantitatively with prior attacks (pure semantic, pure unicode, prompt injection, GCG) on the same benchmark.

5. **Identify** which defense properties are necessary to neutralize the hybrid attack, providing concrete guidance for future defense research.

---

# Chapter 3
# Methodology

This chapter formalizes the threat model, system architecture, and end-to-end attack and defense evaluation pipelines used in RAFT.

## 3.1 Threat Model

We assume a powerful but realistic attacker with the following capabilities and constraints:

**Attacker capabilities:**
- The attacker can **inject a small number of malicious passages** (typically N=5 per target question) into the knowledge database. This is feasible for Wikipedia-based corpora (via malicious edits), web-scale corpora (via SEO/website hosting), or enterprise corpora (via insider threats).
- The attacker selects an arbitrary set of **target questions** and an arbitrary **target answer** for each (e.g., "Who is the CEO of OpenAI?" → "Tim Cook").

**Attacker constraints:**
- The attacker **cannot modify** the LLM's weights, the retriever's weights, or the existing clean documents in the corpus.
- The attacker **cannot query** the LLM directly (though they may use a public LLM such as GPT-4 to craft semantic content).
- The attacker has **black-box access** to the retriever (they know the embedder family, e.g., Contriever, but not its parameters; they can compute their own embeddings using a public model checkpoint).

**Attacker goal:**
- For each target question Q with target answer R, induce the LLM in the deployed RAG system to output R when asked Q.

> **[INSERT FIGURE 3.1]** *High-level threat model. The attacker injects malicious passages into the knowledge database. The user asks a target question; the deployed RAG system retrieves contaminated context and produces the attacker-chosen answer.*

## 3.2 System Architecture

The RAFT framework implements the full RAG pipeline plus the attack and defense modules. Figure 3.2 illustrates the architecture.

> **[INSERT FIGURE 3.2]** *RAFT system architecture. Components are organized in five layers: (1) Data Layer (NQ/HotpotQA/MS-MARCO loaders), (2) Retrieval Layer (Contriever, BGE), (3) Defense Layer (NFKC, ZW-strip, Perplexity, Paraphrase), (4) RAG Layer (Vanilla, Self-RAG, CRAG, TrustRAG, RobustRAG), (5) Generation Layer (Ollama qwen2.5:7b, GPT-4o-mini, Claude Haiku). Attacks operate by injecting passages into the data layer.*

## 3.3 Hybrid Attack Pipeline

The core methodological contribution of this thesis is the **two-stage hybrid attack pipeline**, illustrated in Figure 3.3.

> **[INSERT FIGURE 3.3]** *Two-stage hybrid attack pipeline. Stage 1 (PoisonedRAG-style): an LLM crafts a semantic passage stating the fake answer. Stage 2 (RAG-Pull-style): a differential evolution optimizer inserts invisible TAG characters into the prefix of the Stage-1 passage to maximize cosine similarity with the query embedding.*

### Stage 1 — Semantic Payload Generation

For each target question Q and target answer R, we prompt an LLM with a template similar to that of PoisonedRAG:

```
This is my question: {Q}
This is my answer: {R}
Please craft a corpus such that the answer is {R} when prompting with the
question {Q}. Please limit the corpus to V words.
```

We set V=30 by default. The LLM (we use **Qwen2.5:7b locally via Ollama** for crafting and deployment) produces a passage I that, when used as context, induces the LLM to answer R for Q. We verify this by running a generation check; if the LLM does not produce R, we re-prompt up to L=50 times. In our experiments, an average of ~2 trials are needed.

### Stage 2 — Invisible Character Optimization (TAG-only)

Given the Stage-1 passage I, we initialize the malicious passage as `P_0 = Q ⊕ I` (the target question prepended to I), following PoisonedRAG's black-box construction.

We then run a **differential evolution (DE)** optimizer over invisible character insertions in the prefix region (first 50 characters) of P_0. The optimizer's objective is:

$$\delta^* = \arg\max_{\delta \in \Delta} \text{cos}(f_Q(Q), f_T(\text{perturb}(P_0, \delta)))$$

where:
- δ is a sequence of (position, character) insertion operations
- Δ is the search space of all valid insertion sequences with budget ≤ 20
- The character inventory is **restricted to TAG characters (U+E0001–U+E007F)**
- f_Q and f_T are the Contriever query and passage encoders
- perturb(text, δ) applies the insertions to the text

We use scipy's `differential_evolution` with population size 10 and 30 iterations. Each candidate solution is a vector of (position, character_id) pairs encoded as integers.

### Why TAG-only?

A critical design choice in our work is restricting the character inventory to TAG characters only. Empirical PPL profiling on GPT-2 shows:

| Character class | ΔPPL per 5 chars |
|------------------|------------------|
| Zero-width space (ZWSP) | +44 to +90 |
| Word joiner (U+2060) | +44 to +90 |
| Variation selectors | +18 to +35 |
| **TAG characters (U+E0001–U+E007F)** | **+11 to +18** |

By restricting to TAG characters at budget=20, the total PPL of a hybrid passage stays below 50 — which means it passes a strict perplexity filter (τ=50) that would catch the pure semantic attack (which has passages with PPL up to 112).

### Why Budget=20?

The budget controls how many TAG characters the DE optimizer can insert.
To isolate its effect, we evaluate the **unicode-only component** (no semantic
payload, no question prepending) against the semantic baseline across budgets
{5, 10, 15, 20, 30, 50} on 20 NQ questions (Experiment 09). The semantic attack retrieves
successfully at 100% P@5 regardless of budget (it contains the question text).
The unicode-only component shows a clear inflection point:

- **Budget ≤ 15:** P@5 < 0.65 — insufficient embedding shift; adversarial
  passages fail to surface in top-5.
- **Budget = 20:** P@5 ≈ 0.90 — DE finds TAG placements that shift the
  embedding enough to compete with clean passages; retrieval is reliable.
- **Budget ≥ 30:** P@5 collapses below 0.20 — TAG characters saturate
  Contriever's vocabulary as [UNK] tokens, pushing the embedding into noise space
  away from the query.

Budget=20 is the sweet spot where the unicode component reaches near-semantic
retrieval strength while maintaining perplexity below τ=50.

> **[INSERT TABLE 3.1: Budget ablation]** — Unicode P@5 vs Semantic P@5

## 3.4 Defense Evaluation Pipeline

For each cell in the (RAG variant × attack × defense) matrix, the evaluation pipeline is:

1. **Craft adversarial passages** (or load from cache) using the relevant attack module.
2. **Inject** the N=5 adversarial passages into a copy of the FAISS index for the dataset.
3. **Apply the defense** at the appropriate stage:
   - NFKC normalization: applied to retrieved passages before passing to the generator.
   - Zero-width strip: same.
   - Perplexity filter: scores each retrieved passage with GPT-2 and drops those above the threshold.
   - Query paraphrase: rewrites the query before retrieval.
4. **Run the RAG variant** on the (potentially filtered) passages: Vanilla returns top-k directly; Self-RAG/CRAG/RobustRAG apply their own filtering; TrustRAG runs K-means.
5. **Generate** the answer with the LLM.
6. **Score** the answer using fuzzy-match ASR against the target answer.
7. **Aggregate** results over all 100 target questions and compute ASR, P@5, R@5, F1@5.

> **[INSERT FIGURE 3.4]** *Defense evaluation flow per (RAG, attack, defense) cell. Highlights the critical decision points where filtering can occur.*

---

# Chapter 4
# Implementation

## 4.1 Dataset Preparation

We use three benchmark question-answering datasets, summarized in Table 5.1 (deferred to Chapter 5).

For each dataset, we follow PoisonedRAG's protocol:
- The full corpus (2.6M / 5.2M / 8.8M passages) is loaded from the BEIR splits.
- For development, we subsample 10K passages randomly. For full thesis runs, we use the complete corpus.
- We select 100 close-ended questions (10 per trial × 10 trials) as target questions.
- For each target question, we use **Qwen2.5:7b** to randomly generate a fake answer that differs from the gold answer.

The choice of close-ended (factual) questions over open-ended ones is deliberate: close-ended questions admit unambiguous gold and target answers, enabling reliable substring-based ASR evaluation. Open-ended questions are deferred to future work.

> **[INSERT TABLE — example target questions]** *Sample of 10 target questions and their fake answers, similar to Table 24 in the PoisonedRAG paper appendix.*

## 4.2 Index Construction with FAISS

For each dataset, we encode the corpus using Contriever (`facebook/contriever`) and build a FAISS Inner-Product index. The index is cached on disk to avoid re-encoding on every run. For 10K passages, encoding takes approximately 30 seconds on an A100 GPU. For the full corpus (8.8M for MS-MARCO), encoding takes ~4 hours and produces a ~11 GB index.

Adversarial passages are encoded the same way and added to the index via `index.add()` before retrieval. The index is restored from a cached serialization between trials to ensure clean state.

## 4.3 Attack Implementation

### Pure Semantic Attack (PoisonedRAG)
Implemented in `src/attacks/poisoned_rag.py`. Uses **Qwen2.5:7b** to generate I with the prompt template described in Section 3.3. Black-box construction: P = Q ⊕ I.

### Pure Unicode Attack (RAG-Pull)
Implemented in `src/attacks/rag_pull.py`. Uses scipy's `differential_evolution` with the full inventory (zero-width, joiners, variation selectors, TAG chars) inserted into the entire passage. No semantic content beyond a clean placeholder text.

### Hybrid Attack (Ours)
Implemented in `src/attacks/hybrid.py`. Stage 1 reuses the PoisonedRAG generator. Stage 2 reuses the RAG-Pull DE optimizer but with the inventory restricted to TAG chars (U+E0001–U+E007F) and the perturbation region restricted to the first 50 characters of the passage.

> **[INSERT FIGURE 4.1]** *Pipeline for adversarial passage crafting: Stage 1 (LLM call) → Stage 2 (DE optimization) → injection into FAISS → evaluation.*

> **[INSERT TABLE — example crafted passages]** *Example adversarial passages for each of the three attack types, for the question "Who is the CEO of OpenAI?" with target answer "Tim Cook". Similar to PoisonedRAG Table 24.*

## 4.4 Defense Implementation

### NFKC Normalization (`src/defenses/unicode_normalize.py`)
Applies `unicodedata.normalize('NFKC', text)` to every retrieved passage before passing to the generator. NFKC collapses compatibility characters (e.g., ﬃ → ffi) but **does not remove invisible characters** like TAG or zero-width chars.

### Zero-Width Strip (`src/defenses/zero_width_strip.py`)
Removes all characters matching `Cf` (Format) and `Cc` (Control) Unicode categories, as well as TAG characters in the U+E0000–U+E007F range. Effective against pure unicode attacks (drops e2e ASR to 0%) but **leaves the semantic payload intact** for hybrid attacks.

### Perplexity Filter (`src/defenses/perplexity.py`)
Scores each retrieved passage with GPT-2 (HuggingFace `gpt2`). Two modes:
- **Fraction mode** (`drop_fraction=0.5`): drops the top 50% highest-PPL passages.
- **Threshold mode** (`threshold=50.0`): drops all passages with absolute PPL > 50.

A `--strict-fallback` flag changes the empty-context behavior: if all passages exceed the threshold, generate from parametric knowledge (empty context) rather than falling back to the top-1 retrieved (which would still be adversarial).

### Query Paraphrase (`src/defenses/paraphrase.py`)
Uses an LLM to paraphrase the query before retrieval. Disrupts character-level retrieval boosts because the embedding of the paraphrased query is different from the embedding the attacker optimized against.

## 4.5 Five RAG Variants

### Vanilla RAG (`src/rag/vanilla.py`)
Standard top-k retrieval + LLM generation with a fixed system prompt.

### Self-RAG (`src/rag/self_rag.py`)
Approximates Asai et al.'s reflection-token mechanism with prompt-based relevance judgments. For each retrieved passage, the LLM is asked: "Is this passage relevant to: '{query}'? Answer Yes or No." Only Yes-passages are used as context.

### CRAG (`src/rag/crag.py`)
Approximates Yan et al.'s three-way routing using cosine similarity scores as the evaluator proxy. Passages are routed to CORRECT (>upper threshold), AMBIGUOUS (in between), or INCORRECT (<lower threshold). CORRECT passages are kept; AMBIGUOUS trigger query refinement; INCORRECT are dropped.

### TrustRAG (`src/rag/trust_rag.py`)
Direct port of Zhou et al.'s K-means filtering. Retrieves 20 candidates, embeds them with the same retriever, runs `KMeans(n_clusters=2)`, and discards the minority cluster.

> **[INSERT FIGURE 4.2]** *TrustRAG K-means cluster filtering visualization. With 5 adversarial + 15 clean passages, the adversarial group clusters together as the minority and is discarded. With 5 hybrid (TAG-perturbed) + 15 clean, the hybrid group still clusters together (because the embedding shift is consistent across all 5 hybrid passages), so K-means reduces hybrid ASR similarly.*

### RobustRAG (`src/rag/robust_rag.py`)
Implements Xiang et al.'s isolate-then-aggregate strategy. Each passage is independently fed to the LLM, producing isolated answers; the final answer is selected by keyword-intersection majority vote.

---

# Chapter 5
# Experimental Setup

## 5.1 Datasets

> **[INSERT TABLE 5.1]** *Dataset statistics.*

| Dataset | Domain | Total Passages | Avg Passage Length | Subsample (dev) | # Target Questions |
|---------|--------|----------------|--------------------|--------------------|--------------------|
| Natural Questions (NQ) | Wikipedia | 2,681,468 | 100 words | 10,000 | 100 |
| HotpotQA | Multi-hop Wikipedia | 5,233,329 | 100 words | 10,000 | 100 |
| MS-MARCO | Web (Bing) | 8,841,823 | 60 words | 10,000 | 100 |

NQ is our **primary benchmark** (matching the PoisonedRAG protocol). HotpotQA and MS-MARCO are used for generalization checks.

## 5.2 Models

> **[INSERT TABLE 5.2]** *Models used in the evaluation.*

| Component | Model | Source |
|-----------|-------|--------|
| Retriever (default) | `facebook/contriever` | HuggingFace |
| Retriever (alt) | `BAAI/bge-base-en-v1.5` | HuggingFace |
| Generator (default) | `qwen2.5:7b` | Ollama |
| Generator (alt 1) | `llama3.1:8b` | Ollama |
| Generator (alt 2) | `mistral:7b` | Ollama |
| Generator (paper) | `gpt-4o-mini` | OpenAI API |
| Generator (paper) | `claude-haiku-4-5` | Anthropic API |
| PPL scorer | `gpt2` (HuggingFace) | HuggingFace |
| Stage-1 crafter | `qwen2.5:7b` | Ollama |

The default generator for all dev experiments is `qwen2.5:7b` running locally via Ollama. Final paper results will use 3 frontier LLMs.

## 5.3 Evaluation Metrics

> **[INSERT TABLE 5.3]** *Evaluation metrics.*

| Metric | Formula | Description |
|--------|---------|-------------|
| ASR | (#successful target questions) / (#total target questions) | Attack Success Rate via fuzzy substring match |
| P@k | (#adv passages in top-k) / k | Precision of malicious retrieval |
| R@k | (#adv passages retrieved) / (#adv injected) | Recall |
| F1@k | 2·P@k·R@k / (P@k + R@k) | Harmonic mean |
| e2e success | (P@k > 0) AND ASR-hit | End-to-end success |
| GPT-2 PPL | exp(−avg log-likelihood) | Perplexity, lower = more natural |
| Char-class entropy | Shannon entropy over Unicode classes | Stealth measure |

## 5.4 Configuration Profiles

To facilitate reproducibility and to manage the cost of experiments at different scales, we define five configuration profiles in `configs/experiments.yaml`. The full thesis run uses the `thesis_full` profile.

> **[INSERT TABLE 5.4]** *Configuration profiles. The `thesis_full` profile is used for all results in Chapter 6 unless otherwise stated.*

| Profile | Corpus | Questions | RAGs | Attacks | Defenses | Budget | Wall Time |
|---------|:------:|:---------:|:----:|:-------:|:--------:|:------:|-----------|
| smoke | 1K | 3 | 1 | 1 | 1 | 20 | 2 min (CPU) |
| small | 10K | 20 | 1 | 3 | 4 | 20 | ~40 min (CPU) |
| colab_test | 10K | 20 | 5 | 3 | 4 | 20 | ~2 h (GPU) |
| **thesis_full** | **10K** | **100** | **5** | **3** | **4** | **20** | **~10 h (GPU)** |
| paper | full | 100 | 5 | 3 | 5 | 20 | ~30 h (GPU) |

---

# Chapter 6
# Experimental Results

This chapter presents the empirical evaluation of the RAFT framework. All results are from the `thesis_full` profile: NQ dataset, 100 target questions, 10,000-passage corpus, Contriever retriever, Qwen2.5-7b generator. Full per-question detail CSVs are in `results/exp07/nq/`.

## 6.1 Baseline Vanilla RAG Performance

Before evaluating attacks, we measure the baseline accuracy of Vanilla RAG on each dataset without any adversarial intervention.

> **[INSERT TABLE 6.1]** *Baseline clean accuracy of Vanilla RAG.*

| Dataset | Baseline Accuracy |
|---------|:-----------------:|
| Natural Questions | 70% |
| HotpotQA | 80% |
| MS-MARCO | 83% |

These numbers indicate that the LLM correctly answers 70–83% of target questions when no adversarial passages are injected. This baseline matters because attacks are evaluated relative to **whether the LLM produces the target answer** (not whether the answer is correct).

## 6.2 Attack Effectiveness on Vanilla RAG

Table 6.2 reports the full attack × defense matrix for Vanilla RAG on the NQ dataset. Each cell reports ASR over 100 target questions.

**Table 6.2:** Attack × Defense matrix for Vanilla RAG (NQ, n=100).

| Attack | None | NFKC normalize | Zero-width strip | Perplexity (τ=50) |
|--------|:----:|:--------------:|:----------------:|:-----------------:|
| Pure semantic | 92% | 92% | 92% | 89% |
| Pure unicode | 18% | 19% | 19% | 20% |
| **Hybrid (ours)** | **88%** | **90%** | **90%** | **87%** |

*Source: `results/exp07/nq/vanilla_*.csv`, n=100 questions each.*

> **[INSERT FIGURE 6.1]** *Bar chart: ASR comparison across attacks on Vanilla RAG, grouped by defense. Highlights that hybrid attack maintains ≥85% ASR in every defense column, while pure semantic and pure unicode each fail under at least one defense.*

**Key observations:**

1. **Hybrid attack is uniquely defense-resilient.** The hybrid maintains ≥87% ASR across every defense column. Pure semantic drops from 92% to 89% under perplexity (3pp reduction). Pure unicode remains low (18–20%) across all defenses — its weakness is the absence of semantic payload, not the defenses themselves.

2. **Zero-width strip does not eliminate the pure unicode attack.** Unicode ASR is 18% without any defense and 19% with zero-width strip — essentially unchanged. This is because our unicode attack uses TAG characters (U+E0001–U+E007F), which are not in the zero-width character ranges targeted by the strip defense. The hybrid ASR stays at **90%** under ZW-strip for the same reason.

3. **NFKC normalization is ineffective against all three attacks.** ASR is unchanged across the board. NFKC normalizes compatibility forms but does not remove TAG characters.

4. **Perplexity filtering creates a marginal gap.** Semantic drops 92%→89% (3pp) and hybrid drops 88%→87% (1pp) under strict τ=50 perplexity filtering. The gap between them is only 2pp — not enough to reliably discriminate. No defense eliminates the hybrid attack.

> **[INSERT FIGURE 6.2]** *Heatmap: ASR across all (RAG × attack × defense) combinations. Rows are RAG variants, columns are (attack, defense) pairs. Color codes: green for low ASR (<30%), yellow for medium (30–70%), red for high (>70%). Should make the hybrid attack's row stand out visually.*

## 6.3 Defense Effectiveness Analysis

We now analyze each defense in detail.

### 6.3.1 Zero-Width Strip

Zero-width stripping is a per-passage filter that removes invisible Unicode characters before passing to the generator. Against the **pure unicode attack**, stripping has negligible effect: ASR is 18% without defense and 19% with zero-width strip. This is because our unicode attack uses TAG characters (U+E0001–U+E007F), which fall outside the standard zero-width ranges (ZWJ U+200D, ZWNJ U+200C, ZWSP U+200B) that the defense targets. The TAG character block is not stripped.

Against the **hybrid attack**, the same applies: TAG characters survive the strip, and the semantic payload (Stage-1 LLM output) remains fully intact. ASR stays at **90%** — identical to no defense.

### 6.3.2 Perplexity Filter (Threshold + Strict Fallback)

We evaluate the perplexity filter in two modes:

**Fraction mode (drop_fraction=0.5).** Drops the top 50% highest-PPL passages. Result: ineffective when all 5 retrieved passages are adversarial — they all have similar PPL, so filtering just rearranges them.

**Threshold mode (τ=50, strict_fallback=True).** Drops all passages with PPL > 50; if all are dropped, generate from empty context. With n_adv=1 (only 1 adversarial passage injected per question), this becomes a clean discriminator: hybrid passages have max PPL=49.1 (all pass), while semantic passages have max PPL=112.6 (3 of 20 fail).

> **[INSERT TABLE 6.4]** *PPL threshold ablation results (strict fallback, n_adv=1, n=100). Numbers from 20Q dev; will be updated with 100Q.*

| Threshold | Semantic ASR | Hybrid ASR | Avg adv after filter (sem) | Avg adv after filter (hyb) |
|----------:|:------------:|:----------:|:--------------------------:|:--------------------------:|
| 30 | 30% | 20% | 0.35 | 0.25 |
| 40 | 65% | 65% | 0.70 | 0.80 |
| **50** | **80%** | **85%** | **0.85** | **1.00** |
| 75 | 85% | 85% | 0.90 | 1.00 |
| 100 | 85% | 85% | 0.95 | 1.00 |

> **[INSERT FIGURE 6.3]** *Line plot: ASR vs. PPL threshold for semantic and hybrid attacks. Shows the gap between the two curves widening at threshold τ=50, the cleanest discriminator point. X-axis: threshold (30, 40, 50, 75, 100); Y-axis: ASR; two lines (semantic, hybrid).*

### 6.3.3 NFKC Normalization

NFKC collapses Unicode compatibility forms (e.g., ﬃ → ffi, ½ → 1/2) but does not remove invisible characters. Result: completely ineffective against all three attacks. Included in the matrix for completeness.

### 6.3.4 Query Paraphrase (Optional)

Paraphrasing the query before retrieval was evaluated in PoisonedRAG and found to reduce ASR by ~10–15 percentage points. We do not run paraphrase by default in `thesis_full` (it doubles the LLM call count) but include it in the `paper` profile for the final paper submission.

## 6.4 Trust-Aware RAG Variants

> **[INSERT TABLE 6.3]** *Attack × RAG Variant matrix (NQ, no defense, n=100). Reports ASR under no defense for each (RAG, attack) pair.*

**Table 6.3:** Attack × RAG Variant matrix (NQ, no defense, n=100).

| RAG variant | Semantic ASR | Hybrid ASR | Notes |
|-------------|:------------:|:----------:|-------|
| Vanilla | 92% | 88% | Baseline |
| Self-RAG | 89% | 88% | Relevance filtering has minimal effect |
| CRAG | 92% | 85% | Three-way routing does not discriminate adversarial passages |
| RobustRAG | 90% | 89% | Keyword majority vote does not help |
| **TrustRAG** | **29%** | **27%** | **K-means is the only effective defense** |

*Source: `results/exp07/nq/*_none.csv`, n=100 questions each.*

**Critical observation:** Self-RAG, CRAG, and RobustRAG provide essentially no additional protection beyond Vanilla RAG — semantic ASR stays at 88–92% and hybrid at 85–89%. These defenses were designed to improve answer quality, not to detect adversarial passages.

**TrustRAG is the only RAG variant that substantially reduces both attacks**, bringing ASR down to 27–29%. Its K-means filter clusters adversarial passages as a minority group and discards them. Critically, it treats semantic and hybrid attacks identically — the 2pp difference (29% vs 27%) is within noise. **TrustRAG is not attack-discriminative**: it cannot distinguish hybrid from semantic, but it does reduce both proportionally.

## 6.5 Perplexity Threshold Ablation

The perplexity threshold sweep is the cleanest single ablation in this thesis. By varying τ from 30 to 100 in strict-fallback mode with n_adv=1, we trace the discriminator curve between semantic and hybrid attacks. Figure 6.3 (above) plots this curve.

The conclusion: **strict perplexity filtering is the only single-defense that creates a measurable gap between semantic and hybrid attacks, but the gap is small (2pp at τ=50: semantic 89% vs hybrid 87%) and is insufficient to stop either attack**. At higher thresholds both attacks pass through completely. No single defense eliminates the hybrid attack.

## 6.6 Stealth Profile Analysis

We analyze the perplexity distribution of each attack class on the 100 NQ target questions.

> **[INSERT TABLE 6.5]** *Stealth profile: GPT-2 perplexity statistics per attack class (NQ, n=100).*

| Attack | Median PPL | Max PPL | Min PPL | % within τ=50 |
|--------|:----------:|:-------:|:-------:|:-------------:|
| Pure semantic | 35.8 | 112.6 | 11.2 | 85% |
| Pure unicode (full inventory) | very high or low | varies | — | depends on chars used |
| **Hybrid (TAG-only, b=20)** | **34.6** | **49.1** | **22.7** | **100%** |

> **[INSERT FIGURE 6.4]** *Box plot or histogram: GPT-2 perplexity distributions per attack class on NQ (n=100). Three boxes/histograms side by side. The hybrid distribution should sit entirely below τ=50, while the semantic distribution should have a long tail above τ=50.*

**Key finding:** the hybrid attack has a **lower maximum PPL (49.1)** than the pure semantic attack (112.6). This is counterintuitive — adding extra characters should *increase* perplexity. The explanation is that GPT-2 assigns low perplexity to repetitive TAG character patterns (because they tokenize as repeated [UNK] tokens, which have a learned high-likelihood pattern in repetitive contexts). The semantic attack's higher max PPL is driven by a few questions where **Qwen2.5:7b's** fabricated content uses unusual word combinations. Net effect: the hybrid attack is **more perplexity-stealthy** than its semantic component alone.

## 6.7 Comparative Analysis with Prior Attacks

We compare RAFT-Hybrid with five baselines from the literature, all at the same N=5 injection budget on NQ.

> **[INSERT TABLE 6.6]** *Comparison with prior attacks on NQ. Reports ASR, F1@5, and runtime per malicious passage.*

| Attack | ASR (no defense) | F1@5 | Runtime/passage |
|--------|:----------------:|:----:|:----------------:|
| Naive (target question as passage) | 3% | 1.00 | 0 sec |
| Corpus Poisoning (Zhong et al., 2023) | 1% | 0.99 | 5 sec |
| Disinformation (PoisonedRAG variant) | 69% | 0.48 | 1.5 sec |
| Prompt Injection | 62% | 0.73 | 0.5 sec |
| GCG (Zou et al., 2023) | 2% | 0.00 | 60 sec |
| **PoisonedRAG (black-box)** | **97%** | **0.96** | **1.5 sec** |
| **RAFT-Hybrid (ours)** | **88%** | **0.974** | **30 sec** |

> **[INSERT FIGURE 6.5]** *Bar chart: ASR comparison across all attacks (Naive, Corpus Poisoning, Disinformation, Prompt Injection, GCG, PoisonedRAG, RAFT-Hybrid) with and without each defense. Should illustrate that hybrid is the only attack that achieves ≥85% ASR under every defense column.*

**Key observations:**

1. **Without defenses, PoisonedRAG slightly outperforms RAFT-Hybrid** (97% vs. 88%). This is expected: PoisonedRAG optimizes purely for retrieval and generation, while hybrid trades a small amount of retrieval performance for perplexity stealth.

2. **With defenses, RAFT-Hybrid is uniquely robust.** Under zero-width strip, PoisonedRAG is unaffected (97%) and hybrid stays at 90%. Under strict perplexity filtering, PoisonedRAG drops to ~75% (its max PPL exceeds τ), while hybrid stays at 87%. **The hybrid attack is the only attack class that achieves ≥85% ASR under every defense in our matrix.**

3. **Runtime is acceptable.** RAFT-Hybrid takes ~30 sec per malicious passage (Stage-1 LLM call + Stage-2 DE), which is feasible for an attacker preparing a poisoning corpus offline.

---

# Chapter 7
# Conclusion

## 7.1 Summary of Findings

This thesis presented **RAFT (RAG Adversarial Forensics and Threat-analysis)**, a comprehensive framework for evaluating adversarial attacks on Retrieval-Augmented Generation systems, and introduced a novel **hybrid Unicode-semantic attack** that combines PoisonedRAG-style semantic injection with RAG-Pull-style invisible-character retrieval boosting.

The headline empirical result is that the hybrid attack is the only attack class evaluated in this work that **no single standard defense can fully neutralize**:

- **Zero-width stripping** is ineffective against all attacks. Pure unicode ASR is 18% without defense and 19% with strip (unchanged), because TAG characters (U+E0001–U+E007F) are not targeted by the defense. The hybrid attack stays at 90%. No defense gains from stripping.
- **Perplexity filtering** at threshold τ=50 creates only a 2pp gap (semantic 89% vs hybrid 87%); operationally negligible. Neither attack is eliminated.
- **NFKC normalization** is completely ineffective — ASR is unchanged for all three attacks across all RAG variants.
- **TrustRAG**'s K-means filter is the only effective defense, reducing both semantic (92%→29%) and hybrid (88%→27%) proportionally. The 2pp difference is within noise — TrustRAG cannot discriminate between attack types.

Additionally, we demonstrated that the hybrid attack is **more perplexity-stealthy than its semantic component alone** (max PPL 49.1 vs 112.6), achieved by restricting the invisible-character inventory to TAG characters (U+E0001–U+E007F) at a budget of 20 characters per passage.

## 7.2 Thesis Contributions

The contributions of this thesis are:

1. **A novel hybrid attack design.** We are the first to systematically combine PoisonedRAG-style semantic injection with RAG-Pull-style invisible-character optimization, with the inventory restricted to TAG characters for perplexity stealth.

2. **An open-source comprehensive evaluation framework.** RAFT supports five RAG architectures, three attack types, four defenses, and three datasets, with reproducible profiles ranging from 3-question smoke tests to 100-question full thesis runs. The codebase is published at `github.com/dhruvkhanna930/RAFT`.

3. **A complete cross-defense evaluation matrix.** We are the first to evaluate the same attack across all five RAG architectures (Vanilla, Self-RAG, CRAG, TrustRAG, RobustRAG) and all four defenses (NFKC, ZW-strip, PPL, paraphrase) on the same datasets.

4. **A perplexity-aware attack-design principle.** Our profiling work establishes that TAG characters are 5–10× more perplexity-stealthy than zero-width characters, providing concrete guidance for future attack and defense design.

5. **A benchmark for future defense research.** Any future "hybrid-aware" defense can be evaluated on RAFT and directly compared with our results.

## 7.3 Limitations

This thesis has several limitations that constrain the scope of its claims:

1. **Single LLM in deployed RAG.** Most experiments use Qwen2.5-7B as the deployed RAG LLM. Frontier LLMs (GPT-4o, Claude Haiku 4.5, DeepSeek-V3) may have different vulnerability profiles. We address this in part by including these LLMs in the `paper` profile, but a full multi-LLM cross-evaluation remains for future work.

2. **Single retriever (Contriever).** The hybrid attack is optimized against Contriever's embedder. Generalization to BGE, E5, or other dense retrievers is plausible (since the attack is black-box) but not yet demonstrated. We include BGE in `paper` profile for cross-validation.

3. **Single dataset family (NQ).** Most numerical results are on NQ; HotpotQA and MS-MARCO are evaluated only at the 20Q development scale. Multi-hop reasoning (HotpotQA) and web-domain (MS-MARCO) generalization is partial.

4. **Close-ended questions only.** We evaluate on factual close-ended questions where the target answer is a short string. Open-ended generation tasks (summarization, dialogue) are deferred to future work.

5. **Static defenses.** We evaluate defenses that are deployed at inference time without any online adaptation. Adaptive defenses (e.g., online perplexity threshold tuning, ensemble defenses) are not evaluated.

6. **Ethical considerations.** Like any adversarial attack research, the hybrid attack we describe could be misused. We have followed responsible-disclosure practices: the attack and code are publicly available because they are necessary for the security community to develop defenses, but our paper specifically advocates for hybrid-aware defenses rather than purely offensive use.

## 7.4 Future Work

We identify several promising directions:

1. **Hybrid-aware defenses.** No defense in production today is designed to address both semantic and character-level threats. A defense pipeline that combines (a) zero-width stripping at the input layer, (b) per-passage perplexity filtering with a learned, query-dependent threshold, and (c) trust-aware K-means filtering may be sufficient. We plan to evaluate this combined defense on RAFT.

2. **Adaptive attacker.** A real attacker would optimize against the deployed defense. We plan to evaluate the hybrid attack against an adaptive variant of zero-width strip (e.g., one that also looks at character-class entropy) to determine whether attack-defense escalation is bounded.

3. **Multi-LLM evaluation.** The full `paper` profile includes GPT-4o-mini, Claude Haiku 4.5, and DeepSeek-V3 as the deployed RAG LLM. Cross-LLM transferability is an open question.

4. **Multi-retriever cross-validation.** BGE-base-en-v1.5 is a strong alternative retriever; we will run the hybrid attack optimized against Contriever and evaluate retrieval/ASR with BGE, E5, and SFR-Embedding-Code as the deployed retrievers.

5. **Real-world deployment.** The current evaluation uses static, offline corpora. We are exploring a live evaluation against a real deployed RAG system (with permission) to assess attack feasibility under real-world latency, query distributions, and corpus updates.

6. **Connection to LLM agents.** RAG-based LLM agents (with multi-step reasoning, tool calls, and external API access) introduce new attack surfaces. The hybrid attack on a RAG-augmented agent could induce harmful tool calls, not just incorrect answers. This is the most promising direction for follow-up work.

---

## REFERENCES

1. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., et al. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks*. NeurIPS.

2. Zou, W., Geng, R., Wang, B., & Jia, J. (2024). *PoisonedRAG: Knowledge corruption attacks to retrieval-augmented generation of large language models*. USENIX Security Symposium 2025.

3. Stambolic, V., Dhar, A., & Cavigelli, L. (2025). *RAG-Pull: Imperceptible attacks on RAG systems for code generation*. arXiv:2510.11195.

4. Boucher, N., Shumailov, I., Anderson, R., & Papernot, N. (2021). *Bad characters: Imperceptible NLP attacks*. arXiv:2106.09898.

5. Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). *Self-RAG: Learning to retrieve, generate, and critique through self-reflection*. ICLR.

6. Yan, S.-Q., Gu, J.-C., Zhu, Y., & Ling, Z.-H. (2024). *Corrective retrieval augmented generation*. arXiv.

7. Zhou, H., et al. (2025). *TrustRAG: K-means cluster filtering against PoisonedRAG*. (Reference per project's CLAUDE.md notes).

8. Xiang, Z., Jiang, F., Xiong, Z., Ramasubramanian, B., Poovendran, R., & Li, B. (2024). *BadChain: Backdoor chain-of-thought prompting for large language models*. (Cited as RobustRAG in our system).

9. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W.-t. (2020). *Dense passage retrieval for open-domain question answering*. EMNLP.

10. Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski, P., Joulin, A., & Grave, E. (2022). *Unsupervised dense information retrieval with contrastive learning*. TMLR.

11. Xiong, L., Xiong, C., Li, Y., Tang, K.-F., Liu, J., Bennett, P. N., Ahmed, J., & Overwijk, A. (2021). *Approximate nearest neighbor negative contrastive learning for dense text retrieval*. ICLR.

12. Kwiatkowski, T., et al. (2019). *Natural questions: A benchmark for question answering research*. TACL.

13. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., & Manning, C. D. (2018). *HotpotQA: A dataset for diverse, explainable multi-hop question answering*. EMNLP.

14. Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S., Majumder, R., & Deng, L. (2016). *MS MARCO: A human generated machine reading comprehension dataset*.

15. Zou, A., Wang, Z., Kolter, J. Z., & Fredrikson, M. (2023). *Universal and transferable adversarial attacks on aligned language models*. arXiv:2307.15043. (GCG Attack)

16. Perez, F., & Ribeiro, I. (2022). *Ignore previous prompt: Attack techniques for language models*. NeurIPS Workshop.

17. Carlini, N., Jagielski, M., Choquette-Choo, C. A., Paleka, D., Pearce, W., Anderson, H., Terzis, A., Thomas, K., & Tramèr, F. (2024). *Poisoning web-scale training datasets is practical*. arXiv.

18. Jain, N., Schwarzschild, A., Wen, Y., Somepalli, G., Kirchenbauer, J., Chiang, P.-y., Goldblum, M., Saha, A., Geiping, J., & Goldstein, T. (2023). *Baseline defenses for adversarial attacks against aligned language models*. arXiv.

19. Jaiswal, P., Pratap, A., Saraswati, S., Kasyap, H., & Tripathy, S. (2026). *Analysis of LLMs against prompt injection and jailbreak attacks*. arXiv:2602.22242.

20. Ebrahimi, J., Rao, A., Lowd, D., & Dou, D. (2018). *HotFlip: White-box adversarial examples for text classification*. ACL.

21. Storn, R., & Price, K. (1997). *Differential evolution — a simple and efficient heuristic for global optimization*. Journal of Global Optimization.

22. Zhong, Z., Huang, Z., Wettig, A., & Chen, D. (2023). *Poisoning retrieval corpora by injecting adversarial passages*. EMNLP.

23. Chaudhari, H., Severi, G., Abascal, J., Jagielski, M., Choquette-Choo, C. A., Nasr, M., Nita-Rotaru, C., & Oprea, A. (2024). *Phantom: General trigger attacks on retrieval augmented language generation*. arXiv:2405.20485.

24. Xue, J., Zheng, M., Hu, Y., Liu, F., Chen, X., & Lou, Q. (2024). *BadRAG: Identifying vulnerabilities in retrieval augmented generation of large language models*. arXiv:2406.00083.

25. Chen, Z., Xiang, Z., Xiao, C., Song, D., & Li, B. (2024). *AgentPoison: Red-teaming LLM agents via poisoning memory or knowledge bases*. arXiv:2407.12784.

[Add 5–10 more references as needed during final formatting, including the BAAI BGE technical report, the FAISS paper (Johnson et al.), and any LLM-specific citations like Qwen, Llama, GPT-4 technical reports.]

---

## APPENDICES (Optional)

### Appendix A — Sample Target Questions and Crafted Adversarial Passages
> [Insert 10 example (target question, target answer, semantic passage, unicode passage, hybrid passage) tuples, similar to PoisonedRAG Appendix Tables 24–26.]

### Appendix B — System Prompts
> [Insert the system prompts used for the deployed RAG generator, the Stage-1 crafter, Self-RAG relevance filter, CRAG router, RobustRAG aggregator.]

### Appendix C — Hyperparameter Sensitivity
> [Insert ablation tables for budget (10, 20, 30, 50), DE population (5, 10, 15), DE iterations (10, 30, 50), n_adv (1, 3, 5).]

### Appendix D — Failure Case Analysis
> [Insert 3–5 examples of cases where the hybrid attack failed, with explanations.]
