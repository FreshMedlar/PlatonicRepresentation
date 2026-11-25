"""
Clean Multi-Sentence Corpus Loader for Autoregressive Predictive JEPA.

Loads:
1. Clean consecutive multi-sentence prose documents from `HuggingFaceFW/fineweb-edu` (streamed)
   plus procedural guides from `b-mc2/wikihow_lists` and `soyasis/wikihow_small`.
   Produces sliding contiguous sentence chains [s_i, s_{i+1}, ..., s_{i+T-1}] with zero
   contradictory target pairs and clean train/test document splits.
2. Clean Good-View paraphrase anchors from `community-datasets/tapaco` (and optional clean
   `Recap-DataComp` subset) for semantic invariance anchoring during Stage 1 and joint training.
"""

import glob
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import pyarrow.parquet as pq


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def split_clean_sentences(text: str, min_len: int = 20, max_len: int = 220) -> List[str]:
    """
    Splits an article/document into clean, well-formed natural-language sentences.
    Filters out boilerplate, URLs, code snippets, and fragment headings.
    """
    sentences: List[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if len(para) < min_len:
            continue
        for sent in _SENT_SPLIT_RE.split(para):
            sent = sent.strip()
            if not (min_len <= len(sent) <= max_len):
                continue
            if "http://" in sent or "https://" in sent or "{" in sent or "}" in sent:
                continue
            alpha_ratio = sum(c.isalpha() for c in sent) / max(len(sent), 1)
            if alpha_ratio < 0.65:
                continue
            sentences.append(sent)
    return sentences


def split_clean_book_sentences(text: str, min_len: int = 25, max_len: int = 240) -> List[str]:
    """
    Cleans Project Gutenberg / PG-19 book text by skipping header/footer boilerplate,
    unwrapping single-newline hard wraps within paragraphs, and extracting clean prose sentences.
    """
    n = len(text)
    if n > 20000:
        text = text[int(n * 0.08) : int(n * 0.88)]
    text = text.replace("\r\n", "\n")
    # Rejoin hard-wrapped lines inside paragraphs while preserving paragraph breaks (\n\n+)
    paras = re.split(r"\n\s*\n+", text)
    sentences: List[str] = []
    bad_markers = ("gutenberg", "copyright", "ebook", "etext", "chapter ", "http://", "https://", "www.", "***", "<<", ">>", "_")
    for para in paras:
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) < min_len:
            continue
        low = para.lower()
        if any(m in low for m in bad_markers):
            continue
        for sent in _SENT_SPLIT_RE.split(para):
            sent = sent.strip()
            if not (min_len <= len(sent) <= max_len):
                continue
            if sent[-1] not in ".!?":
                continue
            # Skip verse numbering or all-caps headings
            if re.match(r"^\d+[:.]\d+\s", sent):
                continue
            alpha_cnt = sum(c.isalpha() for c in sent)
            upper_cnt = sum(c.isupper() for c in sent)
            if alpha_cnt / max(len(sent), 1) < 0.72:
                continue
            if upper_cnt / max(alpha_cnt, 1) > 0.35:
                continue
            sentences.append(sent)
    return sentences


class AutoregressiveCorpusLoader:
    """
    Provides:
    - Contiguous multi-sentence document chains for autoregressive latent prediction (Stages 2 & Joint)
    - Clean Good-View paraphrase pairs for semantic invariance (Stage 1 & Joint Anchor)
    - Clean prose + procedural sentences for Conditional Decoder grounding (Stage 4)
    """

    def __init__(
        self,
        max_fineweb_docs: int = 4000,
        max_tapaco_clusters: int = 6000,
        max_wikihow_guides: int = 4500,
        max_recap_anchors: int = 10000,
        max_books: int = 0,
        use_clean_paraphrases: bool = False,
        min_doc_sentences: int = 4,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.max_fineweb_docs = max_fineweb_docs
        self.max_tapaco_clusters = max_tapaco_clusters
        self.max_wikihow_guides = max_wikihow_guides
        self.max_recap_anchors = max_recap_anchors
        self.max_books = max_books
        self.use_clean_paraphrases = use_clean_paraphrases
        self.min_doc_sentences = min_doc_sentences
        self.test_ratio = test_ratio

        self.train_documents: List[List[str]] = []
        self.test_documents: List[List[str]] = []
        self.book_train_documents: List[List[str]] = []
        self.book_test_documents: List[List[str]] = []
        self.wikihow_test_guides: List[Tuple[str, List[str]]] = []
        self.tapaco_clusters: Dict[str, List[str]] = {}
        self.extra_paraphrase_pairs: List[Tuple[str, str]] = []
        self.recap_pairs: List[Tuple[str, str]] = []

        self._load_all()

    def _load_all(self):
        from datasets import load_dataset

        # 1. Clean Good-View Paraphrase Anchors (TaPaCo)
        print("[*] Loading TaPaCo paraphrase clusters for Good-View semantic invariance...")
        tapaco_raw = load_dataset("community-datasets/tapaco", "en", split="train")
        raw_clusters = defaultdict(list)
        min_tp_len = 18 if self.use_clean_paraphrases else 10
        for row in tapaco_raw:
            t = row["paraphrase"].strip()
            if min_tp_len <= len(t) <= 220:
                raw_clusters[row["paraphrase_set_id"]].append(t)
        valid_keys = [k for k, v in raw_clusters.items() if len(v) >= 2]
        self.rng.shuffle(valid_keys)
        for k in valid_keys[: self.max_tapaco_clusters]:
            self.tapaco_clusters[str(k)] = raw_clusters[k]
        print(f"[+] Loaded {len(self.tapaco_clusters)} TaPaCo Good-View clusters.")

        if self.use_clean_paraphrases:
            try:
                paws = load_dataset("google-research-datasets/paws", "labeled_final", split="train")
                paws_pairs = [
                    (r["sentence1"].strip(), r["sentence2"].strip())
                    for r in paws
                    if r["label"] == 1 and 20 <= len(r["sentence1"].strip()) <= 240 and 20 <= len(r["sentence2"].strip()) <= 240
                ]
                self.extra_paraphrase_pairs.extend(paws_pairs)
                mrpc = load_dataset("nyu-mll/glue", "mrpc", split="train")
                mrpc_pairs = [
                    (r["sentence1"].strip(), r["sentence2"].strip())
                    for r in mrpc
                    if r["label"] == 1 and 20 <= len(r["sentence1"].strip()) <= 240 and 20 <= len(r["sentence2"].strip()) <= 240
                ]
                self.extra_paraphrase_pairs.extend(mrpc_pairs)
                qqp = load_dataset("nyu-mll/glue", "qqp", split="train")
                qqp_pairs = [
                    (r["question1"].strip(), r["question2"].strip())
                    for r in qqp
                    if r["label"] == 1 and 20 <= len(r["question1"].strip()) <= 220 and 20 <= len(r["question2"].strip()) <= 220
                ]
                self.rng.shuffle(qqp_pairs)
                self.extra_paraphrase_pairs.extend(qqp_pairs[:25000])
                print(
                    f"[+] Loaded {len(self.extra_paraphrase_pairs)} clean paraphrase pairs "
                    f"(PAWS={len(paws_pairs)}, MRPC={len(mrpc_pairs)}, QQP={min(25000, len(qqp_pairs))})."
                )
            except Exception as e:
                print(f"[!] Warning: extra paraphrase loading failed ({e})")

        # Optional Recap-DataComp subset strictly for Stage 1 good-view anchoring (NEVER fed to Decoder)
        recap_files = sorted(glob.glob("data/recap_datacomp/data/train_data/*.parquet"))
        if recap_files and self.max_recap_anchors > 0:
            for rf in recap_files:
                table = pq.read_table(rf, columns=["org_caption", "re_caption"])
                df = table.to_pandas()
                for _, r in df.iterrows():
                    o = str(r["org_caption"]).strip()
                    rc = str(r["re_caption"]).strip()
                    if 15 <= len(o) <= 220 and 15 <= len(rc) <= 220:
                        self.recap_pairs.append((o, rc))
                    if len(self.recap_pairs) >= self.max_recap_anchors:
                        break
                if len(self.recap_pairs) >= self.max_recap_anchors:
                    break
            print(f"[+] Loaded {len(self.recap_pairs)} Recap-DataComp Good-View pairs (encoder anchor only).")

        # 2. Procedural Multi-Sentence Sequences (WikiHow)
        print("[*] Loading WikiHow procedural sequences...")
        wikihow_docs: List[Tuple[str, List[str]]] = []
        wh_raw = load_dataset("b-mc2/wikihow_lists", split="train")
        for row in wh_raw:
            if row.get("result_type") not in ["stepped_summary", "step"]:
                continue
            title = row.get("title", "").strip()
            lines = [l.strip() for l in row.get("result", "").split("\n") if l.strip()]
            steps = []
            for l in lines:
                c = re.sub(r"^\d+\.\s*", "", l).strip()
                c = re.sub(r"^[•\-\*]\s*", "", c).strip()
                if 10 <= len(c) <= 200:
                    steps.append(c)
            if title and len(steps) >= 3:
                wikihow_docs.append((title, steps))

        try:
            small_raw = load_dataset("soyasis/wikihow_small", split="train")
            for row in small_raw:
                d = row.get("data", {})
                task = d.get("task", "").strip()
                steps = [s.strip() for s in d.get("summary", "").split("\n") if len(s.strip()) > 10]
                if task and len(steps) >= 3:
                    wikihow_docs.append((task, steps))
        except Exception as e:
            print(f"[!] Warning: could not load soyasis/wikihow_small: {e}")

        self.rng.shuffle(wikihow_docs)
        wikihow_docs = wikihow_docs[: self.max_wikihow_guides]
        n_wh_test = max(100, int(len(wikihow_docs) * self.test_ratio))
        wh_train = wikihow_docs[:-n_wh_test]
        self.wikihow_test_guides = wikihow_docs[-n_wh_test:]

        all_train_docs: List[List[str]] = [[title] + steps for title, steps in wh_train]
        all_test_docs: List[List[str]] = [[title] + steps for title, steps in self.wikihow_test_guides]

        # 3. Clean Internet Prose Documents (FineWeb-Edu Streaming)
        if self.max_fineweb_docs > 0:
            print(f"[*] Streaming up to {self.max_fineweb_docs} clean educational documents from HuggingFaceFW/fineweb-edu...")
            try:
                fw_stream = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name="sample-10BT",
                    split="train",
                    streaming=True,
                )
                fw_docs: List[List[str]] = []
                for row in fw_stream:
                    sents = split_clean_sentences(row.get("text", ""))
                    if len(sents) >= self.min_doc_sentences:
                        fw_docs.append(sents)
                    if len(fw_docs) >= self.max_fineweb_docs:
                        break
                self.rng.shuffle(fw_docs)
                n_fw_test = max(50, int(len(fw_docs) * self.test_ratio))
                all_train_docs.extend(fw_docs[:-n_fw_test])
                all_test_docs.extend(fw_docs[-n_fw_test:])
                print(f"[+] Streamed {len(fw_docs)} multi-sentence FineWeb-Edu documents.")
            except Exception as e:
                print(f"[!] Warning: FineWeb-Edu streaming failed ({e}); continuing with WikiHow documents.")

        # 4. Clean Book Narrative Passages (PG-19 Streaming)
        if self.max_books > 0:
            print(f"[*] Streaming up to {self.max_books} classic books from emozilla/pg19...")
            try:
                pg_stream = load_dataset("emozilla/pg19", split="train", streaming=True)
                books_list: List[List[List[str]]] = []
                for row in pg_stream:
                    raw_txt = row.get("text", "")
                    if len(raw_txt) < 60000:
                        continue
                    sents = split_clean_book_sentences(raw_txt)
                    if len(sents) < 40:
                        continue
                    per_book: List[List[str]] = []
                    for start in range(0, len(sents) - 8, 24):
                        chunk = sents[start : start + 24]
                        if len(chunk) >= 8:
                            per_book.append(chunk)
                        if len(per_book) >= 40:
                            break
                    if per_book:
                        books_list.append(per_book)
                    if len(books_list) >= self.max_books:
                        break
                self.rng.shuffle(books_list)
                n_bk_test = max(1, int(len(books_list) * self.test_ratio)) if len(books_list) > 1 else 0
                train_books = books_list[:-n_bk_test] if n_bk_test > 0 else books_list
                test_books = books_list[-n_bk_test:] if n_bk_test > 0 else []
                self.book_train_documents = [c for bk in train_books for c in bk]
                self.book_test_documents = [c for bk in test_books for c in bk]
                all_train_docs.extend(self.book_train_documents)
                all_test_docs.extend(self.book_test_documents)
                print(
                    f"[+] Loaded {len(books_list)} PG-19 books -> {len(self.book_train_documents)} train / "
                    f"{len(self.book_test_documents)} test narrative passages (book-level split)."
                )
            except Exception as e:
                print(f"[!] Warning: PG-19 book streaming failed ({e}); continuing without books.")

        self.rng.shuffle(all_train_docs)
        self.train_documents = all_train_docs
        self.test_documents = all_test_docs
        print(f"[+] Corpus ready: {len(self.train_documents)} train docs, {len(self.test_documents)} test docs.")

    def sample_good_view_pair(self, tapaco_prob: float = 0.7) -> Tuple[str, str]:
        """Samples a semantic paraphrase/good-view pair (defaults mostly to clean TaPaCo/PAWS/QQP/MRPC)."""
        if self.extra_paraphrase_pairs and self.rng.random() < 0.55:
            return self.rng.choice(self.extra_paraphrase_pairs)
        use_tapaco = (not self.recap_pairs) or (self.rng.random() < tapaco_prob)
        if use_tapaco and self.tapaco_clusters:
            cid = self.rng.choice(list(self.tapaco_clusters.keys()))
            views = self.tapaco_clusters[cid]
            if len(views) >= 2:
                for _ in range(4):
                    a, b = self.rng.sample(views, 2)
                    if min(len(a), len(b)) / max(len(a), len(b), 1) >= 0.6:
                        return a, b
                return a, b
        if self.recap_pairs:
            return self.rng.choice(self.recap_pairs)
        return self.rng.choice(self.extra_paraphrase_pairs)

    def build_sliding_windows(
        self,
        split: str = "train",
        window_size: int = 8,
        stride: int = 3,
        max_windows: int = 15000,
    ) -> List[List[str]]:
        """
        Slices contiguous sentence chains [s_i, s_{i+1}, ..., s_{i+window_size-1}] from documents.
        Every window preserves true causal/narrative ordering with no contradictory shifts.
        """
        docs = self.test_documents if split == "test" else self.train_documents
        windows: List[List[str]] = []
        for doc in docs:
            if len(doc) < 2:
                continue
            if len(doc) <= window_size:
                windows.append(doc)
            else:
                for start in range(0, len(doc) - 1, stride):
                    win = doc[start : start + window_size]
                    if len(win) >= 3:
                        windows.append(win)
                    if start + window_size >= len(doc):
                        break
        self.rng.shuffle(windows)
        if max_windows and len(windows) > max_windows:
            windows = windows[:max_windows]
        return windows

    def get_clean_decoder_sentences(self, max_sentences: int = 60000) -> List[str]:
        """
        Returns strictly clean prose + procedural + TaPaCo sentences for Stage 4 Decoder training.
        Excludes noisy alt-text captions so the decoder never memorizes web metadata.
        """
        sents: List[str] = []
        for doc in self.train_documents:
            sents.extend(doc)
        for views in self.tapaco_clusters.values():
            sents.extend(views)
        # Deduplicate while preserving order/diversity
        sents = list(dict.fromkeys(sents))
        self.rng.shuffle(sents)
        return sents[:max_sentences]

    def get_tokenizer_corpus(self, max_texts: int = 12000) -> List[str]:
        """Gathers balanced clean sentences for subword BPE training."""
        return self.get_clean_decoder_sentences(max_sentences=max_texts)
