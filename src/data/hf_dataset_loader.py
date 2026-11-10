"""
Hugging Face Dataset Loader for Good-View Text JEPA.
Provides real-world datasets:
- TaPaCo (The Tatoeba Paraphrase Corpus) for Good Views (Stage 1 / Invariance).
- WikiHow Lists (b-mc2/wikihow_lists) for multi-step procedural trajectories (Stage 3 / Latent Dynamics).

Matches the exact interface of GoodViewGenerator:
- sample_positive_pair() -> (view1_text, view2_text, concept_id, style1, style2)
- sample_reasoning_trajectory() -> (step_texts, step_styles, topic)
- get_all_texts() -> List[str]
"""

import glob
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import pyarrow.parquet as pq


class HFGoodViewLoader:
    """
    Loads real paraphrase clusters from TaPaCo and procedural action chains from WikiHow Lists.
    Caches items locally in memory for low-latency sampling and modest compute.
    """

    STYLES = ["standard", "formal", "casual", "concise"]

    def __init__(
        self,
        max_clusters: int = 5000,
        max_trajectories: int = 2000,
        min_cluster_size: int = 2,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.max_clusters = max_clusters
        self.max_trajectories = max_trajectories
        self.min_cluster_size = min_cluster_size

        self.clusters: Dict[str, List[str]] = {}
        self.trajectories: List[Tuple[str, List[str]]] = []
        self._load_datasets()

    def _load_datasets(self):
        """Loads and filters TaPaCo and WikiHow Lists via the datasets library."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required. Install via `pip install datasets`."
            )

        # Check if Recap-DataComp local shards exist (LeCun Good-View dataset)
        recap_files = sorted(glob.glob("data/recap_datacomp/data/train_data/*.parquet"))
        if recap_files:
            print(f"[*] Loading Good Views from local Recap-DataComp ({len(recap_files)} shards)...")
            self.recap_pairs = []
            for rf in recap_files:
                table = pq.read_table(rf, columns=["org_caption", "re_caption"])
                df = table.to_pandas()
                for _, r in df.iterrows():
                    o = str(r["org_caption"]).strip()
                    rc = str(r["re_caption"]).strip()
                    if 10 <= len(o) <= 300 and 10 <= len(rc) <= 300:
                        self.recap_pairs.append((o, rc))
                    if len(self.recap_pairs) >= max(self.max_clusters * 2, 20000):
                        break
                if len(self.recap_pairs) >= max(self.max_clusters * 2, 20000):
                    break
            print(f"[+] Loaded {len(self.recap_pairs)} Good-View pairs from Recap-DataComp (LeCun dataset).")
        else:
            self.recap_pairs = []

        print("[*] Loading TaPaCo paraphrase dataset for Good-View invariance...")
        tapaco_raw = load_dataset("community-datasets/tapaco", "en", split="train")

        raw_clusters = defaultdict(list)
        for row in tapaco_raw:
            text = row["paraphrase"].strip()
            # Basic sanity filters
            if 10 <= len(text) <= 250:
                raw_clusters[row["paraphrase_set_id"]].append(text)

        # Retain only clusters with >= min_cluster_size views
        filtered_cluster_keys = [
            cid for cid, views in raw_clusters.items() if len(views) >= self.min_cluster_size
        ]
        self.rng.shuffle(filtered_cluster_keys)
        selected_keys = filtered_cluster_keys[: self.max_clusters]

        self.clusters = {cid: raw_clusters[cid] for cid in selected_keys}
        print(f"[+] Loaded {len(self.clusters)} paraphrase clusters from TaPaCo.")

        print("[*] Loading b-mc2/wikihow_lists and soyasis/wikihow_small procedural datasets...")
        wikihow_raw = load_dataset("b-mc2/wikihow_lists", split="train")

        raw_trajectories = []
        for row in wikihow_raw:
            # Retain strictly procedural action steps (exclude needed_items/ingredient shopping lists)
            if row.get("result_type") not in ["stepped_summary", "step"]:
                continue
            title = row.get("title", "").strip()
            text = row.get("result", "")
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            cleaned_steps = []
            for l in lines:
                cleaned = re.sub(r"^\d+\.\s*", "", l).strip()
                cleaned = re.sub(r"^[•\-\*]\s*", "", cleaned).strip()
                if 5 <= len(cleaned) <= 200:
                    cleaned_steps.append(cleaned)
            if len(cleaned_steps) >= 3:
                raw_trajectories.append((title, cleaned_steps))

        try:
            small_raw = load_dataset("soyasis/wikihow_small", split="train")
            for row in small_raw:
                d = row.get("data", {})
                task = d.get("task", "").strip()
                summary = d.get("summary", "")
                steps = [s.strip() for s in summary.split("\n") if len(s.strip()) > 5]
                if len(steps) >= 3:
                    raw_trajectories.append((task, steps))
        except Exception as e:
            print(f"[!] Warning: could not load soyasis/wikihow_small: {e}")

        self.rng.shuffle(raw_trajectories)
        n_train = min(self.max_trajectories, int(len(raw_trajectories) * 0.85))
        self.train_trajectories = raw_trajectories[:n_train]
        self.test_trajectories = raw_trajectories[n_train:]
        self.trajectories = self.train_trajectories
        print(f"[+] Loaded {len(self.train_trajectories)} train trajectories and {len(self.test_trajectories)} unseen test trajectories from combined WikiHow sources.")

    def get_all_texts(self) -> List[str]:
        """Gathers a balanced text corpus for training the subword BPE tokenizer."""
        corpus = []
        if self.recap_pairs:
            for v1, v2 in self.recap_pairs[:8000]:
                corpus.extend([v1, v2])

        for views in self.clusters.values():
            corpus.extend(views)

        for q, steps in self.trajectories:
            corpus.append(q)
            corpus.extend(steps)

        return corpus

    def sample_positive_pair(self) -> Tuple[str, str, str, str, str]:
        """
        Samples two positive views of the identical proposition state z.
        Prefers real Recap-DataComp good views (LeCun dataset) when available.
        Returns:
            (view1_text, view2_text, concept_id, style1, style2)
        """
        s1, s2 = self.rng.sample(self.STYLES, 2)
        if self.recap_pairs:
            v1, v2 = self.rng.choice(self.recap_pairs)
            return v1, v2, "recap", s1, s2

        if not self.clusters:
            raise RuntimeError("No clusters or Recap pairs loaded in HFGoodViewLoader.")

        cluster_id = self.rng.choice(list(self.clusters.keys()))
        views = self.clusters[cluster_id]

        if len(views) >= 2:
            v1, v2 = self.rng.sample(views, 2)
        else:
            v1 = v2 = views[0]

        return v1, v2, str(cluster_id), s1, s2

    def sample_reasoning_trajectory(self, split: str = "train") -> Tuple[List[str], List[str], str]:
        """
        Samples a sequential procedural trajectory.
        Args:
            split: 'train' or 'test'
        Returns:
            step_texts: List of sequential steps [x^(1), ..., x^(T)]
            step_styles: List of style tags per step
            topic: The task goal prompt / topic
        """
        pool = self.test_trajectories if split == "test" else self.train_trajectories
        if not pool:
            pool = self.trajectories
        if not pool:
            raise RuntimeError("No trajectories loaded in HFGoodViewLoader.")

        topic, steps = self.rng.choice(pool)
        step_styles = [self.rng.choice(self.STYLES) for _ in steps]
        return list(steps), step_styles, topic

