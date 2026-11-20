"""
Negative Sample Ingestion Script for Vector Similarity Guardrail
================================================================
Populates the Qdrant vector database with negative (off-charter/adversarial)
sample embeddings sampled from benchmark datasets and Garak probes using the
formula:
    k = min(max(1, ceil(0.10 * N)), 10)

Strict execution: All Qdrant connection parameters are loaded from on_charter_intents.json.
If Qdrant is unreachable or invalid, an explicit RuntimeError is raised.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from impenetra.datasets.registry import DATASET_REGISTRY
from impenetra.models.embeddings import GuardrailEmbedder

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "on_charter_intents.json"
DEFAULT_COLLECTION_NAME = "off_charter_vectors"
VECTOR_DIM = 384  # sentence-transformers/all-MiniLM-L6-v2 dimension


def load_config(config_path: Path | str) -> dict[str, Any]:
    """Strictly load and validate on_charter_intents.json."""
    path = Path(config_path)
    if not path.is_file():
        raise ValueError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "qdrant_config" not in data:
        raise ValueError(f"Configuration file '{path}' missing required 'qdrant_config' section.")

    q_conf = data["qdrant_config"]
    if "url" not in q_conf or not q_conf["url"]:
        raise ValueError(f"'qdrant_config' in '{path}' must specify a valid 'url'.")

    return data


def get_dataset_negative_samples(random_seed: int = 42) -> list[dict[str, Any]]:
    """
    Extract negative prompts from every registered dataset.
    Takes min(10%, 10) random negative samples per dataset.
    """
    import csv

    sampled_items: list[dict[str, Any]] = []
    rng = random.Random(random_seed)

    for name, info in DATASET_REGISTRY.items():
        negatives: list[str] = []
        try:
            if info.path is not None and info.path.exists():
                with open(info.path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        is_blocked = (row.get("expected_blocked") or "").strip().lower() == "true"
                        is_neg = (row.get("type") or "").strip().lower() == "negative"
                        if is_blocked or is_neg:
                            prompt = (row.get("prompt") or "").strip()
                            if prompt:
                                negatives.append(prompt)
            elif info.hf_path is not None:
                from impenetra.datasets.loader import load_guardrail_dataset
                rows = load_guardrail_dataset(name)
                for r in rows:
                    if r.get("expected_blocked") or r.get("type") == "negative":
                        prompt = (r.get("prompt") or "").strip()
                        if prompt:
                            negatives.append(prompt)
        except Exception as e:
            print(f"⚠️ Warning loading dataset '{name}': {e}", file=sys.stderr)

        n = len(negatives)
        if n == 0:
            continue

        k = min(max(1, math.ceil(0.10 * n)), 10)
        chosen = rng.sample(negatives, min(k, n))
        for p in chosen:
            sampled_items.append({
                "prompt": p,
                "source": f"dataset:{name}",
            })

    return sampled_items


def get_garak_probe_negative_samples(random_seed: int = 42) -> list[dict[str, Any]]:
    """
    Extract negative prompts from Garak probes (including custom off-charter probes).
    Takes min(10%, 10) random negative samples per probe.
    """
    sampled_items: list[dict[str, Any]] = []
    rng = random.Random(random_seed)

    probe_classes: list[tuple[str, Any]] = []

    # 1. Custom Impenetra OffCharter probes
    try:
        from impenetra.garak_plugins.probes import off_charter
        for attr_name in dir(off_charter):
            obj = getattr(off_charter, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, off_charter.OffCharterBase)
                and getattr(obj, "active", False)
                and getattr(obj, "prompts", None)
            ):
                probe_classes.append((f"probes.off_charter.{attr_name}", obj))
    except Exception as e:
        print(f"⚠️ Warning importing custom off_charter probes: {e}", file=sys.stderr)

    # 2. Standard benchmark probes from Garak
    standard_probe_names = [
        "probes.dan.DanInTheWild",
        "probes.dan.AutoDANCached",
        "probes.promptinject.PromptInject",
        "probes.encoding.InjectBase64",
        "probes.grandma.Grandma",
        "probes.sysprompt_extraction.SyspromptExtraction",
    ]

    try:
        import garak._plugins
        for p_name in standard_probe_names:
            try:
                plugin = garak._plugins.load_plugin(p_name)
                if plugin:
                    probe_classes.append((p_name, plugin))
            except Exception:
                pass
    except Exception:
        pass

    for name, p_cls in probe_classes:
        prompts = getattr(p_cls, "prompts", None)
        if prompts is None and callable(p_cls):
            try:
                inst = p_cls()
                prompts = getattr(inst, "prompts", None)
            except Exception:
                pass

        if not prompts:
            continue

        prompt_list = [p for p in prompts if isinstance(p, str) and p.strip()]
        n = len(prompt_list)
        if n == 0:
            continue

        k = min(max(1, math.ceil(0.10 * n)), 10)
        chosen = rng.sample(prompt_list, min(k, n))
        for p in chosen:
            sampled_items.append({
                "prompt": p,
                "source": f"garak:{name}",
            })

    return sampled_items


def seed_qdrant_negative_samples(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    client: QdrantClient | None = None,
    batch_size: int = 32,
) -> int:
    """
    Seed Qdrant database with negative samples from datasets and Garak probes.
    Returns total number of points inserted.
    """
    config = load_config(config_path)
    q_conf = config["qdrant_config"]
    url = q_conf["url"]
    api_key = q_conf.get("api_key")
    timeout = int(float(q_conf.get("timeout", 10)))
    collection_name = q_conf.get("collection_name", DEFAULT_COLLECTION_NAME)

    if client is None:
        try:
            client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
            # Test connectivity strictly
            client.get_collections()
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Qdrant at {url}: {e}. "
                "Ensure Qdrant is running (e.g. docker compose up -d)."
            ) from e

    # Ensure collection exists
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

    # 1. Collect negative samples
    dataset_samples = get_dataset_negative_samples()
    garak_samples = get_garak_probe_negative_samples()
    all_samples = dataset_samples + garak_samples

    if not all_samples:
        raise RuntimeError("No negative samples could be extracted from datasets or probes.")

    # Deduplicate prompts
    seen_prompts: set[str] = set()
    unique_samples: list[dict[str, Any]] = []
    for item in all_samples:
        p_clean = item["prompt"].strip()
        if p_clean and p_clean not in seen_prompts:
            seen_prompts.add(p_clean)
            unique_samples.append(item)

    # 2. Encode embeddings using GuardrailEmbedder
    embedder = GuardrailEmbedder()
    tokenizer = embedder.model.tokenizer

    prompts = [item["prompt"] for item in unique_samples]
    embeddings = embedder.encode(prompts, batch_size=batch_size, normalize=True)
    embeddings_list = embeddings.cpu().tolist()

    points: list[PointStruct] = []
    for idx, (item, vec) in enumerate(zip(unique_samples, embeddings_list, strict=True), start=1):
        token_count = len(tokenizer.tokenize(item["prompt"]))
        points.append(
            PointStruct(
                id=idx,
                vector=vec,
                payload={
                    "text": item["prompt"],
                    "source": item["source"],
                    "token_count": token_count,
                },
            )
        )

    # 3. Upsert points into Qdrant in batches
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[i : i + batch_size],
        )

    # 4. Verify insertion
    info = client.get_collection(collection_name)
    print(
        f"✅ Qdrant collection '{collection_name}' successfully populated with "
        f"{info.points_count} negative vector samples ({len(dataset_samples)} from datasets, "
        f"{len(garak_samples)} from Garak probes)."
    )

    return len(points)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate Qdrant DB with negative sample vectors for vector guardrail."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to on_charter_intents.json configuration file",
    )
    args = parser.parse_args()

    try:
        count = seed_qdrant_negative_samples(config_path=args.config)
        print(f"🎉 Done! Inserted {count} negative samples into Qdrant.")
    except Exception as e:
        print(f"❌ Error seeding Qdrant: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
