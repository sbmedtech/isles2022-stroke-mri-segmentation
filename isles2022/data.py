"""Download, discover and split the official dataset with explicit paths."""

import json
from pathlib import Path
import random
import zipfile


def download_dataset(data_dir: Path, record_id: str = "7960856") -> Path:
    """Stream the official ZIP and retain the reference's byte-size check.

    Existing archives are reused only when their size matches Zenodo metadata.
    Size verification is not a cryptographic checksum or a network guarantee.
    """
    import requests

    data_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    response.raise_for_status()
    archive = next(item for item in response.json()["files"] if item["key"] == "ISLES-2022.zip")
    path = data_dir / archive["key"]
    if not path.exists() or path.stat().st_size != archive["size"]:
        with requests.get(archive["links"]["self"], stream=True, timeout=(30, 120)) as stream:
            stream.raise_for_status()
            with path.open("wb") as destination:
                for chunk in stream.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        destination.write(chunk)
    if path.stat().st_size != archive["size"]:
        raise ValueError("Dataset download is incomplete; retry from the official record.")
    return path


def extract_dataset(archive: Path, destination: Path) -> Path:
    """Extract the official archive and return its expected dataset root."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as contents:
        contents.extractall(destination)
    return destination / "ISLES-2022"


def discover_patients(dataset_dir: Path) -> list[str]:
    return sorted(path.name for path in dataset_dir.glob("sub-*") if path.is_dir())


def patient_splits(patient_ids: list[str], seed: int = 42) -> dict:
    """Preserve sorted IDs, Python seed-42 shuffle and the 175/37/38 split."""
    if len(patient_ids) != 250 or len(set(patient_ids)) != 250:
        raise ValueError("The reference partition requires 250 unique patients.")
    shuffled = sorted(patient_ids)
    random.Random(seed).shuffle(shuffled)
    result = {"random_seed": seed, "train": shuffled[:175],
              "validation": shuffled[175:212], "test": shuffled[212:]}
    validate_splits(result)
    return result


def validate_splits(splits: dict) -> None:
    """Reject duplicate/overlapping patient IDs before constructing patches."""
    groups = [splits[key] for key in ("train", "validation", "test")]
    if [len(group) for group in groups] != [175, 37, 38]:
        raise ValueError("Expected a 175/37/38 patient partition.")
    all_ids = sum(groups, [])
    if len(set(all_ids)) != 250:
        raise ValueError("Patient partitions overlap or contain duplicate IDs.")


def load_or_create_splits(dataset_dir: Path, path: Path, seed: int = 42) -> dict:
    expected = patient_splits(discover_patients(dataset_dir), seed)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != expected:
            raise ValueError("An existing patient split differs; review it before continuing.")
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, indent=4) + "\n")
    return expected
