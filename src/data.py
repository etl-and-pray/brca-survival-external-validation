"""Download and read cBioPortal DataHub study archives."""
from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd
import requests

from . import config


def _needed_files(key: str) -> list[str]:
    return [
        config.CLINICAL_PATIENT_FILE,
        config.CLINICAL_SAMPLE_FILE,
        config.EXPRESSION_FILES[key],
    ]


def study_dir(key: str) -> Path:
    return config.DATA_RAW / config.STUDIES[key]


def _download_archive(study: str, archive: Path) -> bool:
    """Try each archive URL in turn; return True on success."""
    for template in config.DATAHUB_URLS:
        url = template.format(study=study)
        print(f"  trying {url}")
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(archive, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            return True
        except requests.RequestException as e:
            print(f"  failed: {e}")
    return False


def _download_files_from_github(study: str, out_dir: Path, needed: list[str]) -> bool:
    """Fallback: fetch only the needed files from the GitHub DataHub repo (git LFS)."""
    try:
        for name in needed:
            url = config.GITHUB_FILE_URL.format(study=study, file=name)
            print(f"  trying {url}")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(out_dir / name, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
        return True
    except requests.RequestException as e:
        print(f"  failed: {e}")
        return False


def download_study(key: str, force: bool = False) -> Path:
    """Get the files this project needs for one study into data/raw/<study>/.

    Order: archive from DataHub -> single files from GitHub -> manual instructions.
    Files are written by base name, so archive paths never build output paths.
    """
    out_dir = study_dir(key)
    needed = _needed_files(key)
    if not force and all((out_dir / f).exists() for f in needed):
        print(f"[{key}] already present: {out_dir}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    study = config.STUDIES[key]
    archive = config.DATA_RAW / f"{study}.tar.gz"

    print(f"[{key}] downloading {study}")
    if _download_archive(study, archive):
        wanted, found, expression_like = set(needed), set(), []
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if member.isfile() and "mrna" in name.lower():
                    expression_like.append(name)
                if member.isfile() and name in wanted:
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    (out_dir / name).write_bytes(src.read())
                    found.add(name)
        archive.unlink()
        missing = wanted - found
        if missing:
            raise FileNotFoundError(
                f"{study}: files not found in archive: {sorted(missing)}\n"
                f"Expression-like files that ARE in the archive: {sorted(expression_like)}\n"
                f"Set the right name in config.EXPRESSION_FILES."
            )
        print(f"[{key}] extracted {sorted(found)} -> {out_dir}")
        return out_dir

    if _download_files_from_github(study, out_dir, needed):
        print(f"[{key}] downloaded files from GitHub -> {out_dir}")
        return out_dir

    raise RuntimeError(
        f"Could not download {study} automatically.\n"
        f"Manual option: open https://www.cbioportal.org/datasets, download '{study}', "
        f"extract it, and copy these files into {out_dir}:\n  " + "\n  ".join(needed)
    )


def read_cbio_table(path: Path) -> pd.DataFrame:
    """cBioPortal tables are tab-separated with '#' metadata lines on top."""
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)


def load_clinical(key: str) -> pd.DataFrame:
    """Patient + sample clinical tables merged on PATIENT_ID (one row per patient)."""
    d = study_dir(key)
    patient = read_cbio_table(d / config.CLINICAL_PATIENT_FILE)
    sample = read_cbio_table(d / config.CLINICAL_SAMPLE_FILE)
    # keep the first sample per patient (TCGA PanCancer and METABRIC are ~1:1)
    sample = sample.drop_duplicates("PATIENT_ID", keep="first")
    # avoid duplicated column names other than the join key
    dup = [c for c in sample.columns if c in patient.columns and c != "PATIENT_ID"]
    sample = sample.drop(columns=dup)
    return patient.merge(sample, on="PATIENT_ID", how="left")


def load_expression(key: str, genes: list[str], sample_to_patient: dict[str, str]) -> pd.DataFrame:
    """Return a patients x genes matrix restricted to `genes`.

    Duplicate gene symbols are averaged. Samples are mapped to patient IDs via
    the clinical sample table; samples that cannot be mapped are dropped.
    """
    path = study_dir(key) / config.EXPRESSION_FILES[key]
    df = read_cbio_table(path)
    df = df.dropna(subset=["Hugo_Symbol"])
    df["Hugo_Symbol"] = df["Hugo_Symbol"].replace(config.GENE_ALIASES)
    df = df[df["Hugo_Symbol"].isin(genes)]
    df = df.drop(columns=[c for c in ("Entrez_Gene_Id",) if c in df.columns])
    df = df.groupby("Hugo_Symbol").mean(numeric_only=True)
    mat = df.T  # samples x genes
    mat.index = mat.index.map(lambda s: sample_to_patient.get(s))
    mat = mat[mat.index.notna()]
    mat = mat[~mat.index.duplicated(keep="first")]
    return mat.astype(float)
