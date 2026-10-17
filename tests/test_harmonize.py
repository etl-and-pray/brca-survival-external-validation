"""Smoke test for data parsing and harmonisation using tiny synthetic cBioPortal-style files.

Run:  python tests/test_harmonize.py      (or: pytest tests/)
Needs only numpy and pandas; the survival libraries are not imported here.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src import data as data_mod  # noqa: E402
from src import harmonize  # noqa: E402

HEADER = "#Patient Identifier\t#x\n#desc\t#desc\n#STRING\t#STRING\n#1\t#1\n"


def _write_cbio(path: Path, df: pd.DataFrame, with_meta: bool = True):
    body = df.to_csv(sep="\t", index=False)
    path.write_text(("#meta line 1\n#meta line 2\n#meta line 3\n#meta line 4\n" if with_meta else "") + body)


def make_fake_study(root: Path, key: str, n: int, rng):
    d = root / config.STUDIES[key]
    d.mkdir(parents=True)
    ids = [f"{key[:2].upper()}-{i:03d}" for i in range(n)]
    time = rng.uniform(5, 300, n)
    dead = rng.integers(0, 2, n)
    if key == "tcga":
        patient = pd.DataFrame({
            "PATIENT_ID": ids, "AGE": rng.integers(30, 85, n),
            "OS_MONTHS": time, "OS_STATUS": [f"{x}:{'DECEASED' if x else 'LIVING'}" for x in dead],
            "DFS_MONTHS": time, "DFS_STATUS": [f"{x}:{'Recurred/Progressed' if x else 'DiseaseFree'}" for x in dead],
            "AJCC_PATHOLOGIC_TUMOR_STAGE": rng.choice(["STAGE I", "STAGE IIA", "STAGE IIIC", "STAGE IV", "STAGE X", None], n),
            "SUBTYPE": rng.choice(["BRCA_LumA", "BRCA_LumB", "BRCA_Her2", "BRCA_Basal", "BRCA_Normal", None], n),
        })
        sample = pd.DataFrame({"PATIENT_ID": ids, "SAMPLE_ID": [i + "-01" for i in ids]})
        sample_ids = list(sample["SAMPLE_ID"])
    else:
        patient = pd.DataFrame({
            "PATIENT_ID": ids, "AGE_AT_DIAGNOSIS": rng.uniform(30, 85, n),
            "OS_MONTHS": time, "OS_STATUS": [f"{x}:{'DECEASED' if x else 'LIVING'}" for x in dead],
            "RFS_MONTHS": time, "RFS_STATUS": [f"{x}:{'Recurred' if x else 'Not Recurred'}" for x in dead],
            "CLAUDIN_SUBTYPE": rng.choice(["LumA", "LumB", "Her2", "Basal", "Normal", "claudin-low", "NC"], n),
        })
        sample = pd.DataFrame({"PATIENT_ID": ids, "SAMPLE_ID": ids,
                               "TUMOR_STAGE": rng.choice([1, 2, 3, 4, np.nan], n)})
        sample_ids = ids

    _write_cbio(d / config.CLINICAL_PATIENT_FILE, patient)
    _write_cbio(d / config.CLINICAL_SAMPLE_FILE, sample)

    genes = config.PAM50[:40] + ["NOT_IN_PANEL"]
    expr = pd.DataFrame(rng.gamma(2.0, 50, (len(genes), n)), columns=sample_ids)
    expr.insert(0, "Entrez_Gene_Id", range(len(genes)))
    expr.insert(0, "Hugo_Symbol", genes)
    _write_cbio(d / config.EXPRESSION_FILES[key], expr, with_meta=False)


def test_pipeline():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config.DATA_RAW = tmp / "raw"
        config.DATA_PROCESSED = tmp / "processed"
        for key, n in (("tcga", 200), ("metabric", 300)):
            make_fake_study(config.DATA_RAW, key, n, rng)

        # parsers
        assert harmonize.parse_status(pd.Series(["1:DECEASED", "0:LIVING", None])).tolist()[:2] == [1, 0]
        st = harmonize.parse_tcga_stage(pd.Series(["STAGE IIIC", "STAGE IV", "STAGE IA", "STAGE X", None]))
        assert st.tolist()[:3] == [3, 4, 1] and st.isna().tolist()[3:] == [True, True]
        sub = harmonize.parse_subtype(pd.Series(["BRCA_LumA", "claudin-low", "NC", None]))
        assert sub.tolist() == ["LumA", "Other", "Other", "Unknown"]

        for ep in ("OS", "RFS"):
            data = harmonize.build_datasets(endpoint=ep)
            assert set(data) == {"tcga", "metabric"}
            for key, d in data.items():
                clin, expr = d["clin"], d["expr"]
                assert clin.index.equals(expr.index), "clinical/expression must be aligned"
                assert clin["time"].max() <= config.HORIZON_MONTHS
                assert set(clin["event"].unique()) <= {0, 1}
                assert (clin.loc[clin["time"] == config.HORIZON_MONTHS, "event"] == 0).all()
                assert expr.notna().all().all()
                assert np.allclose(expr.mean(), 0, atol=1e-8) and np.allclose(expr.std(ddof=0), 1, atol=1e-8)
                assert "NOT_IN_PANEL" not in expr.columns
            assert list(data["tcga"]["expr"].columns) == list(data["metabric"]["expr"].columns)

            reloaded = harmonize.load_processed(endpoint=ep)
            pd.testing.assert_frame_equal(reloaded["tcga"]["clin"], data["tcga"]["clin"], check_dtype=False)

        frame = harmonize.model_frame(data["tcga"]["clin"], data["tcga"]["expr"])
        assert {"age", "stage", "subtype", "time", "event"} <= set(frame.columns)
    print("harmonize smoke test: OK")


if __name__ == "__main__":
    test_pipeline()
