import json

import pandas as pd
import pytest

from src.hydraulic_twin import predict
from src.hydraulic_twin.config import PROCESSED_DATA_DIR

pytestmark = pytest.mark.skipif(
    not (PROCESSED_DATA_DIR / "features.csv").exists(),
    reason="Run preprocessing first",
)


def test_corrupt_model_file_triggers_retraining(tmp_path, monkeypatch):
    bad_model = tmp_path / "valve_health_pipeline.joblib"
    bad_model.write_bytes(b"this is not a model")
    features_file = tmp_path / "valve_health_features.json"
    features_file.write_text(json.dumps(["x"]))

    monkeypatch.setattr(predict, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(predict, "MODEL_PATH", bad_model)
    monkeypatch.setattr(predict, "FEATURES_PATH", features_file)
    predict._load_model.cache_clear()

    with pytest.warns(UserWarning):
        model, names = predict._load_model()

    row = pd.read_csv(PROCESSED_DATA_DIR / "features.csv").iloc[[0]]
    assert set(names) <= set(row.columns)
    assert int(model.predict(row[names])[0]) in (73, 80, 90, 100)
    # the retrained model was saved for next time
    assert json.loads(features_file.read_text()) == names
    predict._load_model.cache_clear()