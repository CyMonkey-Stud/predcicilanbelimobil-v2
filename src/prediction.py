import json
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from .features import make_features

ROOT = Path(__file__).resolve().parents[1]


def load_assets(root=ROOT):
    root = Path(root)
    metadata = json.loads((root / 'models/model_metadata.json').read_text(encoding='utf-8'))
    if metadata['versions']['scikit-learn'] != sklearn.__version__:
        raise RuntimeError('Versi scikit-learn berbeda dari versi training.')
    models = {task: joblib.load(root / 'models' / info['file']) for task, info in metadata['models'].items()}
    catalog = pd.read_csv(root / 'data/katalog_mobil.csv')
    return models, metadata, catalog


def predict(models, metadata, price, down_payment, tenure, income):
    raw = pd.DataFrame([dict(price_idr=price, down_payment=down_payment,
                             loan_tenure=tenure, monthly_income=income)])
    outputs, warnings = {}, []
    for task, model in models.items():
        info = metadata['models'][task]
        x = make_features(raw, task, info['feature_set'])
        expected = next((step.feature_names_in_ for _, step in model.steps if hasattr(step, 'feature_names_in_')), None)
        if list(x.columns) != info['feature_columns'] or (expected is not None and list(expected) != list(x.columns)):
            raise RuntimeError('Skema fitur berbeda dari skema training.')
        outputs[task] = model.predict(x)[0].item()
        for col, limits in info['training_ranges'].items():
            if not limits['min'] <= raw[col].iloc[0] <= limits['max']:
                warnings.append(col)
    return float(outputs['reg']), int(outputs['clf']), sorted(set(warnings))
