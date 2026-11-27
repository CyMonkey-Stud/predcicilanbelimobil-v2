from dataclasses import dataclass, asdict
import math
import pandas as pd
from .financial_calculator import calculate
from .features import make_features, formula_status
from .recommendation import recommend
from .eligibility import explain_eligibility


@dataclass(frozen=True)
class Request:
    company: str
    model: str
    body: str
    price: int
    income: int
    other: int
    budget: int
    dp: int
    tenure: int
    mode: str
    rate: float


def model_trace(model, x, task):
    transformed = x
    steps = []
    for name, stage in model.steps[:-1]:
        if stage is None or stage == 'passthrough':
            steps.append({'Tahap': name, 'Proses': 'Passthrough (tanpa transformasi)'})
        else:
            transformed = stage.transform(transformed)
            steps.append({'Tahap': name, 'Proses': type(stage).__name__})
    estimator = model.steps[-1][1]
    values = list(transformed.iloc[0] if hasattr(transformed, 'iloc') else transformed[0])
    detail = {'steps': steps, 'estimator': type(estimator).__name__, 'features': pd.DataFrame({'Fitur': x.columns, 'Nilai input': x.iloc[0].to_numpy(), 'Nilai ke estimator': values})}
    if task == 'reg' and hasattr(estimator, 'coef_') and len(estimator.coef_) == len(values):
        detail['contributions'] = pd.DataFrame({'Fitur': x.columns, 'Nilai transformasi': values, 'Koefisien': estimator.coef_, 'Kontribusi (Rp)': estimator.coef_ * values})
        detail['intercept'] = float(estimator.intercept_)
        detail['reconstructed'] = float(detail['contributions']['Kontribusi (Rp)'].sum() + detail['intercept'])
    if task == 'clf' and hasattr(estimator, 'tree_'):
        tree = estimator.tree_
        nodes = estimator.decision_path(transformed).indices
        path = []
        for node in nodes:
            idx = tree.feature[node]
            if idx < 0:
                path.append({'Node': int(node), 'Fitur': 'Leaf', 'Nilai': None, 'Ambang': None, 'Cabang': 'Kelas ' + str(estimator.classes_[tree.value[node][0].argmax()])})
            else:
                # sklearn tree compares float32 inputs internally
                import numpy as np
                val = float(np.float32(values[idx]))
                threshold = float(tree.threshold[node])
                path.append({'Node': int(node), 'Fitur': x.columns[idx], 'Nilai': val, 'Ambang': threshold, 'Cabang': 'Kiri (≤)' if val <= threshold else 'Kanan (>)'})
        detail['path'] = pd.DataFrame(path)
        detail['depth'] = estimator.get_depth()
    return detail


def simulate(req, models, metadata, asset_error=None):
    if req.budget < 0:
        raise ValueError('Anggaran DP tidak boleh negatif.')
    f = calculate(req.price, req.dp, req.tenure, req.income, req.other, req.rate, req.mode)
    raw = pd.DataFrame([dict(price_idr=req.price, down_payment=f.down_payment, loan_tenure=req.tenure, monthly_income=req.income)])
    result = {'request': req, 'finance': f, 'raw': raw, 'budget_ok': f.down_payment <= req.budget,
              'eligibility': explain_eligibility(req, f), 'rule_label': int(formula_status(raw)[0]), 'models': {}, 'errors': {}, 'warnings': [], 'trace_errors': {}}
    for task in ('reg', 'clf'):
        try:
            if asset_error:
                raise RuntimeError(asset_error)
            info, model = metadata['models'][task], models[task]
            x = make_features(raw, task, info['feature_set'])
            expected = next((step.feature_names_in_ for _, step in model.steps if hasattr(step, 'feature_names_in_')), None)
            if list(x.columns) != info['feature_columns'] or (expected is not None and list(expected) != list(x.columns)):
                raise RuntimeError('Skema fitur berbeda dari skema training.')
            prediction = float(model.predict(x)[0])
            if not math.isfinite(prediction) or (task == 'reg' and prediction < 0) or (task == 'clf' and prediction not in (0, 1)):
                raise ValueError('Output model tidak valid.')
            result['models'][task] = {'name': info['name'], 'prediction': prediction, 'features': x}
            for col, limits in info['training_ranges'].items():
                value = raw[col].iloc[0]
                if not limits['min'] <= value <= limits['max']:
                    result['warnings'].append({'Model': task, 'Fitur': col, 'Input': float(value), 'Minimum training': limits['min'], 'Maksimum training': limits['max']})
            try:
                result['models'][task]['trace'] = model_trace(model, x, task)
            except Exception as exc:
                result['trace_errors'][task] = f'{type(exc).__name__}: {exc}'
        except Exception as exc:
            result['errors'][task] = f'{type(exc).__name__}: {exc}'
    return result


def result_checks(result):
    r, f = result['request'], result['finance']
    checks = [
        ('Validasi input', 'Lulus', 'Nilai finansial dan skema diterima.'),
        ('Pokok pembiayaan', 'Lulus' if f.principal + f.down_payment == r.price else 'Gagal', 'Harga dikurangi DP.'),
        ('Rasio seluruh cicilan', 'Lulus' if f.affordable else 'Melebihi batas', 'Batas 35% penghasilan.'),
        ('Anggaran DP', 'Lulus' if result['budget_ok'] else 'Melebihi batas', 'Uang muka dibandingkan anggaran.'),
    ]
    for task, label in [('reg', 'Regresi ML'), ('clf', 'Klasifikasi ML')]:
        if task in result['errors']:
            checks.append((label, 'Gagal', result['errors'][task]))
        else:
            pred = result['models'][task]['prediction']
            if task == 'reg':
                error = abs(pred - f.base_installment)
                checks.append((label, 'Lulus' if error <= 1 else 'Periksa', f'Selisih dari rumus dataset Rp {error:,.4f}; ambang diagnostik Rp 1.'))
            else:
                checks.append((label, 'Lulus' if pred == result['rule_label'] else 'Periksa', 'Dibandingkan dengan aturan label dataset tanpa bunga/utang lain.'))
    checks.append(('Rentang data latih', 'Periksa' if result['warnings'] else ('Tidak diuji' if result['errors'] else 'Lulus'), f"{len(result['warnings'])} peringatan model-fitur."))
    for task, error in result['trace_errors'].items():
        checks.append((f'Interpretasi {task}', 'Gagal', error))
    return pd.DataFrame(checks, columns=['Tahap', 'Status', 'Keterangan'])


def recommend_for(req, catalog, explore, ranking, body):
    return recommend(catalog, req.income, req.budget, req.dp, req.tenure, req.other, req.rate, req.mode, explore, ranking, body, (req.company, req.model), return_stats=True)


def export_result(result):
    row = asdict(result['request']) | asdict(result['finance'])
    row['dp_budget_ok'] = result['budget_ok']
    e = result['eligibility']
    row['simulation_eligible'] = e['eligible']
    row['status_simulasi'] = 'LAYAK' if e['eligible'] else 'TIDAK LAYAK'
    row['alasan_status'] = ('Rasio dan anggaran DP terpenuhi' if e['eligible'] else '; '.join(
        reason for failed, reason in [(not e['debt_ok'], 'Rasio seluruh cicilan melebihi 35%'),
                                     (not e['dp_ok'], 'Uang muka melebihi anggaran')] if failed))
    for key in ('total_debt', 'debt_limit', 'debt_gap', 'dp_gap', 'min_income', 'income_shortfall'):
        row[key] = e[key]
    row['dataset_rule_label'] = result['rule_label']
    for task, data in result['models'].items():
        row['ml_' + task] = data['prediction']
    return pd.DataFrame([row]).to_csv(index=False).encode('utf-8-sig')
