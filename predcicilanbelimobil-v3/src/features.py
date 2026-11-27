import numpy as np
import pandas as pd

RAW_REG = ['price_idr', 'down_payment', 'loan_tenure']
RAW_CLF = ['monthly_income', *RAW_REG]
FIN_REG = [*RAW_REG, 'loan_amount', 'dp_ratio', 'base_installment']
FIN_CLF = [*RAW_CLF, 'loan_amount', 'dp_ratio', 'base_installment', 'estimated_dti', 'log_estimated_dti']

def make_features(frame, task, scenario):
    if task not in ('reg', 'clf') or scenario not in ('raw', 'financial'):
        raise ValueError('task/scenario tidak dikenal')
    raw_cols = RAW_REG if task == 'reg' else RAW_CLF
    x = frame[raw_cols].astype(float).copy()
    if not np.isfinite(x).all().all() or (x.price_idr <= 0).any() or (x.loan_tenure <= 0).any() or (x.down_payment < 0).any() or (x.down_payment >= x.price_idr).any():
        raise ValueError('Input finansial tidak valid')
    if task == 'clf' and (x.monthly_income <= 0).any():
        raise ValueError('Penghasilan harus positif')
    if scenario == 'raw':
        return x
    x['loan_amount'] = x.price_idr - x.down_payment
    x['dp_ratio'] = x.down_payment / x.price_idr
    x['base_installment'] = x.loan_amount / x.loan_tenure
    if task == 'clf':
        x['estimated_dti'] = x.base_installment / x.monthly_income
        x['log_estimated_dti'] = np.log(x.estimated_dti)
    return x[FIN_REG if task == 'reg' else FIN_CLF]

def formula_installment(frame):
    x = make_features(frame, 'reg', 'raw')
    return np.floor((x.price_idr - x.down_payment) / x.loan_tenure).to_numpy()

def formula_status(frame, threshold=0.35):
    x = make_features(frame, 'clf', 'raw')
    return (formula_installment(x) <= threshold * x.monthly_income).astype(int)
