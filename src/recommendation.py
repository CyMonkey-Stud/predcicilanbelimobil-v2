import pandas as pd
from .financial_calculator import calculate, DP_OPTIONS, TENORS


def recommend(catalog, income, budget, dp_percent, tenure, other_debt=0, rate=0,
              mode='dataset', explore=False, ranking='Cicilan terendah', body='Semua', exclude=None):
    orders = {
        'Cicilan terendah': ['installment', 'total_payment', 'down_payment', 'tenure'],
        'DP terendah': ['down_payment', 'installment', 'total_payment', 'tenure'],
        'Tenor terpendek': ['tenure', 'total_payment', 'installment', 'down_payment'],
        'Total pembayaran terendah': ['total_payment', 'installment', 'down_payment', 'tenure'],
        'Harga mobil tertinggi': ['price', 'installment', 'down_payment', 'tenure'],
    }
    if ranking not in orders or budget < 0:
        raise ValueError('Kriteria rekomendasi tidak valid.')
    rows = []
    for car in catalog.itertuples(index=False):
        if (car.company, car.model) == exclude or (body != 'Semua' and car.body_style != body):
            continue
        for d in DP_OPTIONS if explore else (dp_percent,):
            for t in TENORS if explore else (tenure,):
                f = calculate(int(car.price_idr), d, t, income, other_debt, rate, mode)
                if f.affordable and f.down_payment <= budget:
                    rows.append(dict(company=car.company, model=car.model, body_style=car.body_style,
                                     price=f.price, dp_percent=d, down_payment=f.down_payment, tenure=t,
                                     installment=f.installment, ratio=f.ratio, total_payment=f.total_payment))
    if not rows:
        return pd.DataFrame()
    cols = orders[ranking] + ['company', 'model']
    ascending = [not (ranking == 'Harga mobil tertinggi' and c == 'price') for c in cols]
    return (pd.DataFrame(rows).sort_values(cols, ascending=ascending)
            .drop_duplicates(['company', 'model']).head(5).reset_index(drop=True))
