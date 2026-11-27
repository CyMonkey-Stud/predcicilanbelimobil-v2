from .financial_calculator import calculate, DP_OPTIONS, TENORS


def explain_eligibility(req, finance):
    debt = finance.installment + req.other
    limit = req.income * 35 // 100
    debt_gap = limit - debt
    dp_gap = req.budget - finance.down_payment
    min_income = (debt * 100 + 34) // 35
    alternatives = []
    if debt_gap < 0 or dp_gap < 0:
        for dp in DP_OPTIONS:
            for tenure in TENORS:
                if (dp, tenure) == (req.dp, req.tenure):
                    continue
                f = calculate(req.price, dp, tenure, req.income, req.other, req.rate, req.mode)
                if f.affordable and f.down_payment <= req.budget:
                    alternatives.append({'dp_percent': dp, 'tenure': tenure,
                                         'down_payment': f.down_payment, 'installment': f.installment,
                                         'ratio': f.ratio, 'total_payment': f.total_payment})
        alternatives.sort(key=lambda x: (abs(x['dp_percent'] - req.dp) // 5 + abs(x['tenure'] - req.tenure) // 12,
                                         x['total_payment'], x['installment'], x['down_payment']))
    return {'eligible': debt_gap >= 0 and dp_gap >= 0, 'debt_ok': debt_gap >= 0, 'dp_ok': dp_gap >= 0,
            'total_debt': debt, 'debt_limit': limit, 'debt_gap': debt_gap, 'dp_gap': dp_gap,
            'min_income': min_income, 'income_shortfall': max(0, min_income - req.income),
            'other_exhausts_limit': req.other >= limit, 'alternatives': alternatives}
