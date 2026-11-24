from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

TENORS = (12, 24, 36, 48, 60)
DP_OPTIONS = (20, 25, 30, 35, 40)

@dataclass(frozen=True)
class FinanceResult:
    price: int
    down_payment: int
    principal: int
    tenure: int
    base_installment: int
    installment: int
    total_payment: int
    ratio: float
    installment_limit: int
    remaining_income: int
    affordable: bool


def calculate(price, dp_percent, tenure, income, other_debt=0, rate=0, mode='dataset'):
    if price <= 0 or income <= 0 or other_debt < 0 or dp_percent not in DP_OPTIONS or tenure not in TENORS:
        raise ValueError('Input finansial tidak valid.')
    rate = Decimal(str(rate))
    if not rate.is_finite() or not 0 <= rate <= 30 or mode not in ('dataset', 'flat'):
        raise ValueError('Mode atau bunga tidak valid.')
    if mode == 'dataset' and rate != 0:
        raise ValueError('Mode dataset tidak menggunakan bunga.')
    dp = price * dp_percent // 100
    principal = price - dp
    base = principal // tenure
    if mode == 'dataset':
        installment, total = base, price
    else:
        amount = Decimal(principal) / tenure + Decimal(principal) * rate / 1200
        installment = int(amount.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        total = dp + installment * tenure
    debt = installment + other_debt
    return FinanceResult(price, dp, principal, tenure, base, installment, total,
                         debt / income, max(0, income * 35 // 100 - other_debt),
                         income - debt, debt * 100 <= income * 35)
