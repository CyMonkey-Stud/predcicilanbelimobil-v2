import unittest
from dataclasses import replace
from src.workflow import Request, simulate, export_result
from src.financial_calculator import calculate
from src.eligibility import explain_eligibility


class EligibilityTest(unittest.TestCase):
    req = Request('Daihatsu', 'Ayla', 'Hatchback', 170_000_000, 15_000_000, 0, 100_000_000, 25, 48, 'dataset', 0)

    def assess(self, **changes):
        r = replace(self.req, **changes)
        return explain_eligibility(r, calculate(r.price, r.dp, r.tenure, r.income, r.other, r.rate, r.mode))

    def test_pass(self):
        e = self.assess()
        self.assertTrue(e['eligible'])
        self.assertEqual(e['debt_gap'], 2_593_750)
        self.assertEqual(e['dp_gap'], 57_500_000)

    def test_ratio_only_fail(self):
        e = self.assess(income=5_000_000)
        self.assertFalse(e['eligible'])
        self.assertTrue(e['dp_ok'])
        self.assertEqual(e['debt_gap'], -906_250)
        self.assertEqual(e['min_income'], 7_589_286)
        self.assertTrue(e['alternatives'])
        for a in e['alternatives']:
            self.assertLessEqual(a['ratio'], .35)
            self.assertLessEqual(a['down_payment'], 100_000_000)

    def test_dp_only_fail(self):
        e = self.assess(budget=40_000_000)
        self.assertFalse(e['eligible'])
        self.assertTrue(e['debt_ok'])
        self.assertEqual(e['dp_gap'], -2_500_000)
        self.assertTrue(e['alternatives'])

    def test_both_fail(self):
        e = self.assess(income=5_000_000, budget=20_000_000)
        self.assertFalse(e['debt_ok'])
        self.assertFalse(e['dp_ok'])
        self.assertFalse(e['alternatives'])

    def test_exact_boundary(self):
        e = self.assess(price=120_000_000, dp=40, tenure=60, income=4_000_000, other=200_000, budget=48_000_000)
        self.assertTrue(e['eligible'])
        self.assertEqual(e['debt_gap'], 0)
        self.assertEqual(e['dp_gap'], 0)
        self.assertFalse(self.assess(price=120_000_000, dp=40, tenure=60, income=4_000_000, other=200_001)['eligible'])

    def test_other_debt_exhausts_limit(self):
        e = self.assess(other=5_250_000)
        self.assertTrue(e['other_exhausts_limit'])
        self.assertFalse(e['alternatives'])

    def test_flat_and_export(self):
        r = replace(self.req, mode='flat', rate=5, other=3_000_000)
        s = simulate(r, {}, {}, 'model unavailable')
        self.assertFalse(s['eligibility']['eligible'])
        self.assertEqual(s['eligibility']['debt_gap'], -937_500)
        csv = export_result(s).decode('utf-8-sig')
        self.assertIn('TIDAK LAYAK', csv)
        self.assertIn('Rasio seluruh cicilan melebihi 35%', csv)
