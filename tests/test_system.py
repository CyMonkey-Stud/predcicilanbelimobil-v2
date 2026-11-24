import unittest
from src.financial_calculator import calculate, DP_OPTIONS, TENORS
from src.prediction import load_assets, predict
from src.recommendation import recommend

class SystemTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models, cls.metadata, cls.catalog = load_assets()

    def test_formula(self):
        f = calculate(300_000_000, 20, 48, 20_000_000)
        self.assertEqual(f.installment, 5_000_000)
        self.assertEqual(f.total_payment, 300_000_000)
        self.assertTrue(f.affordable)

    def test_threshold(self):
        self.assertTrue(calculate(240_000_000, 30, 48, 10_000_000).affordable)
        self.assertFalse(calculate(240_000_000, 30, 48, 10_000_000, 1).affordable)

    def test_flat(self):
        f = calculate(300_000_000, 20, 48, 20_000_000, rate=5, mode='flat')
        self.assertEqual(f.installment, 6_000_000)
        self.assertEqual(f.total_payment, 348_000_000)

    def test_invalid(self):
        for args in [(300, 100, 48, 100), (300, 20, 0, 100), (300, 20, 48, 0)]:
            with self.assertRaises(ValueError):
                calculate(*args)

    def test_monotonicity(self):
        for car in self.catalog.itertuples():
            for t in TENORS:
                vals = [calculate(int(car.price_idr), d, t, 15_000_000).installment for d in DP_OPTIONS]
                self.assertEqual(vals, sorted(vals, reverse=True))
            for d in DP_OPTIONS:
                vals = [calculate(int(car.price_idr), d, t, 15_000_000).installment for t in TENORS]
                self.assertEqual(vals, sorted(vals, reverse=True))

    def test_prediction(self):
        f = calculate(300_000_000, 20, 48, 20_000_000)
        reg, clf, outside = predict(self.models, self.metadata, f.price, f.down_payment, f.tenure, 20_000_000)
        self.assertLess(abs(reg-f.installment), 1)
        self.assertEqual(clf, 1)
        self.assertFalse(outside)

    def test_outside(self):
        _, _, outside = predict(self.models, self.metadata, 170_000_000, 34_000_000, 60, 1_000_000)
        self.assertIn('monthly_income', outside)

    def test_recommendation(self):
        for mode, rate in [('dataset', 0), ('flat', 5)]:
            for ranking in ['Cicilan terendah','DP terendah','Tenor terpendek','Total pembayaran terendah','Harga mobil tertinggi']:
                r = recommend(self.catalog, 15_000_000, 80_000_000, 25, 48, 500_000, rate, mode, True, ranking, exclude=('Daihatsu','Ayla'))
                self.assertGreater(len(r), 0)
                self.assertLessEqual(len(r), 5)
                self.assertTrue((r.ratio <= .35).all())
                self.assertTrue((r.down_payment <= 80_000_000).all())
                self.assertFalse(r.duplicated(['company','model']).any())
                self.assertFalse(((r.company=='Daihatsu') & (r.model=='Ayla')).any())

    def test_no_recommendation(self):
        self.assertTrue(recommend(self.catalog, 1_000_000, 0, 20, 60).empty)

if __name__ == '__main__':
    unittest.main()
