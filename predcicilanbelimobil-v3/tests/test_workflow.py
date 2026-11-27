import unittest
from dataclasses import replace
from src.prediction import load_assets, predict
from src.workflow import Request, simulate, result_checks, recommend_for, export_result

class WorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models, cls.meta, cls.catalog = load_assets()
        cls.req = Request('Daihatsu','Ayla','Hatchback',170_000_000,15_000_000,0,100_000_000,25,48,'dataset',0)

    def run_sim(self, req=None):
        return simulate(req or self.req,self.models,self.meta)

    def test_actual_model_trace(self):
        s = self.run_sim()
        self.assertFalse(s['errors'])
        self.assertFalse(s['trace_errors'])
        self.assertEqual(s['finance'].installment,2_656_250)
        reg,clf,_ = predict(self.models,self.meta,self.req.price,s['finance'].down_payment,self.req.tenure,self.req.income)
        self.assertAlmostEqual(reg,s['models']['reg']['prediction'])
        self.assertEqual(clf,s['models']['clf']['prediction'])
        self.assertAlmostEqual(s['models']['reg']['trace']['reconstructed'],reg,places=4)
        self.assertEqual(s['models']['clf']['trace']['path'].iloc[-1]['Cabang'],f'Kelas {clf}')

    def test_flat_keeps_ml_definition(self):
        a = self.run_sim()
        b = self.run_sim(replace(self.req, mode='flat',rate=5,other=3_000_000))
        self.assertEqual(b['finance'].installment,3_187_500)
        self.assertFalse(b['finance'].affordable)
        self.assertEqual(a['models']['clf']['prediction'],b['models']['clf']['prediction'])
        self.assertEqual(a['models']['reg']['prediction'],b['models']['reg']['prediction'])

    def test_outside_and_budget(self):
        s = self.run_sim(replace(self.req,income=1_000_000,budget=1))
        self.assertTrue(any(w['Fitur']=='monthly_income' for w in s['warnings']))
        self.assertFalse(s['budget_ok'])
        self.assertFalse(s['finance'].affordable)
        self.assertIn('Periksa',result_checks(s)['Status'].tolist())

    def test_model_failure_fallback(self):
        s = simulate(self.req,{}, {},'PKL tidak tersedia')
        self.assertEqual(set(s['errors']),{'reg','clf'})
        self.assertEqual(s['finance'].installment,2_656_250)
        self.assertIn('Gagal',result_checks(s)['Status'].tolist())
        self.assertTrue(export_result(s))

    def test_single_model_failure(self):
        s = simulate(self.req,{'reg':self.models['reg']},self.meta)
        self.assertIn('clf',s['errors'])
        self.assertIn('reg',s['models'])

    def test_recommendation_stats(self):
        rows,stats = recommend_for(self.req,self.catalog,True,'Cicilan terendah','Semua')
        self.assertEqual(stats['combinations'],stats['cars_considered']*25)
        self.assertLessEqual(stats['both_pass'],min(stats['ratio_pass'],stats['dp_pass']))
        self.assertEqual(stats['displayed'],len(rows))
        self.assertLessEqual(len(rows),stats['unique_cars'])
        self.assertTrue((rows.ratio<=.35).all())
        self.assertTrue((rows.down_payment<=self.req.budget).all())

    def test_empty_recommendation_stats(self):
        rows,stats = recommend_for(replace(self.req,budget=0),self.catalog,False,'Cicilan terendah','Semua')
        self.assertTrue(rows.empty)
        self.assertGreater(stats['combinations'],0)
        self.assertEqual(stats['dp_pass'],0)
        self.assertEqual(stats['displayed'],0)

    def test_export(self):
        data = export_result(self.run_sim()).decode('utf-8-sig')
        self.assertIn('simulation_eligible',data)
        self.assertIn('Daihatsu',data)
        self.assertIn('2656250',data)

if __name__ == '__main__': unittest.main()
