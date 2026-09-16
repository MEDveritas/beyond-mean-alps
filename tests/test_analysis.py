"""Data-free tests for consequential numerical and release invariants."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]


def functions(module, names, **context):
    """Load pure source functions without running their private-data workflow."""
    tree=ast.parse((ROOT/'analysis'/f'{module}.py').read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(nodes)!=len(names):
        raise AssertionError('Missing requested source function')
    namespace={'np':np,'pd':pd,'StandardScaler':StandardScaler,'trapezoid':trapezoid,**context}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),module,'exec'),namespace)
    return namespace


class NumericalTests(unittest.TestCase):
    def test_replay_accepts_filename_change_preserving_original_manifest(self):
        validate=functions('prediction',['validate_replay_manifest'],json=json)['validate_replay_manifest']
        stored={'data_hash':'unchanged-data','config':{'data_stem':'input_dataset','seed':123},'features':['x']}
        original=copy.deepcopy(stored)
        current=copy.deepcopy(stored)
        current['config']['data_stem']='baseline_clinical_alps'
        validate(stored,current)
        self.assertEqual(stored,original)

    def test_replay_still_rejects_changed_data_model_settings_or_features(self):
        validate=functions('prediction',['validate_replay_manifest'],json=json)['validate_replay_manifest']
        stored={'data_hash':'unchanged-data','config':{'data_stem':'input_dataset','seed':123},'features':['x']}
        for field in ['data_hash','config','features']:
            changed=copy.deepcopy(stored)
            if field=='config':changed[field]['seed']=456
            elif field=='features':changed[field]=['x','y']
            else:changed[field]='different-data'
            with self.subTest(field=field),self.assertRaises(ValueError):validate(stored,changed)

    def test_log1p_preserves_zero_rejects_negative(self):
        f=functions('primary',['h_wmh'])['h_wmh']
        np.testing.assert_allclose(f(pd.Series([0.,1.,3.])),[0.,np.log(2),np.log(4)])
        with self.assertRaises(ValueError):f(pd.Series([-0.1,1.]))

    def test_scaler_excludes_validation_and_test(self):
        f=functions('prediction',['preprocess_fit'])['preprocess_fit']
        train=pd.DataFrame({'x':[0.,2.]})
        _,va,te,scaler=f(train,pd.DataFrame({'x':[101.]}),pd.DataFrame({'x':[-99.]}),['x'])
        np.testing.assert_allclose(scaler.mean_,[1.])
        np.testing.assert_allclose(va,[[100.]])
        np.testing.assert_allclose(te,[[-100.]])

    def test_censor_km_left_limit_and_support(self):
        fs=functions('prediction',['censor_km','km_lookup','eval_weight_columns'],CFG=SimpleNamespace(min_censor_survival=.05))
        km=fs['censor_km']([1.,2.,3.],[0,1,0])
        self.assertEqual(float(fs['km_lookup'](km,1.,left=True)),1.)
        self.assertAlmostEqual(float(fs['km_lookup'](km,1.)),2/3)
        train=pd.DataFrame({'time_months':[1.,2.,3.],'event':[0,1,0]})
        test=pd.DataFrame({'time_months':[.5,1.5,2.5],'event':[0,1,0]})
        weights=fs['eval_weight_columns'](test,train,[2.,3.])
        np.testing.assert_allclose(weights['wc_2'],[0,1.5,0])
        np.testing.assert_allclose(weights['wn_2'],[0,0,1.5])
        self.assertTrue((weights['supported_3']==0).all())

    def test_ipcw_ties_and_brier_denominator(self):
        cfg=SimpleNamespace(horizons=(12.,60.),ibs_grid=(12.,60.),core_repeats=1)
        fs=functions('prediction',['prepare_metrics','finite_repeat_mean','metric_vector'],CFG=cfg,
                     PRED_TIMES=np.array([12.,60.]),METRIC_KEYS=[(12.,'AUC'),(12.,'Brier'),('12_60','IBS')])
        # One case, one control and one early-censored participant; tied risk.
        values={'surv':[.5,.5,.5],'wc':[2.,0.,0.],'wn':[0.,2.,0.],
                'case':[1.,0.,0.],'control':[0.,1.,0.],'supported':[1.,1.,1.],'G':[.5,.5,.5]}
        cube={k:np.repeat(np.array(v)[None,:,None],2,axis=2) for k,v in values.items()}
        prepared=fs['prepare_metrics'](cube,np.ones(3,bool))
        metrics=fs['metric_vector'](prepared,np.ones(3))
        self.assertEqual(metrics[(12.,'AUC')],.5)
        self.assertAlmostEqual(metrics[(12.,'Brier')],1/3)
        self.assertAlmostEqual(metrics[('12_60','IBS')],1/3)

    def test_paired_resampling_preserves_auc_direction(self):
        cfg=SimpleNamespace(horizons=(12.,60.),ibs_grid=(12.,60.),core_repeats=1)
        fs=functions('prediction',['prepare_metrics','finite_repeat_mean','metric_vector'],CFG=cfg,
                     PRED_TIMES=np.array([12.,60.]),METRIC_KEYS=[])
        cube={k:np.repeat(np.array(v)[None,:,None],2,axis=2) for k,v in
              {'surv':[.1,.9],'wc':[1.,0.],'wn':[0.,1.],'case':[1.,0.],
               'control':[0.,1.],'supported':[1.,1.],'G':[1.,1.]}.items()}
        prepared=fs['prepare_metrics'](cube,np.ones(2,bool))
        self.assertEqual(fs['metric_vector'](prepared,[3,2])[(12.,'AUC')],1.)

    def test_multiplicity_known_values(self):
        f=functions('prediction',['adjust_pvalues'])['adjust_pvalues']
        np.testing.assert_allclose(f([.01,.04,.03],'holm'),[.03,.06,.06])
        np.testing.assert_allclose(f([.01,.04,.03],'bh'),[.03,.04,.04])


class ReleaseTests(unittest.TestCase):
    def test_all_python_compiles(self):
        for path in ROOT.rglob('*.py'):
            compile(path.read_text(encoding='utf-8'),str(path),'exec')

    def test_notebook_has_no_saved_outputs(self):
        nb=json.loads((ROOT/'Manuscript_Analyses.ipynb').read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                self.assertEqual(cell['outputs'],[])
                self.assertIsNone(cell['execution_count'])
                compile(''.join(cell['source']),'notebook','exec')

    def test_references_are_aggregate(self):
        manifest=json.loads((ROOT/'reference/manifest.json').read_text())
        self.assertEqual(len(manifest),26)
        for spec in manifest:
            columns=pd.read_csv(ROOT/'reference'/spec['reference'],nrows=0).columns
            self.assertFalse(set(columns)&{'PTID','subject_id','INDEX_DATE','row_index'})


if __name__=='__main__':unittest.main()
