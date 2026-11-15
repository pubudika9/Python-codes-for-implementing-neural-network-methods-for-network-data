"""Self-contained assertions: python tests_smoke.py"""
import numpy as np
import torch
from nqar_common import (simulate_panel,cce_design,ols_control_function,
                          outcome_qr, known_kernel_weights, PARAMETERS)
from deep_cfqr_cce import deep_crossfit_residuals
from graph_attention_w import fit_graph_attention


def test_dgp():
    d=simulate_panel(n=12,t=36,m=2,burn=12,seed=9,network_rule='nonlinear')
    assert d.W.shape==(36,12,12)
    assert np.allclose(d.W.sum(axis=-1),1,atol=1e-10)
    assert np.allclose(np.diagonal(d.W,axis1=1,axis2=2),0)
    assert np.all(d.W[:,~d.mask] == 0)
    assert np.max(np.abs(d.W[1]-d.W[2])) > 1e-5
    assert np.allclose(known_kernel_weights(d.Z,d.mask).sum(axis=-1),1)
    for proxy in ['rbar','rbar_new','sbar','sbar_new']:
        for lag in [0,1]:
            R,p=cce_design(d,proxy,lag)
            assert R.shape[0]==36 and np.isfinite(R).all()
    return d


def test_deep(d):
    R,_=cce_design(d,'sbar_new',0)
    eps0=ols_control_function(d,R,folds=3,gap=1)
    eps=deep_crossfit_residuals(d,R,folds=3,gap=1,epochs=8,seed=12)
    assert eps.shape==d.Z.shape and np.isfinite(eps).all()
    for z in [eps,eps0]:
        est=outcome_qr(d,R,z,d.W,lags=0)
        for p in PARAMETERS:
            assert len(est[p])==12 and np.isfinite(est[p]).all()


def test_graph(d):
    learned,metrics,split=fit_graph_attention(d,epochs=8,seed=12)
    assert split<d.Y.shape[1]
    assert np.all(np.isfinite(learned))
    assert np.allclose(learned.sum(axis=-1),1,atol=1e-5)
    assert np.all(learned[:,~d.mask]==0)
    assert metrics['test_MAE']>=0


if __name__=='__main__':
    torch.set_num_threads(1)
    data=test_dgp()
    test_deep(data)
    test_graph(data)
    print('All DGP, CCE, deep residual, QR, and GNN weight assertions passed.')
