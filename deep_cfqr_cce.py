"""Deep control-function CFQR--CCE simulation.

A shared neural representation of CCE averages is fitted to Z with node-specific
loadings and auxiliary slopes. Out-of-fold residuals enter ordinary node-specific
quantile regression. The network W is *known from the observed Z kernel*.
This is NOT a proof of identification, NED convergence, or causal inference.

Examples:
 python deep_cfqr_cce.py --n 30 --t 60 --replications 2 --epochs 80
 python deep_cfqr_cce.py --n-grid 50 100 --t-grid 50 100 --replications 500 --epochs 250
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from nqar_common import (PARAMETERS, simulate_panel, cce_design,
                         ols_control_function, outcome_qr, error_summary,
                         aggregate_repetitions)


class DeepAuxiliary(nn.Module):
    """z_it/scale_z = intercept_i + beta_zi*(xz_it standardized)
                         + a_iz' NN_phi(R_t standardized)."""
    def __init__(self, n: int, q: int, hidden: int=20, latent: int=4):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(q,hidden),nn.Tanh(),
                                   nn.Linear(hidden,hidden),nn.Tanh(),
                                   nn.Linear(hidden,latent))
        self.intercept=nn.Parameter(torch.zeros(n))
        self.beta_z=nn.Parameter(torch.zeros(n))
        self.loadings=nn.Parameter(0.10*torch.randn(n,latent))

    def forward(self, R_t, Xz_it):
        h_t=self.encoder(R_t)                    # (T,latent)
        factor=self.loadings@h_t.T                # (N,T)
        return self.intercept[:,None]+self.beta_z[:,None]*Xz_it+factor


def _train_one_fold(data,R,train_id,epochs=150,seed=11,
                    hidden=20,latent=4,patience=35,lr=0.012):
    """Returns Zhat for ALL times, fitted on training times only.

    Uses training-only scaling and reserves the end of train_id for stopping.
    Time indices are never included in the network: only measured CCE averages.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    N,T=data.Z.shape
    z_scale=max(float(data.Z[:,train_id].std()),1e-4)
    z_mean=float(data.Z[:,train_id].mean())
    rmean=R[train_id].mean(axis=0)
    rstd=np.maximum(R[train_id].std(axis=0),1e-4)
    xmean=float(data.Xz[:,train_id].mean())
    xstd=max(float(data.Xz[:,train_id].std()),1e-4)
    rn=torch.tensor((R-rmean)/rstd,dtype=torch.float32)
    xn=torch.tensor((data.Xz-xmean)/xstd,dtype=torch.float32)
    zn=torch.tensor((data.Z-z_mean)/z_scale,dtype=torch.float32)

    nval=max(4,len(train_id)//6) if len(train_id)>=24 else 0
    fitid=train_id[:-nval] if nval else train_id
    valid=train_id[-nval:] if nval else np.array([],dtype=int)
    model=DeepAuxiliary(N,R.shape[1],hidden=hidden,latent=latent)
    optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    best_state=None
    best_value=float('inf')
    no_improve=0
    for epoch in range(epochs):
        model.train()
        pred=model(rn,xn)
        mse=(pred[:,fitid]-zn[:,fitid]).square().mean()
        penalty=1e-4*(model.loadings.square().mean()+model.beta_z.square().mean())
        loss=mse+penalty
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),3.0)
        optim.step()
        model.eval()
        with torch.no_grad():
            newpred=model(rn,xn)
            value=(newpred[:,valid]-zn[:,valid]).square().mean().item() if nval else loss.item()
        if value < best_value-1e-6:
            best_value=value
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            no_improve=0
        else:
            no_improve+=1
        if epoch >= 30 and no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        zhat=model(rn,xn).numpy()*z_scale+z_mean
    return zhat


def deep_crossfit_residuals(data,R,folds=3,gap=1,epochs=150,seed=123,
                            hidden=20,latent=4):
    """Blocked time-fold out-of-fold epsilon estimates, with an embargo gap."""
    n,t=data.Z.shape
    if not 2<=folds<=t//4:
        raise ValueError('Use 2 <= folds <= T//4.')
    eps=np.zeros_like(data.Z)
    for f,holdout in enumerate(np.array_split(np.arange(t),folds)):
        exclude=np.zeros(t,dtype=bool)
        for k in holdout:
            exclude[max(0,k-gap):min(t,k+gap+1)]=True
        train=np.flatnonzero(~exclude)
        if len(train)<16:
            raise ValueError('Insufficient train times for network first stage.')
        pred=_train_one_fold(data,R,train,epochs=epochs,
                            seed=seed+100*f,hidden=hidden,latent=latent)
        eps[:,holdout]=data.Z[:,holdout]-pred[:,holdout]
    return eps


def run_one(n,t,tau,delta,m,rep_seed,aux_rule,network_rule,proxy,lags,folds,
            gap,epochs,hidden,latent,burn):
    data=simulate_panel(n=n,t=t,m=m,tau=tau,delta0=delta,seed=rep_seed,burn=burn,
                        auxiliary_rule=aux_rule,network_rule=network_rule)
    R,p=cce_design(data,proxy,lags)
    ols_eps=ols_control_function(data,R,folds=folds,gap=gap)
    deep_eps=deep_crossfit_residuals(data,R,folds=folds,gap=gap,
                   epochs=epochs,seed=rep_seed+17,hidden=hidden,latent=latent)
    results=[]
    estimates={}
    for name,eps in [('CFQR_CCE_OLS',ols_eps),('Deep_CFQR_CCE',deep_eps),
                     ('Oracle_epsilon_CCE',data.eps_true)]:
        est=outcome_qr(data,R,eps,data.W,tau=tau,lags=p)
        estimates[name]=est
        line={'N':n,'T':t,'tau':tau,'delta0':delta,'rep_seed':rep_seed,
              'auxiliary_rule':aux_rule,'network_rule':network_rule,
              'proxy':proxy,'lags':p,'Method':name,
              'EpsRMSE':float(np.sqrt(((eps-data.eps_true)**2).mean()))}
        line.update(error_summary(est,data.truth))
        results.append(line)
    return results,estimates,data


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n',type=int,default=30)
    ap.add_argument('--t',type=int,default=60)
    ap.add_argument('--n-grid',type=int,nargs='+',default=None)
    ap.add_argument('--t-grid',type=int,nargs='+',default=None)
    ap.add_argument('--replications',type=int,default=1)
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--tau',type=float,default=0.5)
    ap.add_argument('--delta',type=float,default=0.8)
    ap.add_argument('--factors',type=int,default=2)
    ap.add_argument('--auxiliary-rule',choices=['linear','nonlinear'],default='linear')
    ap.add_argument('--network-rule',choices=['kernel','nonlinear'],default='kernel')
    ap.add_argument('--proxy',choices=['rbar','rbar_new','sbar','sbar_new'],default='sbar_new')
    ap.add_argument('--lags',default='0',help='Integer or auto (ceil(T^(1/3)))')
    ap.add_argument('--folds',type=int,default=3)
    ap.add_argument('--gap',type=int,default=1)
    ap.add_argument('--epochs',type=int,default=150)
    ap.add_argument('--hidden',type=int,default=20)
    ap.add_argument('--latent',type=int,default=4)
    ap.add_argument('--burn',type=int,default=100)
    ap.add_argument('--outdir',default='results_deep_cf')
    args=ap.parse_args()
    out=Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)
    ngrid=args.n_grid if args.n_grid else [args.n]
    tgrid=args.t_grid if args.t_grid else [args.t]
    rec=[]
    for n in ngrid:
        for t in tgrid:
            for rep in range(args.replications):
                print(f'Deep CFQR-CCE: N={n} T={t} rep={rep+1}/{args.replications}',flush=True)
                seed=args.seed+10_000*n+100*t+rep
                rows,estimates,data=run_one(
                    n,t,args.tau,args.delta,args.factors,seed,args.auxiliary_rule,
                    args.network_rule,args.proxy,args.lags,args.folds,args.gap,
                    args.epochs,args.hidden,args.latent,args.burn)
                rec.extend(rows)
                if rep==0:
                    info={'node':np.arange(n)}
                    for p in PARAMETERS:
                        info[f'true_{p}']=data.truth[p]
                        for method in estimates:
                            info[f'{method}_{p}']=estimates[method][p]
                    pd.DataFrame(info).to_csv(out/f'node_estimates_N{n}_T{t}.csv',index=False)
    df=pd.DataFrame(rec)
    df.to_csv(out/'replication_results.csv',index=False)
    summary=aggregate_repetitions(df,['N','T','tau','delta0','auxiliary_rule','network_rule',
                                      'proxy','lags','Method'])
    summary.to_csv(out/'monte_carlo_summary.csv',index=False)
    print('\n',summary[['N','T','Method','AbsBias_Gamma','RMSE_Gamma',
                         'AbsBias_beta1','RMSE_beta1']].to_string(index=False))
    print('Saved outputs in:',out.resolve())

if __name__=='__main__':
    main()
