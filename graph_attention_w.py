"""Graph attention learning of time-varying W_t for factor-augmented NQAR.

A two-layer MESSAGE-PASSING GNN forms node embeddings from observed Z and Xz.
An edge-attention MLP produces masked, row-normalized weights W_ijt.
Training is SUPERVISED by *observed network weights* in this simulation.
The real network would have to be measured (transactions, contacts, etc.).
No true W is fed to the estimator's outcome regression after graph training.

This is not an outcome-only identification method or a proof of consistency.

Examples:
 python graph_attention_w.py --n 30 --t 65 --epochs 100
 python graph_attention_w.py --n 60 --t 100 --replications 10 --epochs 200
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
                         aggregate_repetitions, known_kernel_weights)


class DynamicGraphAttention(nn.Module):
    """Two message-passing layers + masked neighborhood attention.

    The GNN uses ONLY observed Z_it, Xz_it and spatial neighbor support.
    It never receives contemporaneous Y_t to construct attention weights.
    """
    def __init__(self, mask: np.ndarray, dspace: np.ndarray,
                 node_dim=12, hidden=24, eta_init=1.0):
        super().__init__()
        n=mask.shape[0]
        src,dst=np.nonzero(mask)
        deg=mask.sum(axis=1,keepdims=True)
        adjacency=(mask.astype(np.float32)/deg).astype(np.float32)
        self.register_buffer('neighbor_avg',torch.tensor(adjacency))
        self.register_buffer('src',torch.tensor(src,dtype=torch.long))
        self.register_buffer('dst',torch.tensor(dst,dtype=torch.long))
        self.register_buffer('distance',torch.tensor(dspace[src,dst],dtype=torch.float32))
        self.n=n
        self.self1=nn.Linear(2,node_dim)
        self.neigh1=nn.Linear(2,node_dim,bias=False)
        self.self2=nn.Linear(node_dim,node_dim)
        self.neigh2=nn.Linear(node_dim,node_dim,bias=False)
        # abs(embedding difference), elementwise embedding similarity,
        # abs(Z difference), abs(Xz difference), sin(Z sum), spatial distance.
        self.attn=nn.Sequential(nn.Linear(2*node_dim+4,hidden),nn.Tanh(),
                                nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.eta=nn.Parameter(torch.tensor(float(eta_init)))

    def forward(self, node_features: torch.Tensor,
                z_raw: torch.Tensor, xz_raw: torch.Tensor):
        """node_features: (T,N,2), z_raw and xz_raw: (T,N)

        Returns p[t,e] where e=(src[e], dst[e]); rows sum to one.
        """
        av1=torch.matmul(self.neighbor_avg,node_features)     # (T,N,2)
        h=torch.tanh(self.self1(node_features)+self.neigh1(av1))
        av2=torch.matmul(self.neighbor_avg,h)                 # (T,N,node_dim)
        h=torch.tanh(self.self2(h)+self.neigh2(av2))
        hi,hj=h[:,self.src],h[:,self.dst]                     # (T,E,node_dim)
        zi,zj=z_raw[:,self.src],z_raw[:,self.dst]
        xi,xj=xz_raw[:,self.src],xz_raw[:,self.dst]
        dz=torch.abs(zi-zj)
        dx=torch.abs(xi-xj)
        sz=torch.sin(zi+zj)
        dist=self.distance[None,:].expand_as(dz)
        edge=torch.cat([torch.abs(hi-hj),hi*hj,
                 dz[:,:,None],dx[:,:,None],sz[:,:,None],dist[:,:,None]],dim=2)
        adjustment=2.0*torch.tanh(self.attn(edge).squeeze(-1))
        scores=-torch.nn.functional.softplus(self.eta)*dz+adjustment
        # Numerically stable sparse masked softmax over each i's neighbors.
        edge_source=self.src[None,:].expand_as(scores)
        maximum=torch.full((scores.shape[0],self.n),-1e30,device=scores.device)
        maximum=maximum.scatter_reduce(1,edge_source,scores,reduce='amax',include_self=True)
        es=torch.exp(scores-maximum.gather(1,edge_source))
        denom=torch.zeros((scores.shape[0],self.n),device=scores.device)
        denom=denom.scatter_add(1,edge_source,es)
        prob=es/denom.gather(1,edge_source).clamp_min(1e-30)
        return prob


def fit_graph_attention(data,epochs=180,seed=321,hidden=24,node_dim=12,
                        learning_rate=0.012):
    """Train with network observations from first 70% of times.

    In the simulation the observed weight labels are data.W. In real data,
    these must come from observed edge intensities or similar information.
    The last 30% is held out when reporting graph prediction error.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    n,t=data.Z.shape
    split=max(10,int(0.7*t))
    nvalid=max(4,int(0.15*split))
    fit_id=np.arange(split-nvalid)
    valid_id=np.arange(split-nvalid,split)
    test_id=np.arange(split,t)
    # Standardize inputs with training-period statistics only.
    node_raw=np.stack([data.Z.T,data.Xz.T],axis=-1)
    mean=node_raw[:split].reshape(-1,2).mean(axis=0)
    sd=np.maximum(node_raw[:split].reshape(-1,2).std(axis=0),1e-5)
    features=torch.tensor((node_raw-mean)/sd,dtype=torch.float32)
    z=torch.tensor(data.Z.T,dtype=torch.float32)
    xz=torch.tensor(data.Xz.T,dtype=torch.float32)
    model=DynamicGraphAttention(data.mask,data.dspace,node_dim=node_dim,hidden=hidden)
    src,dst=np.nonzero(data.mask)
    teacher=torch.tensor(data.W[:,src,dst],dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)
    best_value=float('inf')
    best_state=None
    wait=0
    for epoch in range(epochs):
        model.train()
        p=model(features,z,xz)
        loss=-(teacher[fit_id]*torch.log(p[fit_id].clamp_min(1e-12))).sum(dim=1).mean()/n
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            pval=model(features,z,xz)
            v=-(teacher[valid_id]*torch.log(pval[valid_id].clamp_min(1e-12))).sum(dim=1).mean().item()/n
        if v < best_value-1e-7:
            best_value=v
            best_state={k:x.detach().clone() for k,x in model.state_dict().items()}
            wait=0
        else:
            wait+=1
        if epoch>=35 and wait>=35:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred=model(features,z,xz).numpy()
    W=np.zeros_like(data.W)
    W[:,src,dst]=pred
    def quality(w, times):
        a=np.asarray(data.W[times][:,src,dst],dtype=float)
        b=np.maximum(np.asarray(w[times][:,src,dst],dtype=float),1e-12)
        return {'MAE':float(np.abs(a-b).mean()),
                'KL_per_node':float(np.mean(np.sum(a*np.log((a+1e-12)/b),axis=1)/n))}
    graph_metrics={**{'train_'+k:v for k,v in quality(W,fit_id).items()},
                   **{'test_'+k:v for k,v in quality(W,test_id).items()}}
    return W,graph_metrics,split




def fit_graph_from_outcomes(data,R,eps_hat,tau=0.5,epochs=180,seed=321,
                            hidden=24,node_dim=12,anchor_strength=0.10):
    """EXPLORATORY alternative: learn W from the outcome pinball loss.

    Does NOT use simulated true W as labels. Uses an optional KL anchor to
    the original Z-based kernel and node-specific structural/nuisance heads.
    Because W and Gamma are jointly unknown, the result need not identify
    either. The estimated W may merely improve fit. No causal claims.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    n,t=data.Z.shape
    split=max(10,int(0.7*t))
    nvalid=max(4,int(0.15*split))
    fit_id=np.arange(1,split-nvalid)
    val_id=np.arange(max(1,split-nvalid),split)
    test_id=np.arange(split,t)

    node_raw=np.stack([data.Z.T,data.Xz.T],axis=-1)
    mean=node_raw[:split].reshape(-1,2).mean(axis=0)
    sd=np.maximum(node_raw[:split].reshape(-1,2).std(axis=0),1e-5)
    feat=torch.tensor((node_raw-mean)/sd,dtype=torch.float32)
    z=torch.tensor(data.Z.T,dtype=torch.float32)
    xz=torch.tensor(data.Xz.T,dtype=torch.float32)
    y=torch.tensor(data.Y.T,dtype=torch.float32)
    ylag=torch.tensor(data.Ylag.T,dtype=torch.float32)
    xy=torch.tensor(data.Xy.T,dtype=torch.float32)
    r=torch.tensor(R,dtype=torch.float32)
    eps=torch.tensor(eps_hat.T,dtype=torch.float32)
    graph=DynamicGraphAttention(data.mask,data.dspace,node_dim=node_dim,hidden=hidden)
    src,dst=graph.src,graph.dst
    src_batch=src[None,:].expand(t,-1)
    kernel=known_kernel_weights(data.Z,data.mask)
    anchor=torch.tensor(kernel[:,src.numpy(),dst.numpy()],dtype=torch.float32)

    # Coefficients for preliminary differentiable quantile fitting.
    # The final node-specific coefficients are refitted by exact LP after W learning.
    coef=nn.Parameter(torch.tensor(np.column_stack([
                np.full(n,0.25),np.full(n,0.35),np.full(n,-1.0),
                np.full(n,0.15),np.full(n,0.8),np.zeros(n)]),dtype=torch.float32))
    delta=nn.Parameter(torch.zeros(n,R.shape[1],dtype=torch.float32))
    opt=torch.optim.Adam(list(graph.parameters())+[coef,delta],lr=0.009,weight_decay=1e-5)
    best_loss=float('inf')
    best_state=None
    wait=0

    def get_loss(indices,p):
        # Weighted contemporaneous and lagged neighboring outcomes.
        wy=torch.zeros((t,n),dtype=torch.float32)
        wy=wy.scatter_add(1,src_batch,p*y[:,dst])
        wlag=torch.zeros_like(wy)
        tmp=torch.zeros((t-1,n),dtype=torch.float32)
        tmp=tmp.scatter_add(1,src_batch[:-1],p[:-1]*y[:-1,dst])
        wlag=torch.cat([wlag[:1],tmp],dim=0)
        q=(coef[:,0][None,:]*wy + coef[:,1][None,:]*ylag
           +coef[:,2][None,:]*xy + coef[:,3][None,:]*wlag
           +coef[:,4][None,:]*eps + coef[:,5][None,:]
           +r@delta.T)
        resid=y[indices]-q[indices]
        pin=torch.maximum(tau*resid,(tau-1.0)*resid).mean()
        anch=(anchor[indices]*(torch.log(anchor[indices].clamp_min(1e-12))-
                              torch.log(p[indices].clamp_min(1e-12)))).sum(dim=1).mean()/n
        # Small ridge to stabilize a highly underidentified joint problem.
        ridge=0.0005*(coef.square().mean()+delta.square().mean())
        return pin+anchor_strength*anch+ridge

    for epoch in range(epochs):
        graph.train()
        prob=graph(feat,z,xz)
        loss=get_loss(fit_id,prob)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(graph.parameters())+[coef,delta],4.0)
        opt.step()
        graph.eval()
        with torch.no_grad():
            value=get_loss(val_id,graph(feat,z,xz)).item()
        if value<best_loss-1e-6:
            best_loss=value
            best_state=({k:v.detach().clone() for k,v in graph.state_dict().items()},
                        coef.detach().clone(),delta.detach().clone())
            wait=0
        else:
            wait+=1
        if epoch>=35 and wait>=35:
            break
    graph.load_state_dict(best_state[0])
    graph.eval()
    with torch.no_grad():
        pred=graph(feat,z,xz).numpy()
    W=np.zeros_like(data.W)
    W[:,src.numpy(),dst.numpy()]=pred

    # True W is used ONLY for held-out diagnostics, not for optimization.
    a=data.W[test_id][:,src.numpy(),dst.numpy()]
    b=np.maximum(pred[test_id],1e-12)
    info={'train_mode':'outcome',
          'test_MAE':float(np.abs(a-b).mean()),
          'test_KL_per_node':float(np.mean(np.sum(a*np.log((a+1e-12)/b),axis=1)/n)),
          'train_KL_per_node':float('nan')}
    return W,info,split


def run_one(n,t,tau,delta,m,seed,auxiliary_rule,network_rule,proxy,lags,
            epochs,hidden,node_dim,folds,gap,burn,train_mode,anchor_strength):
    data=simulate_panel(n=n,t=t,m=m,tau=tau,delta0=delta,burn=burn,seed=seed,
                        auxiliary_rule=auxiliary_rule,network_rule=network_rule)
    R,p=cce_design(data,proxy,lags)
    eps_hat=ols_control_function(data,R,folds=folds,gap=gap)
    if train_mode=='supervised':
        learned,quality,split=fit_graph_attention(data,epochs=epochs,
                           seed=seed+19,hidden=hidden,node_dim=node_dim)
    else:
        learned,quality,split=fit_graph_from_outcomes(
            data,R,eps_hat,tau=tau,epochs=epochs,seed=seed+19,
            hidden=hidden,node_dim=node_dim,anchor_strength=anchor_strength)
    kernel=known_kernel_weights(data.Z,data.mask)
    network_options={'True_W_oracle':data.W,
                     'Known_kernel_reference':kernel,
                     'GNN_attention_W':learned}
    estimates={}
    rows=[]
    for method,W in network_options.items():
        est=outcome_qr(data,R,eps_hat,W,tau=tau,lags=p)
        estimates[method]=est
        row={'N':n,'T':t,'tau':tau,'delta0':delta,'rep_seed':seed,
             'network_rule':network_rule,'auxiliary_rule':auxiliary_rule,
             'graph_training':train_mode,'proxy':proxy,'lags':p,'Method':method,
             'train_KL_W':quality['train_KL_per_node'],
             'test_KL_W':quality['test_KL_per_node'],
             'test_MAE_W':quality['test_MAE']}
        row.update(error_summary(est,data.truth))
        rows.append(row)
    return rows,estimates,data,learned,kernel,quality,split


def make_figure(data,Wnew,Wref,split,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    idx=min(data.Y.shape[1]-1,split+3)
    fig,axs=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    vmax=max(float(np.max(data.W[idx])),float(np.max(Wnew[idx])))
    for ax,w,title in zip(axs,[data.W,Wref,Wnew],
                           ['True W (simulation)','Kernel W','GNN attention W']):
        im=ax.imshow(w[idx],vmin=0,vmax=vmax,cmap='viridis',interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('Neighbor j')
        ax.set_ylabel('Node i')
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=0.75,label='Row-normalized weight')
    fig.savefig(output,dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n',type=int,default=30)
    p.add_argument('--t',type=int,default=65)
    p.add_argument('--n-grid',type=int,nargs='+')
    p.add_argument('--t-grid',type=int,nargs='+')
    p.add_argument('--replications',type=int,default=1)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--tau',type=float,default=0.5)
    p.add_argument('--delta',type=float,default=0.8)
    p.add_argument('--factors',type=int,default=2)
    p.add_argument('--network-rule',choices=['kernel','nonlinear'],default='nonlinear')
    p.add_argument('--auxiliary-rule',choices=['linear','nonlinear'],default='linear')
    p.add_argument('--proxy',choices=['rbar','rbar_new','sbar','sbar_new'],default='sbar_new')
    p.add_argument('--lags',default='0')
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--gap',type=int,default=1)
    p.add_argument('--burn',type=int,default=100)
    p.add_argument('--epochs',type=int,default=180)
    p.add_argument('--hidden',type=int,default=24)
    p.add_argument('--node-dim',type=int,default=12)
    p.add_argument('--train-mode',choices=['supervised','outcome'],default='supervised',
                   help='supervised requires observed network labels; outcome is exploratory')
    p.add_argument('--anchor-strength',type=float,default=0.10,
                   help='KL penalty to known kernel in exploratory outcome training')
    p.add_argument('--outdir',default='results_graph_attention')
    args=p.parse_args()
    out=Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)
    rec=[]
    for n in (args.n_grid or [args.n]):
        for t in (args.t_grid or [args.t]):
            for rep in range(args.replications):
                print(f'GNN attention: N={n}, T={t}, rep={rep+1}/{args.replications}',flush=True)
                seed=args.seed+10_000*n+100*t+rep
                rows,estimates,data,learned,kernel,quality,split=run_one(
                    n,t,args.tau,args.delta,args.factors,seed,args.auxiliary_rule,
                    args.network_rule,args.proxy,args.lags,args.epochs,
                    args.hidden,args.node_dim,args.folds,args.gap,args.burn,
                    args.train_mode,args.anchor_strength)
                rec.extend(rows)
                if rep==0:
                    info={'node':np.arange(n)}
                    for param in PARAMETERS:
                        info[f'true_{param}']=data.truth[param]
                        for method in estimates:
                            info[f'{method}_{param}']=estimates[method][param]
                    pd.DataFrame(info).to_csv(out/f'node_estimates_N{n}_T{t}.csv',index=False)
                    # Save a few snapshots, not a very large N x N x T tensor.
                    pos=np.array([0,split,min(t-1,split+3),t-1])
                    np.savez_compressed(out/f'W_snapshots_N{n}_T{t}.npz',
                      times=pos,W_true=data.W[pos],W_kernel=kernel[pos],
                      W_learned=learned[pos],mask=data.mask,Z=data.Z[:,pos])
                    make_figure(data,learned,kernel,split,out/f'W_comparison_N{n}_T{t}.png')
                    print('Graph held-out diagnostics:',quality)
    df=pd.DataFrame(rec)
    df.to_csv(out/'replication_results.csv',index=False)
    summary=aggregate_repetitions(df,['N','T','tau','delta0','network_rule',
                                       'auxiliary_rule','graph_training','proxy','lags','Method'])
    summary.to_csv(out/'monte_carlo_summary.csv',index=False)
    print('\n',summary[['N','T','Method','AbsBias_Gamma','RMSE_Gamma',
                         'AbsBias_beta1','RMSE_beta1']].to_string(index=False))
    print('Saved outputs in:',out.resolve())

if __name__=='__main__':
    main()
