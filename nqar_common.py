"""Shared DGP and econometric routines for the two NQAR neural experiments.

The starting design is the user's October MATLAB code and November 2026 manuscript.
Variables are stored as node x time (N,T), except W: time x node x node.
These are simulation examples, not established structural inference procedures.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import solve
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, eye as sparse_eye
from scipy.stats import norm

PARAMETERS = ('Gamma', 'lambda', 'rho', 'beta1', 'delta')

@dataclass
class Simulation:
    Y: np.ndarray
    Xy: np.ndarray
    Xz: np.ndarray
    Z: np.ndarray
    W: np.ndarray  # (T,N,N) TRUE weight matrices
    mask: np.ndarray
    dspace: np.ndarray
    Ylag: np.ndarray
    eps_true: np.ndarray
    factors: np.ndarray
    truth: dict
    network_rule: str


def lattice_mask(n: int, d0: float = 1.25, dc: float = 1.75):
    """Expanding 2-D lattice, fixed spatial L-infinity cutoff."""
    if n < 4:
        raise ValueError('Use N>=4 for the lattice network.')
    ncols = math.ceil(math.sqrt(n))
    xy = np.array([((i // ncols) * d0, (i % ncols) * d0)
                   for i in range(n)], dtype=float)
    dist = np.abs(xy[:, None, :] - xy[None, :, :]).max(axis=2)
    mask = (dist <= dc) & (dist > 0)
    if not np.all(mask.sum(axis=1) > 0):
        raise ValueError('Some nodes have no neighbors; increase dc.')
    return mask, dist


def known_kernel_weights(Z: np.ndarray, mask: np.ndarray, eta: float = 1.0):
    """W[t,i,j] proportional to exp(-eta*|z_it-z_jt|), within mask."""
    zt = Z.T  # T,N
    diff = np.abs(zt[:, :, None] - zt[:, None, :])
    raw = np.exp(-eta * diff) * mask[None, :, :]
    return raw / raw.sum(axis=2, keepdims=True)


def nonlinear_weights(Z: np.ndarray, Xz: np.ndarray, mask: np.ndarray,
                      eta: float = 1.0):
    """An alternative, nonlinear, time-varying true network for simulation."""
    zt, xt = Z.T, Xz.T
    dz = np.abs(zt[:, :, None] - zt[:, None, :])
    dx = np.abs(xt[:, :, None] - xt[:, None, :])
    sz = zt[:, :, None] + zt[:, None, :]
    score = -eta * dz - 0.65 * dx + 0.40 * np.sin(sz)
    score = np.where(mask[None], score, -1e9)
    score -= score.max(axis=2, keepdims=True)
    p = np.exp(score) * mask[None]
    return p / p.sum(axis=2, keepdims=True)


def simulate_panel(n=50, t=80, m=2, tau=0.5, delta0=0.8, burn=100,
                   seed=123, auxiliary_rule='linear', network_rule='kernel'):
    """Generate Xy, Xz, Z, endogenous W, and simultaneous Y.

    Z includes eps; outcome includes delta*eps, so the network is endogenous.
    The contemporaneous spatial outcome is solved, not approximated by lagged Y.
    """
    if not (0 < tau < 1):
        raise ValueError('tau must lie in (0,1)')
    if m not in (1, 2, 3):
        raise ValueError('This example supports m=1,2,3.')
    rng = np.random.default_rng(seed)
    length = t + burn
    mask, dspace = lattice_mask(n)
    f = np.zeros((length, m))
    v = np.zeros((n, length))
    xz = np.zeros((n, length))
    eps = np.zeros((n, length))
    for k in range(1, length):
        f[k] = 0.70 * f[k-1] + np.sqrt(1-0.70**2) * rng.normal(size=m)
        v[:, k] = 0.80 * v[:, k-1] + np.sqrt(1-0.80**2)*rng.normal(size=n)
        xz[:, k] = 0.40*xz[:, k-1] + np.sqrt(1-0.40**2)*rng.normal(size=n)
        eps[:, k] = 0.30*eps[:, k-1] + np.sqrt(1-0.30**2)*rng.normal(size=n)
    gx = 0.5 + rng.normal(size=(n,m))
    gy = 0.5 + rng.normal(size=(n,m))
    gz = 0.5 + rng.normal(size=(n,m))
    beta_z = 0.8 + 0.10*rng.normal(size=n)
    xy = gx @ f.T + v
    z = beta_z[:, None]*xz + gz @ f.T + eps
    if auxiliary_rule == 'nonlinear':
        # Factor-dependent nonlinear nuisance; no access to f during estimation.
        nuisance = 0.55*np.sin(f[:, 0])
        if m > 1:
            nuisance += 0.35*f[:, 0]*f[:, 1]
        node_load = 0.7 + 0.1*rng.normal(size=n)
        z += node_load[:, None]*nuisance[None, :]
    elif auxiliary_rule != 'linear':
        raise ValueError('auxiliary_rule must be linear or nonlinear')
    if network_rule == 'kernel':
        w_full = known_kernel_weights(z, mask)
    elif network_rule == 'nonlinear':
        w_full = nonlinear_weights(z, xz, mask)
    else:
        raise ValueError('network_rule must be kernel or nonlinear')

    def coeff(center):
        draw = 0.04 * rng.uniform(-1,1,size=n)
        return center + draw - draw.mean()
    Gamma, lam, rho = coeff(0.25), coeff(0.35), coeff(0.15)
    delt = np.zeros(n) if abs(delta0) < 1e-12 else coeff(delta0)
    beta = -1.0 + 0.01*rng.normal(size=n)
    u = rng.normal(size=(n,length)) - norm.ppf(tau)
    y = np.zeros((n,length),dtype=float)
    for k in range(length):
        prev = y[:,k-1] if k > 0 else np.zeros(n)
        wprev = w_full[k-1] @ prev if k > 0 else np.zeros(n)
        rhs = lam*prev+rho*wprev+beta*xy[:,k]+gy@f[k]+delt*eps[:,k]+u[:,k]
        a = np.eye(n)-Gamma[:,None]*w_full[k]
        y[:,k] = solve(a,rhs,assume_a='gen',check_finite=False)
    keep = slice(burn,length)
    Y, W, Z = y[:,keep], w_full[keep], z[:,keep]
    Ylag = y[:, burn-1:length-1]
    return Simulation(Y=Y, Xy=xy[:,keep], Xz=xz[:,keep], Z=Z,
           W=W, mask=mask, dspace=dspace, Ylag=Ylag,
           eps_true=eps[:,keep], factors=f[keep],
           truth={'Gamma':Gamma,'lambda':lam,'rho':rho,'beta1':beta,'delta':delt},
           network_rule=network_rule)


def cce_design(data: Simulation, proxy='sbar_new', lags=0):
    """Contemporaneous cross-sectional averages and optionally their lag blocks.

    The active MATLAB routine uses proxy sbar_new=(xbar_y,zbar), lags=0,
    even though comments/README describe other bases and increasing pT.
    """
    avg = {'ybar':data.Y.mean(axis=0), 'xybar':data.Xy.mean(axis=0),
           'zbar':data.Z.mean(axis=0), 'xzbar':data.Xz.mean(axis=0)}
    choices = {
        'rbar':('ybar','xybar','zbar','xzbar'),
        'rbar_new':('ybar','xybar','zbar'),
        'sbar':('xybar','zbar','xzbar'),
        'sbar_new':('xybar','zbar'),
    }
    if proxy not in choices:
        raise ValueError(f'Unknown proxy {proxy}: {list(choices)}')
    base = np.column_stack([avg[key] for key in choices[proxy]])
    if lags == 'auto':
        lags = int(math.ceil(data.Y.shape[1] ** (1/3)))
    lags = int(lags)
    if lags < 0 or lags >= data.Y.shape[1]-6:
        raise ValueError('Invalid lag order relative to T.')
    blocks = []
    for lag in range(lags+1):
        block = np.zeros_like(base)
        if lag == 0:
            block[:] = base
        else:
            block[lag:] = base[:-lag]
        blocks.append(block)
    return np.column_stack(blocks), lags


def ols_control_function(data: Simulation, R: np.ndarray, folds=3, gap=1):
    """Out-of-fold unit-specific CCE-OLS auxiliary residuals."""
    n,t = data.Z.shape
    if folds < 2 or folds > t//4:
        raise ValueError('Use 2<=folds<=T//4 for cross-fitting.')
    out = np.zeros_like(data.Z)
    for test_id in np.array_split(np.arange(t),folds):
        excluded = np.zeros(t,dtype=bool)
        for k in test_id:
            excluded[max(0,k-gap):min(t,k+gap+1)] = True
        train = np.flatnonzero(~excluded)
        if len(train) <= R.shape[1]+4:
            raise ValueError('Too few training times for OLS; lower lags/folds/gap.')
        for i in range(n):
            X = np.column_stack([np.ones(t),data.Xz[i],R])
            co = np.linalg.lstsq(X[train],data.Z[i,train],rcond=1e-9)[0]
            out[i,test_id] = data.Z[i,test_id]-X[test_id]@co
    return out


def quantile_lp(X: np.ndarray, y: np.ndarray, tau: float):
    """Exact pinball-loss quantile regression using SciPy HiGHS linear programming."""
    X=np.asarray(X,dtype=float)
    y=np.asarray(y,dtype=float)
    n,k=X.shape
    if n < k+3:
        raise ValueError(f'Too many regressors: sample size {n}, columns {k}.')
    scale=np.maximum(np.max(np.abs(X),axis=0),1e-10)
    Xs=X/scale
    # y = Xb + u_plus - u_minus. Cost tau*u_plus+(1-tau)*u_minus.
    c=np.concatenate((np.zeros(k),np.full(n,tau),np.full(n,1-tau)))
    A=hstack([csr_matrix(Xs),sparse_eye(n,format='csr'),
              -sparse_eye(n,format='csr')],format='csr')
    bounds=[(None,None)]*k+[(0,None)]*(2*n)
    ans=linprog(c,A_eq=A,b_eq=y,bounds=bounds,method='highs')
    if not ans.success:
        raise RuntimeError(f'QR LP failed: {ans.message}')
    return ans.x[:k]/scale


def outcome_qr(data: Simulation, R: np.ndarray, eps_hat: np.ndarray,
               W: np.ndarray, tau=0.5, lags=0):
    """Node-wise CFQR--CCE with (possibly learned) time-varying W.

    Outcome design: [intercept, W_tY_t, Y_{i,t-1}, X^y_it,
                     W_{t-1}Y_{t-1}, CCE terms, eps_hat_it].
    Discard t=0 so every lagged network regressor uses W_{t-1} available.
    """
    n,t=data.Y.shape
    assert W.shape == (t,n,n)
    wy=np.einsum('tij,jt->it',W,data.Y,optimize=True)
    wlag=np.zeros_like(wy)
    wlag[:,1:]=np.einsum('tij,jt->it',W[:-1],data.Y[:,:-1],optimize=True)
    start=max(1,int(lags))
    result={name:np.zeros(n) for name in PARAMETERS}
    for i in range(n):
        X=np.column_stack((np.ones(t),wy[i],data.Ylag[i],data.Xy[i],
                           wlag[i],R,eps_hat[i]))
        co=quantile_lp(X[start:],data.Y[i,start:],tau)
        result['Gamma'][i]=co[1]
        result['lambda'][i]=co[2]
        result['beta1'][i]=co[3]
        result['rho'][i]=co[4]
        result['delta'][i]=co[-1]
    return result


def error_summary(est: dict, truth: dict):
    out={}
    for name in PARAMETERS:
        e=np.asarray(est[name])-np.asarray(truth[name])
        out['Bias_'+name]=float(e.mean())
        out['AbsBias_'+name]=float(np.abs(e).mean())
        out['MSE_'+name]=float(np.mean(e**2))
        out['RMSE_'+name]=float(np.sqrt(np.mean(e**2)))
    return out


def aggregate_repetitions(df, group_cols):
    """MC RMSE: sqrt(E_s[mean_i error^2]), not E_s[sqrt(MSE_s)]."""
    import pandas as pd
    rows=[]
    for key,block in df.groupby(group_cols,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        row=dict(zip(group_cols,key))
        row['replications']=len(block)
        for name in PARAMETERS:
            for metric in ['Bias','AbsBias','MSE']:
                row[f'{metric}_{name}']=block[f'{metric}_{name}'].mean()
            row[f'RMSE_{name}']=np.sqrt(row[f'MSE_{name}'])
        rows.append(row)
    return pd.DataFrame(rows)
