# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, math, argparse
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

def clamp(x, lo, hi): return max(lo, min(hi, x))
def _ema(arr, length):
    if length <= 1: return np.array(arr, dtype=float)
    alpha = 2.0/(length+1.0); out = np.full(len(arr), np.nan, float); s = np.nan
    for i,x in enumerate(arr):
        if np.isnan(x): out[i] = np.nan if np.isnan(s) else s; continue
        s = x if np.isnan(s) else (alpha*x + (1.0-alpha)*s); out[i]=s
    return out
def _vema(src, Ls):
    out = np.full(len(src), np.nan, float); s=np.nan
    for i,x in enumerate(src):
        L = max(1.0, float(Ls[i])); alpha=2.0/(L+1.0)
        if np.isnan(x): out[i]=np.nan if np.isnan(s) else s; continue
        s = x if np.isnan(s) else (alpha*x + (1.0-alpha)*s); out[i]=s
    return out
def _sma(arr, L): return pd.Series(arr).rolling(L, min_periods=L).mean().to_numpy()
def _stdev(arr, L): return pd.Series(arr).rolling(L, min_periods=L).std(ddof=0).to_numpy()
def _percent_rank(arr, L):
    s=pd.Series(arr); out=np.full(len(arr), np.nan, float)
    for i in range(len(arr)):
        start=max(0,i-L+1); w=s.iloc[start:i+1]
        if len(w)<L: out[i]=np.nan; continue
        out[i]=float(w.rank(pct=True).iloc[-1]-1e-12)
    return out
def _rolling_quantile(arr, L, q): return pd.Series(arr).rolling(L, min_periods=L).quantile(q).to_numpy()
def tanh_curve(x): 
    e2x=math.exp(2*x); return (e2x-1.0)/(e2x+1.0)
def hurst_simple(close, window=50):
    ret=np.log(close/np.roll(close,1)); ret[0]=0.0; H=np.zeros_like(ret)
    for i in range(len(ret)):
        if i<window: H[i]=0.5; continue
        r=ret[i-window+1:i+1]; m=np.mean(r); c=0.0; mx=-1e9; mn=1e9
        for x in r: c+=x-m; mx=max(mx,c); mn=min(mn,c)
        R=mx-mn; S=np.std(r); H[i]=0.5 if (R<=0 or S<=0) else max(0,min(1, math.log(R/S)/math.log(window*0.5)))
    return H
def entropy5(ret, lb=100):
    e=np.zeros_like(ret)
    for i in range(len(ret)):
        if i<lb: e[i]=0.5; continue
        w=ret[i-lb+1:i+1]; lo=np.nanmin(w); hi=np.nanmax(w); rng=hi-lo
        if rng<=0 or np.isnan(rng): e[i]=0.5; continue
        bins=np.zeros(5); 
        for x in w:
            if np.isnan(x): continue
            idx=int(min(4,max(0,(x-lo)/rng*5.0))); bins[idx]+=1.0
        en=0.0; total=float(lb)
        for b in bins:
            p=b/total
            if p>0: en-=p*math.log(p)
        e[i]=en/math.log(5.0)
    return e

@dataclass
class Profile:
    name:str="Base"; kp_mult:float=1.0; ki_mult:float=1.0; kd_mult:float=1.0; sigma:float=0.06; bias:float=1.00; cd:float=1.00; inv_mult:float=1.10
    @staticmethod
    def from_name(name:str):
        if name=="Aggressive": return Profile("Aggressive",1.20,1.00,1.25,0.05,1.05,0.95,1.122)
        if name=="Defensive": return Profile("Defensive",0.85,1.15,0.90,0.08,0.95,1.05,1.078)
        return Profile("Base",1.0,1.0,1.0,0.06,1.00,1.00,1.10)

@dataclass
class Flags:
    FF_RV:bool=True; FF_PID:bool=True; FF_SANDBOX:bool=True; FF_RV_NL:bool=True
    FF_PID_NORM_TELEM:bool=True; FF_BUDGET_STRICT:bool=True; FF_PROFILE:bool=True; FF_PID_ADAPTIVE_GAINS:bool=True
    FF_PID_I_DECAY:bool=False; FF_PID_I_DRAIN_MIDRV:bool=False; FF_RV_AWARE_GATES:bool=False

@dataclass
class Params:
    RISK_OFF_ENTER:float=0.95; RISK_OFF_EXIT:float=0.85; ATTEN_FLOOR:float=0.60; REL_VALVE_ENABLED:bool=True; FUSE_BARS:int=40
    AC_MIN_ATR_Z:float=0.20; AC_SIDE_GAP_ATR:float=0.50; AC_BURST_LEN:int=12; AC_MAX_BURSTS:int=3; NEUTRAL_LO:float=0.48; NEUTRAL_HI:float=0.52
    CD_FLIP_THRESH:int=4; NEUTRAL_SIGMA_BASE:float=0.06; PID_ERR_LEN:int=96; PID_KP_BASE:float=0.45; PID_KI_BASE:float=0.03; PID_KD_BASE:float=0.15; PID_IMAX:float=0.60
    HURST_HI:float=0.55; HURST_LO:float=0.45; RV_SMOOTH_EMA:int=3; I_DECAY_FACTOR:float=0.85; I_DRAIN_RATE:float=0.02

def load_ohlcv(path): 
    df=pd.read_csv(path)
    try: ts=pd.to_datetime(df["timestamp"], unit="ms")
    except: ts=pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"]=ts; df=df.sort_values("timestamp").reset_index(drop=True)
    return df

def compute_all(df, flags, profile, pars, build_tag, symbol, tf, collect_alert_samples=5):
    close=df["close"].to_numpy(float); high=df["high"].to_numpy(float); low=df["low"].to_numpy(float); openp=df["open"].to_numpy(float); n=len(df)
    ret=np.log(close/np.roll(close,1)); ret[0]=0.0
    H_raw=hurst_simple(close,50) if flags.FF_RV else np.full(n,0.5)
    E_raw=entropy5(ret,100) if flags.FF_RV else np.full(n,0.5)
    E_smooth=_ema(E_raw,5)
    def wins(x,L=256,loq=0.05,hiq=0.95):
        lo=_rolling_quantile(x,L,loq); hi=_rolling_quantile(x,L,hiq); out=x.copy()
        for i in range(n):
            if np.isnan(lo[i]) or np.isnan(hi[i]): out[i]=x[i]
            else: out[i]=min(max(x[i],lo[i]),hi[i])
        return out
    H=wins(H_raw); E=wins(E_smooth)
    mu=_sma(ret,100); sd=_stdev(ret,100); sd=np.where(sd<=1e-8,1e-8,sd); z=(ret-mu)/sd
    K_raw=_ema(np.abs(z),5) if flags.FF_RV else np.zeros(n)
    K_pr=_percent_rank(K_raw,200); K_w=wins(K_pr); K=np.maximum(K_w,0.0)
    wH,wE,wK=0.40,0.35,0.25; s=wH+wE+wK; wH_n,wE_n,wK_n=wH/s,wE/s,wK/s
    Hn=np.clip((H-0.30)/0.40,0,1); En=np.clip(E,0,1); Kn=np.clip(K,0,1)
    RV_raw=(wH_n*Hn + wE_n*(1.0-En) + wK_n*(1.0-Kn)) if flags.FF_RV else np.full(n,0.5)
    RV=pd.Series(RV_raw).ewm(span=pars.RV_SMOOTH_EMA, adjust=False).mean().to_numpy()
    RVs=pd.Series(RV_raw).ewm(span=3, adjust=False).mean().to_numpy()
    rv_fast=pd.Series(RV).ewm(span=3, adjust=False).mean().to_numpy(); rv_slow=pd.Series(RV).ewm(span=21, adjust=False).mean().to_numpy(); rv_mom=rv_fast-rv_slow
    rv_low=RVs<=0.40; rv_mid=(RVs>0.40)&(RVs<0.65); rv_high=RVs>=0.65
    rv_band=np.where(rv_low,"low",np.where(rv_mid,"mid","high"))
    ro_bars=np.zeros(n,int); rn_bars=np.zeros(n,int); risk_off_state=np.zeros(n,bool); riskoff_fuse=np.zeros(n,int)
    enter,exit=pars.RISK_OFF_ENTER, pars.RISK_OFF_EXIT
    for i in range(n):
        if RV[i]>enter: ro_bars[i]=(ro_bars[i-1]+1) if i>0 else 1; rn_bars[i]=0
        elif RV[i]<exit or (rv_mom[i]<0): rn_bars[i]=(rn_bars[i-1]+1) if i>0 else 1; ro_bars[i]=0
        else: ro_bars[i]=0; rn_bars[i]=0
        prev=risk_off_state[i-1] if i>0 else False; state=prev; fuse=max(riskoff_fuse[i-1]-1,0) if i>0 else 0
        if (not prev) and ro_bars[i]>=2: state=True; fuse=pars.FUSE_BARS
        elif prev:
            early=pars.REL_VALVE_ENABLED and fuse==0 and (RV[i]< (enter-0.07)) and (rv_mom[i]<-0.04)
            if rn_bars[i]>=2 or early: state=False; fuse=0
        risk_off_state[i]=state; riskoff_fuse[i]=fuse
    rvd=np.abs(RVs-0.5); sigma=profile.sigma; neutral_strength=np.exp(-(rvd*rvd)/(2.0*sigma*sigma))
    global_damp=pars.ATTEN_FLOOR + (1.0-pars.ATTEN_FLOOR)*(1.0 - neutral_strength)
    def _atr(h,l,c,L=14):
        tr=np.maximum.reduce([h-l, np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))]); tr[0]=h[0]-l[0]
        return _ema(tr,L)
    atr14=_atr(high,low,close,14); atr_mu50=_sma(atr14,50); atr_sd50=_stdev(atr14,50); atr_sd50=np.where(atr_sd50<=1e-8,1e-8,atr_sd50)
    atr_z=(atr14-atr_mu50)/atr_sd50; atr_ratio=atr14/np.maximum(atr_mu50,1e-8)
    def fmap(rv,lo,hi): return lo + rv*(hi-lo)
    len_pt_base=fmap(RVs,14.0,34.0); len_ma_base=fmap(RVs,9.0,26.0)
    len_pt_eff=np.round(np.clip(len_pt_base,14,34)); len_ma_eff=np.round(np.clip(len_ma_base,9,26))
    rvn=(RVs-0.5)*4.0
    mult_pt_eff= (0.7 + 0.7*np.array([0.5*(1.0+math.tanh(x)) for x in rvn])) if (flags.FF_RV_NL) else fmap(RVs,0.7,1.4)
    mult_pt_eff=np.clip(mult_pt_eff,0.7,1.4)
    vema_pt=_vema(close,len_pt_eff); vema_ma=_vema(close,len_ma_eff); ema_pt=vema_pt
    slope_pt=(ema_pt-np.roll(ema_pt,1))/np.maximum(atr14,1e-8); slope_pt[0]=0.0
    macd_raw=_ema(close,12)-_ema(close,26); mom_ma=(macd_raw-np.roll(macd_raw,1))/np.maximum(atr14,1e-8); mom_ma[0]=0.0
    lt_line=_ema(close,55); trend_lt=np.sign(ema_pt-lt_line)
    ema_ma=vema_ma; dev_ema=(close-ema_ma)/np.maximum(atr14,1e-8)
    w_pt,w_ma,w_lt,w_xv,w_dv=0.40,0.30,0.15,0.10,0.05; s1=w_pt+w_ma+w_lt; s2=w_xv+w_dv
    w_pt_n,w_ma_n,w_lt_n=w_pt/s1,w_ma/s1,w_lt/s1; w_xv_n,w_dv_n=(w_xv/s2 if s2>0 else 0.0),(w_dv/s2 if s2>0 else 0.0)
    cts_raw=(w_pt_n*slope_pt + w_ma_n*mom_ma + w_lt_n*trend_lt - w_xv_n*np.abs(atr_z) - w_dv_n*np.abs(dev_ema))
    cts_ema=_ema(cts_raw,3); cts_final=cts_ema*global_damp
    def sign(x): return int(x>0) - int(x<0)
    sgn=np.array([sign(x) for x in cts_final]); flip=np.zeros(n,bool)
    for i in range(1,n):
        if sgn[i]!=0 and sgn[i-1]!=0 and sgn[i]!=sgn[i-1]: flip[i]=True
    mag=np.minimum(np.abs(rv_mom),0.10); CD_WIN_base=np.round(55.0 - (mag/0.10)*(55.0-21.0))
    cd_mult_raw=np.where((flags.FF_RV_AWARE_GATES & rv_mid & (~risk_off_state)), 1.50*profile.cd, 1.0)
    cd_mult=np.clip(cd_mult_raw,1.00,1.75); CD_WIN_eff=np.round(np.clip(CD_WIN_base*cd_mult,21,89))
    flip_cnt=np.zeros(n,int)
    for i in range(n):
        win=int(CD_WIN_eff[i]) if np.isfinite(CD_WIN_eff[i]) else 21; start=max(0,i-win+1); flip_cnt[i]=int(np.sum(flip[start:i+1]))
    cooldown_state=np.zeros(n,bool); cd_bars=np.zeros(n,int)
    for i in range(n):
        prev=cooldown_state[i-1] if i>0 else False; cdb=cd_bars[i-1] if i>0 else 0; state=prev
        if flip_cnt[i]>=4: state=True; cdb=13
        elif prev: cdb=max(cdb-1,0); 
        if prev and cdb==0: state=False
        if state and (rv_mom[i]<-0.05) and True: state=False; cdb=0
        cooldown_state[i]=state; cd_bars[i]=cdb
    ema20=_ema(close,20); atrSafe=np.maximum(atr14,1e-8); ext=(close-ema20)/atrSafe; ext_abs=np.abs(ext)
    side_gap_ok_long=(ema20-low)>=0.50*atr14; side_gap_ok_short=(high-ema20)>=0.50*atr14
    gate_bias_raw=np.where((flags.FF_RV_AWARE_GATES & rv_mid & (~risk_off_state)), 0.02*profile.bias, 0.0)
    gate_bias=np.clip(gate_bias_raw,0.00,0.03); base_gate=np.where(atr_ratio>1.30,2.2,2.0)
    gate_enter=base_gate+0.15+gate_bias; gate_exit=base_gate-0.15+0.5*gate_bias
    need_htf=np.zeros(n,bool)
    for i in range(n):
        prev=need_htf[i-1] if i>0 else False
        need_htf[i]= (not prev and ext_abs[i]>gate_enter[i]) or (prev and not (ext_abs[i]<gate_exit[i])) or (prev and (ext_abs[i]>=gate_exit[i]))
        if prev and ext_abs[i]<gate_exit[i]: need_htf[i]=False
    ema20_60m=_ema(close,80); htf_ok=np.where(cts_final>0, close>ema20_60m, close<ema20_60m)
    gate5_pass=(~need_htf) | htf_ok
    pb_long=(close<=ema20+0.5*atr14) | (close<openp); pb_short=(close>=ema20-0.5*atr14) | (close>openp)
    gate1_pass=(ext_abs <= (base_gate+gate_bias)) | (pb_long & side_gap_ok_long) | (pb_short & side_gap_ok_short)
    dcts=np.diff(cts_final, prepend=cts_final[0]); cts_pr=_percent_rank(cts_final,200)
    gate2_pass=np.where(cts_final>0, (cts_pr<=0.95)&(dcts>=0), (cts_pr>=0.05)&(dcts<=0))
    atr_mu50=_sma(atr14,50); atr_sd50=_stdev(atr14,50); atr_sd50=np.where(atr_sd50<=1e-8,1e-8,atr_sd50); atr_z=(atr14-atr_mu50)/atr_sd50
    E_raw=entropy5(np.log(close/np.roll(close,1)),100); E_s=_ema(E_raw,5); E=wins(E_s)
    jsd_high=E>=0.85; gate3_pass=~(jsd_high | (atr_z<0.20))
    gate_pass=gate1_pass & gate2_pass & gate3_pass & gate5_pass & (~cooldown_state)
    PID_KP=0.45*profile.kp_mult; PID_KI=0.03*profile.ki_mult; PID_KD=0.15*profile.kd_mult
    Kp_eff=PID_KP*(0.8+0.4*RVs); Ki_eff=PID_KI*(0.7+0.6*np.abs(RVs-0.5)*2.0); Kd_eff=PID_KD*(0.8+0.4*RVs)
    Kp_final=Kp_eff if flags.FF_PID_ADAPTIVE_GAINS else PID_KP; Ki_final=Ki_eff if flags.FF_PID_ADAPTIVE_GAINS else PID_KI; Kd_final=Kd_eff if flags.FF_PID_ADAPTIVE_GAINS else PID_KD
    EV=pd.Series(cts_final).ewm(span=96, adjust=False).mean().to_numpy(); err=cts_final-EV; err_s=_ema(err,3)
    P=np.zeros(n); I=np.zeros(n); D=np.zeros(n); i_decay_evt=np.zeros(n,int); i_drain_evt=np.zeros(n,int); P_prev=0.0
    for i in range(n):
        Pi=max(-0.60, min(0.60, Kp_final[i]*err[i])); Ii=I[i-1] if i>0 else 0.0
        if flags.FF_PID_I_DECAY and i>0 and (math.copysign(1,Pi)!=math.copysign(1,P_prev)) and (abs(Pi)>1e-6): Ii=Ii*0.85; i_decay_evt[i]=1
        P_prev=Pi; Ii=max(-0.60, min(0.60, Ii + Ki_final[i]*err[i]))
        if flags.FF_PID_I_DRAIN_MIDRV and rv_mid[i] and (not risk_off_state[i]):
            I_new=Ii*(1.0-0.02); 
            if abs(I_new-Ii)>1e-9: i_drain_evt[i]=1
            Ii=I_new
        P[i]=Pi; I[i]=Ii; D[i]=Kd_final[i]*(err_s[i] - (err_s[i-1] if i>0 else 0.0))
    rv_gate=np.where(RVs<0.40,0.25, np.where(RVs<0.55,0.60,1.00))
    rs_scale_eff=np.clip(1.0 + rv_gate*(P+I-D), 0.60, 1.40); gamma_eff=np.clip(1.0 + 0.50*rv_gate*(P+I-D), 0.70, 1.30)
    INV_WIN=288; pr95_abs=_rolling_quantile(np.abs(cts_final), INV_WIN, 0.95); inv_thr=np.maximum(1.0, pr95_abs*profile.inv_mult); inv_ok=np.abs(cts_final)<=inv_thr
    alert_armed=gate_pass & (~risk_off_state) & (~cooldown_state)
    neutral_mask=np.abs(RVs-0.5) < profile.sigma
    risk_off_pct=float(np.mean(risk_off_state))*100.0; neutral_pct=float(np.mean(neutral_mask))*100.0
    alerts=alert_armed.astype(int); rising=(alerts==1) & (np.roll(alerts,1)==0); rising[0]=alerts[0]==1; total_alerts=int(np.sum(rising))
    dt=pd.to_datetime(df["timestamp"]); days=max(1.0, (dt.iloc[-1]-dt.iloc[0]).total_seconds()/86400.0); trades_per_day=float(total_alerts)/days
    flip_rate=float(np.mean(np.array([ (sgn[i]!=0 and sgn[i-1]!=0 and sgn[i]!=sgn[i-1]) for i in range(1,len(sgn)) ])))*100.0
    mid_mask=rv_mid; abs_cts=np.abs(cts_final[mid_mask]); overshoot_p95=float(np.nanquantile(abs_cts,0.95)) if abs_cts.size else float("nan"); overshoot_med=float(np.nanmedian(abs_cts)) if abs_cts.size else float("nan")
    i_decay_rate=float(np.mean(i_decay_evt))*100.0; i_drain_rate=float(np.mean(i_drain_evt))*100.0
    blocked_gates=float(np.mean((~gate_pass) & (~risk_off_state) & (~cooldown_state)))*100.0; blocked_risk=float(np.mean(risk_off_state))*100.0; blocked_cooldown=float(np.mean(cooldown_state & (~risk_off_state)))*100.0
    samples=[]; sample_count=0
    for i in range(len(df)):
        if rising[i]:
            ts=int(pd.Timestamp(dt.iloc[i]).value//10**6)
            samples.append({"schema":"1.0.0","apex":"2.3","build": "S5-F1.1-soft-k2","mode":"indicator","symbol":symbol,
                            "tf":tf,"rv":float(RVs[i]),"rv_band":str(rv_band[i]),"cts":float(cts_final[i]),
                            "gates":{"g1":int(gate1_pass[i]),"g2":int(gate2_pass[i]),"g3":int(gate3_pass[i]),"g5":int(gate5_pass[i])},
                            "gate_bias":float(gate_bias[i]),"risk":{"state":int(risk_off_state[i]),"fuse":0},
                            "cooldown":int(cooldown_state[i]),"cooldown_mult":float(cd_mult[i]),
                            "pid":{"p":float(P[i]),"i":float(I[i]),"d":float(D[i])},
                            "i_decay_evt":int(i_decay_evt[i]),"i_drain_evt":int(i_drain_evt[i]),"reason":"notify","ts":ts})
            sample_count+=1
            if sample_count>=(collect_alert_samples or 0): break
    return {"risk_off_pct":risk_off_pct,"neutral_pct":neutral_pct,"trades_per_day":trades_per_day,"flip_rate_pct":flip_rate,
            "overshoot_med_midRV":overshoot_med,"overshoot_p95_midRV":overshoot_p95,"i_decay_rate_pct":i_decay_rate,"i_drain_rate_pct":i_drain_rate,
            "blocked_gates_pct":blocked_gates,"blocked_risk_pct":blocked_risk,"blocked_cooldown_pct":blocked_cooldown,"samples":samples}

def run_experiment(data_path, symbol, tf, build_tag, experiment="baseline", blocks=8, block_len=10000, profile_name="Base", out_dir=None):
    df=load_ohlcv(data_path); out_dir=out_dir or os.getcwd(); profile=Profile.from_name(profile_name); pars=Params()
    BASE=Flags(); DECAY=Flags(FF_PID_I_DECAY=True); DRAIN=Flags(FF_PID_I_DRAIN_MIDRV=True); RVG=Flags(FF_RV_AWARE_GATES=True)
    DECAY_DRAIN=Flags(FF_PID_I_DECAY=True, FF_PID_I_DRAIN_MIDRV=True); FULL=Flags(FF_PID_I_DECAY=True, FF_PID_I_DRAIN_MIDRV=True, FF_RV_AWARE_GATES=True)
    grids={"baseline":{"Baseline":BASE},
           "grid":{"Baseline":BASE,"DECAY":DECAY,"DRAIN":DRAIN,"RVGATES":RVG,"DECAY+DRAIN":DECAY_DRAIN,"FULL":FULL}}
    selected=grids["baseline"] if experiment=="baseline" else grids["grid"]
    n=len(df); block_begins=list(range(0,n,block_len))[:blocks]; 
    if len(block_begins)==0: block_begins=[0]
    blocks_idx=[]; 
    for s in block_begins:
        e=min(n, s+block_len)
        if e-s<1000: break
        blocks_idx.append((s,e))
    metrics=[]; samples_path=os.path.join(out_dir, f"alerts_{symbol}_{tf}_{build_tag}_{experiment}.jsonl"); sp=open(samples_path,"w",encoding="utf-8")
    for cfg_name, flags in selected.items():
        for bi,(s,e) in enumerate(blocks_idx):
            chunk=df.iloc[s:e].reset_index(drop=True)
            res=compute_all(chunk, flags, profile, pars, build_tag, symbol, tf, collect_alert_samples=5)
            row={"config":cfg_name,"block":bi+1,"start":str(chunk['timestamp'].iloc[0]),"end":str(chunk['timestamp'].iloc[-1])}
            for k,v in res.items():
                if k!="samples": row[k]=v
            metrics.append(row)
            for sm in res["samples"]:
                sm["config"]=cfg_name; sm["block"]=bi+1; sp.write(json.dumps(sm)+"\n")
    sp.close()
    mdf=pd.DataFrame(metrics); out_metrics=os.path.join(out_dir, f"metrics_{symbol}_{tf}_{build_tag}_{experiment}.csv"); mdf.to_csv(out_metrics, index=False)
    summary=mdf.groupby("config").mean(numeric_only=True).reset_index(); out_summary=os.path.join(out_dir, f"metrics_{symbol}_{tf}_{build_tag}_{experiment}_summary.csv"); summary.to_csv(out_summary, index=False)
    print(f"\n✅ Complete!")
    print(f"   Metrics: {out_metrics}")
    print(f"   Summary: {out_summary}")
    print(f"   Alerts: {samples_path}")
    return f"Metrics: {out_metrics}\nSummary: {out_summary}\nAlerts: {samples_path}"

def main():
    parser = argparse.ArgumentParser(description='ATLAS Apex S5-F1.1-soft-k2 Backtest')
    parser.add_argument('--data', required=True, help='Path to OHLCV CSV file')
    parser.add_argument('--symbol', required=True, help='Symbol name (e.g., BTCUSDT)')
    parser.add_argument('--tf', required=True, help='Timeframe (e.g., 15m)')
    parser.add_argument('--experiment', default='grid', choices=['baseline', 'grid'], help='Experiment type')
    parser.add_argument('--blocks', type=int, default=8, help='Number of blocks')
    parser.add_argument('--block_len', type=int, default=2800, help='Block length in bars')
    parser.add_argument('--profile', default='Base', choices=['Base', 'Aggressive', 'Defensive'], help='Profile name')
    parser.add_argument('--out_dir', default=None, help='Output directory (default: current directory)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"ATLAS Apex S5-F1.1-soft-k2 Backtest")
    print(f"{'='*60}")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.tf}")
    print(f"Experiment: {args.experiment}")
    print(f"Blocks: {args.blocks}")
    print(f"Block Length: {args.block_len}")
    print(f"Profile: {args.profile}")
    print(f"{'='*60}\n")
    
    run_experiment(
        data_path=args.data,
        symbol=args.symbol,
        tf=args.tf,
        build_tag="S5-F1.1-soft-k2",
        experiment=args.experiment,
        blocks=args.blocks,
        block_len=args.block_len,
        profile_name=args.profile,
        out_dir=args.out_dir
    )

if __name__ == "__main__":
    main()