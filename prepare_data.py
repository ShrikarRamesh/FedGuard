"""
FedGuard - Step 0: build federated hospital nodes from the PhysioNet/CinC 2019 sepsis data.

Download first (open access, CC-BY 4.0, no login needed):
    wget -r -N -c -np -nH --cut-dirs=3 https://physionet.org/files/challenge-2019/1.0.0/training/
  or
    aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/ ./training

Then:
    python prepare_data.py --raw ./training --out ./data

Nodes = hospital system x ICU unit  ->  A_MICU, A_SICU, B_MICU, B_SICU  (+ *_UNK if unit is missing)
This gives genuinely non-IID clients: two real hospital systems, two care-unit types.

Output per node:  data/<node>.npz  with
    X      [rows, F]  forward-filled, per-node train-standardized values (NaN -> 0 after scaling)
    M      [rows, F]  1 if the value was actually measured at that hour, else 0  (missingness is informative)
    S      [rows, 3]  static-ish: age, gender, ICULOS (scaled)
    y      [rows]     SepsisLabel (already shifted 6 h early by the challenge organisers)
    offs   [P+1]      patient boundaries into the row arrays
    split  [P]        0=train 1=val 2=test  (patient-level, stratified on 'ever septic')
    pid    [P]        patient file ids
Windows are cut lazily in the Dataset (see WindowDataset) so RAM stays small.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

DYN = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2',
       'SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose',
       'Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC',
       'Fibrinogen','Platelets']
STATIC = ['Age','Gender','ICULOS']

def node_of(df, hosp):
    u1, u2 = df['Unit1'].iloc[0], df['Unit2'].iloc[0]
    unit = 'MICU' if u1 == 1 else 'SICU' if u2 == 1 else 'UNK'
    return f'{hosp}_{unit}'

def load(raw):
    nodes = {}
    for hosp, sub in (('A','training_setA'), ('B','training_setB')):
        files = sorted(glob.glob(os.path.join(raw, sub, '*.psv')))
        print(f'hospital {hosp}: {len(files)} patients')
        for f in files:
            df = pd.read_csv(f, sep='|')
            nodes.setdefault(node_of(df, hosp), []).append((os.path.basename(f)[:-4], df))
    return nodes

def split_patients(labels, rng, frac=(0.7, 0.15, 0.15)):
    split = np.zeros(len(labels), dtype=np.int8)
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]; rng.shuffle(idx)
        n_tr, n_va = int(frac[0]*len(idx)), int(frac[1]*len(idx))
        split[idx[n_tr:n_tr+n_va]] = 1
        split[idx[n_tr+n_va:]] = 2
    return split

def build_node(pats, rng):
    raw_vals, masks, stat, ys, offs, pids = [], [], [], [], [0], []
    for pid, df in pats:
        v = df[DYN].to_numpy(np.float32)
        masks.append((~np.isnan(v)).astype(np.float32))
        raw_vals.append(pd.DataFrame(v).ffill().to_numpy(np.float32))   # causal: past values only
        stat.append(df[STATIC].to_numpy(np.float32))
        ys.append(df['SepsisLabel'].to_numpy(np.int8))
        offs.append(offs[-1] + len(df)); pids.append(pid)
    ever = np.array([y.max() for y in ys])
    split = split_patients(ever, rng)
    X, M, S, y = map(np.concatenate, (raw_vals, masks, stat, ys))
    offs = np.array(offs)
    # normalisation stats from THIS node's TRAIN patients only (no cross-site or test leakage)
    tr_rows = np.concatenate([np.arange(offs[i], offs[i+1]) for i in np.where(split == 0)[0]])
    mu, sd = np.nanmean(X[tr_rows], 0), np.nanstd(X[tr_rows], 0)
    mu = np.nan_to_num(mu); sd = np.where(np.nan_to_num(sd) < 1e-6, 1.0, sd)
    X = np.nan_to_num((X - mu) / sd).astype(np.float32)        # never-measured -> 0 (= train mean), M flags it
    S = np.nan_to_num(S)
    S[:, 0] = (S[:, 0] - 60) / 20; S[:, 2] = np.log1p(S[:, 2]) / 5
    stats = dict(patients=len(pats), septic_patients=int(ever.sum()),
                 septic_patient_rate=round(float(ever.mean()), 4), rows=int(len(y)),
                 positive_row_rate=round(float(y.mean()), 4),
                 split_counts=[int((split == k).sum()) for k in range(3)])
    return dict(X=X, M=M, S=S.astype(np.float32), y=y, offs=offs, split=split, pid=np.array(pids)), stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='./training'); ap.add_argument('--out', default='./data')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    summary = {}
    for node, pats in sorted(load(a.raw).items()):
        arrs, stats = build_node(pats, rng)
        np.savez_compressed(os.path.join(a.out, f'{node}.npz'), **arrs)
        summary[node] = stats; print(node, stats)
    json.dump(summary, open(os.path.join(a.out, 'node_summary.json'), 'w'), indent=2)

# ---------------------------------------------------------------------------------------------
class WindowDataset:
    """torch-style Dataset: one sample per (patient, hour). Lookback L hours, left-padded with zeros.
    Returns x [L, 2F+3] (values | mask | static) and y (label at the last hour)."""
    def __init__(self, npz_path, split=0, L=24):
        d = np.load(npz_path)
        self.X, self.M, self.S, self.y, offs = d['X'], d['M'], d['S'], d['y'], d['offs']
        pats = np.where(d['split'] == split)[0]
        self.index = np.concatenate([np.arange(offs[p], offs[p+1]) for p in pats])
        self.start = np.repeat(offs[pats], offs[pats+1] - offs[pats])   # patient start row per sample
        self.L = L
    def __len__(self): return len(self.index)
    def __getitem__(self, i):
        t, s = self.index[i], self.start[i]
        lo = max(s, t - self.L + 1)
        feat = np.concatenate([self.X[lo:t+1], self.M[lo:t+1], self.S[lo:t+1]], 1)
        x = np.zeros((self.L, feat.shape[1]), np.float32); x[-len(feat):] = feat
        return x, np.float32(self.y[t])
    def labels(self): return self.y[self.index]   # for class weights / samplers

if __name__ == '__main__':
    main()
