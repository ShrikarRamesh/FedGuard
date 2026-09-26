# Demo runbook

Two front ends show the same pipeline artefacts:

- **Streamlit app** (`app/`): driven entirely by real run outputs. It supports live Flower training.
- **Stand-alone HTML** (`app/static/fedguard_demo.html`): its *Load results* button reads `results/results.json`.

## 1. Produce the artefacts (once)

```bash
fedguard alerts  -e fedguard          # thresholds tuned on validation, alert metrics on test
fedguard explain -e fedguard          # global Integrated Gradients + attention rollout
fedguard attack  -e fedavg --seed 0   # gradient inversion, no DP vs FedGuard DP
fedguard report                       # results/summary.csv, tables, figures
fedguard export                       # results/results.json (+ results_full.json), schema-validated
```

## 2. Run the app

```bash
cd app
streamlit run streamlit_app.py            # real artefacts only
streamlit run streamlit_app.py -- --demo  # synthetic layout data, with a red "DEMO DATA" banner
```

The app works offline (fonts are bundled in `app/static/fonts`, SIL OFL). Each page names the command to run if one of its artefacts is missing. It never falls back to synthetic data silently.

## 3. Bedside patients: selection rule

`fedguard export` picks the demo patients from the **global test set** with a fixed seed (0). The rule for septic patients does not look at model performance:

1. **Three septic patients:** a seeded random draw among septic test patients with 36–96 h stays and an unambiguous onset (not positive from the first hour).
2. **One non-septic patient, "false alarm avoided":** a seeded random draw among non-septic test patients (36–96 h) for whom threshold-only alerting raises ≥ 1 alarm and the uncertainty-gated policy raises none, at the validation-tuned thresholds of the FedGuard seed-0 run. The build spec requires one such example. The selection criterion is shown in the app next to the patient.
3. **Two non-septic patients:** a seeded random draw among the remaining non-septic test patients with 36–96 h stays.

The export also records:
- **Risk curves:** the FedGuard (ε = 3, seed 0) model's MC-Dropout mean and std, T = 50.
- **Vitals:** raw clinical units, forward-filled within the stay. Before the first measurement they are null, shown as `--`.
- **Attributions:** Integrated Gradients per variable for the prediction at each hour.

Predictions are never edited.

## 4. Live federated training on one machine (4 SuperNodes)

```powershell
.\scripts\deploy\start_local.ps1            # SuperLink + 4 SuperNodes (A_MICU, A_SICU, B_MICU, B_SICU)
.\scripts\deploy\run_flower.ps1 -Rounds 40  # submits the FedGuard Flower app (FedAvg); -Fast for a 2-round check
.\scripts\deploy\stop_local.ps1
```

Open the app, go to *Train together → Live Flower deployment*, and it tails `runs/flower/<run>/events.jsonl` every 2 s.

Tested on this laptop (Windows 11, Flower 1.38.0): all 4 SuperNodes register, and every round receives 4/4 results with 0 failures.

## 5. Live demo across four laptops on a LAN

Every laptop needs the repository and `pip install -e .` in a Python 3.11 venv (see README). On the machine that holds the processed data, export one directory per hospital node:

```bash
fedguard data export-node --node A_MICU --out node_A_MICU   # copy node_A_MICU/ to the A_MICU laptop only
fedguard data export-node --node A_SICU --out node_A_SICU
fedguard data export-node --node B_MICU --out node_B_MICU
fedguard data export-node --node B_SICU --out node_B_SICU
```

Each exported directory contains **only that node's patients**. A SuperNode never sees another hospital's records.

| Machine | Command |
|---|---|
| Server laptop | `scripts\deploy\start_server.ps1` (prints its LAN IP; opens ports 9092 fleet and 9093 control) |
| Hospital laptop k | `scripts\deploy\start_client.ps1 -Server <server-ip> -Node A_MICU -DataDir C:\fedguard\node_A_MICU` |
| Any machine with the repo | `scripts\deploy\run_flower.ps1 -Connection lan -Rounds 40` after adding the connection below |

The connection goes in `$FLWR_HOME/config.toml` (default `~/.flwr/config.toml`):

```toml
[superlink.lan]
address = "<server-ip>:9093"
insecure = true
```

Allow inbound TCP 9092/9093 on the server's firewall. The demo uses `--insecure` (no TLS) on a closed LAN only. For anything else, enable TLS and SuperNode authentication (Flower docs).

Run the Streamlit app on the **server laptop**, where the ServerApp writes `runs/flower/<timestamp>_<seed>/events.jsonl`. In the Flower deployment the server holds no patient data. Its validation curve is the client-weighted mean of each hospital's own validation AUROC. The exact global metrics come afterwards from `fedguard fl eval-checkpoints runs/flower/<run>`.
