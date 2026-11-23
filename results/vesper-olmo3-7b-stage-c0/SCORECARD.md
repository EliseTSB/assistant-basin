# Stage C0 Scorecard (OLMo 3 7B Registered v1)

- **Stage C1 Gate (`C0.0` + `C0.1` + `C0.2` + `C0.3`):** **FAIL**
- **Receipt SHA-256 (uncompressed):** `fe7883be37ecf0b7a0a694fc1dd035215d05c27c297cc0959ef23f379258fb1c`
- **Constants SHA-256:** `3be64ef80f12d34fc59d7e561d5617dd96ecb605649d5f64e1a98bd974acbe98`
- **Design SHA-256:** `59b9da1a585dbecba26954ae41a394f9a1a385bcc831a234b859d9d5e52a78bf` (`mode=strict`)

## Gate & Prediction Verdicts

| ID | Verdict | Key Metric | Holm `p_raw` (`p_holm`) |
|---|---|---|---|
| **C0.0** | **PASS** | finite `True`, sham `0.0`, tel p95 `0.0425` (`n=13856`), cap max `0.0000`, fde p95 `0.0759` | — |
| **C0.1** | **FAIL** | `ρ_within=0.7911`, `ρ(R_mf)=0.6421` (`p=0.0018`), `Δρ=0.0732` (`q05=-0.0176`), `frac_err(R_mf/R_ridge)=0.363/0.118` | `0.000100` (`0.000100`) |
| **C0.2** | **FAIL** | pooled gap ratio `0.2837` (`q05=0.2616`), `Ret(ll)=0.8046` | `0.000015` (`0.000061`) |
| **C0.3** | **PASS** | `mean(d_t)=+0.1026` (`q05=+0.0760`) | `0.000031` (`0.000061`) |
| **C0.4** | **FAIL** | `geo_ratio=1.777x` (`16/16`), cross mean `1.235` vs F `0.299`, max `1.714` | `0.000015` (`0.000061`) |
