# Analysis methods

This document summarizes the statistical specifications implemented in the analysis modules. Module-to-table and figure mappings are listed in the [README](../README.md#analyses).

## Exposure and covariates

The primary exposure is `abs(alps_L - alps_R)`. Bilateral mean ALPS is `(alps_L + alps_R) / 2`. Relative asymmetry, `2 * abs(L - R) / (L + R)`, and the absolute log-ratio, `abs(log(L / R))`, are sensitivity definitions. The prediction extension adds the raw absolute difference.

The primary reference covariates are age, sex, education, baseline MMSE, `log1p(WMH)`, brain parenchymal fraction (BPF), and bilateral mean ALPS. WMH is measured in mL and must be nonnegative. BPF is the supplied UCD tissue fraction, `(TOTAL_GRAY + TOTAL_WHITE + WMH) / TOTAL_BRAIN`. Sex is coded male = 0 and female = 1.

## Primary Cox analysis

The Cox model uses follow-up in days and separate baseline hazards for CN and MCI. The expanded model adds absolute ALPS asymmetry. Fits use robust standard errors, no additional cluster variable, and no penalization. A one-degree-of-freedom likelihood-ratio test compares the models on the same participants. Proportional hazards are assessed using a rank-transformed Schoenfeld residual test.

## Standardization

| Analysis | Scaling |
|---|---|
| Primary Cox, directional/component, outlier, and influence analyses | Full-cohort mean and population SD (`ddof=0`) |
| Component competition | Full-cohort mean and population SD (`ddof=0`), fixed during bootstrap resampling |
| Structural sensitivity | Core predictors use full-cohort scaling (`ddof=0`); structural additions use the common analysis sample (`ddof=0`) |
| Technical adjustment and alternative asymmetry definitions | Available full-cohort values, sample SD (`ddof=1`), fixed after complete-case selection |
| Paired reorientation comparison | Method-specific values standardized within the paired sample using sample SD (`ddof=1`) |
| Spline | Analysis-sample mean and sample SD (`ddof=1`) |
| Adjusted absolute risks | Primary full-cohort scaling (`ddof=0`) |
| Neural prediction | StandardScaler fitted on inner-training participants, then applied to validation and test participants |

## Directional and structural analyses

Left, right, and mean ALPS are evaluated separately with clinical and structural adjustment. Signed and absolute differences additionally adjust for mean ALPS. Tensor-component asymmetries are raw absolute bilateral differences for xproj, xassoc, yproj, and zassoc. The zassoc competition analysis fits the reference model, each exposure separately, and both exposures jointly on a common sample. Observed-model Wald inference uses `robust=True`; bootstrap coefficient fits use `robust=False`.

Tail analyses exclude values above the full-cohort 99th, 97.5th, and 95th percentiles. Influence analyses remove the 1, 5, or 10 observations with the largest absolute delta-beta for the asymmetry coefficient. Full-cohort scaling remains fixed after exclusions.

Structural asymmetry is `2 * abs(L - R) / (L + R)`. Ventricular burden is normalized by intracranial volume. One FreeSurfer record is selected per participant and date using processing status, QC rank, and measurement completeness. Optional ST30SV/ST89SV fields contribute to completeness ranking when present. Same-day matching is primary; sensitivity matching uses the nearest examination within 365 days, preferring the earlier date when equally distant. The same-day common sample has 1,097 participants and 110 events; the 365-day sample has 1,101 participants and 111 events.

## Technical and processing analyses

Technical blocks comprise scanner/protocol, within-scan motion, head orientation, and acquisition-plane orientation. Each expanded model and its reference use the same complete-case sample. Comprehensive technical adjustment uses 936 participants and 105 events; nonvarying manufacturer indicators are omitted from fitting.

The paired processing comparison includes 1,130 participants and 115 events. Each method supplies its own bilateral mean and absolute ALPS asymmetry. Agreement is summarized using Pearson and Spearman correlations, ICC(A,1), and mean method difference.

FA and MD analyses use composite and ROI-specific absolute bilateral differences from the VECREG measurements, with VECREG-derived mean ALPS as a covariate. The primary-method ALPS comparison in Figure 4B uses primary-method mean ALPS; the two mean-ALPS specifications are distinct. Benjamini–Hochberg correction covers the six scalar-asymmetry Wald tests.

## Spline and absolute risk

The restricted cubic spline has knots at the 10th, 50th, and 90th exposure percentiles, with one linear and one nonlinear term. Relative hazards are referenced to median asymmetry. The calculation grid contains 240 points spanning the 1st–99th percentiles; Figure 3A displays the 5th–95th percentile range. Pointwise confidence intervals use the spline-basis contrast and the fitted `variance_matrix_` covariance matrix. Likelihood-ratio tests assess overall contribution and nonlinearity.

Absolute conversion risks are estimated by marginal standardization at mean asymmetry and 1 SD above the mean. Other observed covariates and diagnosis strata remain fixed. Horizons use `months * 365.25 / 12` days. Each bootstrap model predicts both exposure settings for the original target cohort, and risks are averaged over that cohort.

| Bootstrap analysis | Participant resamples | Seed |
|---|---:|---:|
| Absolute risks | 500 | 20260902 |
| Paired reorientation coefficients | 1,000 | 20260902 |
| Joint ALPS/zassoc coefficients | 300 | 20260830 |
| Paired prediction differences | 300 | 20260828 + 717 |

## Neural survival prediction

Reference predictors comprise age, sex, education, baseline diagnosis, MMSE, BPF, `log1p(WMH)`, and mean ALPS. Extended models add raw absolute ALPS asymmetry. Five repetitions of five-fold cross-validation use joint diagnosis/event stratification. Within each outer-training set, one of five stratified inner folds is used for early stopping. Standardization and survival-time transformations are fitted on inner-training data.

| Setting | Value |
|---|---|
| Models | DeepSurv, CoxTime, DeepHitSingle |
| Hidden layers | 16 and 8 units |
| Dropout / batch normalization | 0.15 / disabled |
| Optimizer / learning rate | Adam / 0.001 |
| Maximum epochs / patience | 256 / 25 |
| DeepSurv batch | Full inner-training set |
| CoxTime and DeepHitSingle batch | 256 |
| DeepHitSingle duration bins / alpha / sigma | 24 / 0.20 / 0.10 |
| DeepHitSingle survival interpolation | Factor 10 |
| Base seed | 20260828 |
| Training seed offsets | 111, 222, 333 |

Outer splits use `20260828 + 1009 * repeat`; inner splits use `20260828 + 10000 * repeat + fold`. Training adds the corresponding seed offset to the inner-split seed. Repeat and fold indices start at zero. Three seed-specific survival predictions are averaged within each test fold. Performance is calculated separately within each repetition and then averaged across repetitions.

## Prediction evaluation

Prediction follow-up months are derived from days divided by `365.25 / 12`. Cumulative/dynamic AUC and IPCW Brier scores are evaluated at 12, 36, and 60 months. The censoring Kaplan–Meier distribution is estimated from each outer-training set. Case weights use `1 / G(T-)`; controls use `1 / G(h)`. The minimum supported censoring survival is 0.05. AUC ties receive half credit. Brier loss is divided by all evaluated participants, including early-censored participants with zero weight.

IBS integrates Brier scores on a 17-point grid, every 3 months from 12 to 60 months, and divides by the 48-month interval. Prediction is limited to model and censoring support. Horizons with fewer than 20 cases or controls are flagged as low information; the 12-month horizon is an information check.

The paired bootstrap resamples participants while keeping their predictions and censoring weights fixed across repetitions and model pairs. Percentile intervals therefore describe uncertainty conditional on those fitted predictions and weights. Approximate two-sided P values use twice the smaller bootstrap tail fraction with a plus-one correction.

Holm correction covers the three 60-month AUC contrasts. Benjamini–Hochberg correction covers 12 secondary contrasts: three 36-month AUC, six 36/60-month Brier, and three IBS differences. Positive AUC differences and negative Brier/IBS differences favor the extended model.

Calibration is descriptive and repetition-specific. IPCW logistic models jointly estimate the intercept and slope; calibration-in-the-large is fitted separately with slope fixed at one. Estimates are averaged across repetitions. The performance calculations use the unrecalibrated predictions.
