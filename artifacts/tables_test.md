
#### 30 s head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.3432 | 0.1053 | 0.0200 |  |
| + clock-aware vol model | 0.3320 | 0.1050 | 0.0209 | -0.0112 [-0.0119, -0.0105] |
| + no-move atom & fat-tailed law | 0.3271 | 0.1037 | 0.0099 | -0.0161 [-0.0172, -0.0150] |
| + drift (structural model) | 0.3231 | 0.1022 | 0.0076 | -0.0201 [-0.0214, -0.0189] |
| + calibration from prior | 0.3223 | 0.1020 | 0.0050 | -0.0209 [-0.0223, -0.0195] |
| direct GBM (rival) | 0.3233 | 0.1021 | 0.0066 | -0.0199 [-0.0214, -0.0185] |
| **final: blend** | 0.3221 | 0.1019 | 0.0057 | -0.0211 [-0.0226, -0.0197] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 1.4%, e∈(0, 30]: 4.1%, e∈(30, 60]: 5.9%, e∈(60, 120]: 7.0%, e∈(120, 180]: 7.1%, e∈(180, 240]: 7.0%, e∈(240, 270]: 7.2%

quantile coverage (nominal 5/10/25/50/75/90/95%): 4.6/9.2/23.5/51.2/76.5/90.8/95.3; pinball skill vs naive Gaussian 2.93%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 1.9% | 0.0555 | 0.0446 | 19.6% | 0.0009 |
| -10 bps | 5.6% | 0.1484 | 0.1228 | 17.3% | 0.0035 |
| -5 bps | 10.9% | 0.2674 | 0.2223 | 16.9% | 0.0076 |
| +5 bps | 10.7% | 0.2600 | 0.2174 | 16.4% | 0.0051 |
| +10 bps | 5.5% | 0.1448 | 0.1199 | 17.2% | 0.0022 |
| +20 bps | 1.8% | 0.0553 | 0.0439 | 20.6% | 0.0006 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 13.9% | 14.0% | 0.6525 | 0.5349 | 18.0% | 0.935 |
| ±10 bps | 6.1% | 6.2% | 0.3210 | 0.2645 | 17.6% | 0.984 |
| ±20 bps | 1.9% | 2.0% | 0.1135 | 0.0907 | 20.2% | 0.996 |

#### 60 s head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.4413 | 0.1410 | 0.0213 |  |
| + clock-aware vol model | 0.4283 | 0.1402 | 0.0211 | -0.0130 [-0.0140, -0.0120] |
| + no-move atom & fat-tailed law | 0.4262 | 0.1394 | 0.0115 | -0.0151 [-0.0162, -0.0140] |
| + drift (structural model) | 0.4235 | 0.1384 | 0.0095 | -0.0177 [-0.0190, -0.0166] |
| + calibration from prior | 0.4232 | 0.1383 | 0.0080 | -0.0180 [-0.0193, -0.0169] |
| direct GBM (rival) | 0.4250 | 0.1386 | 0.0095 | -0.0163 [-0.0174, -0.0152] |
| **final: blend** | 0.4234 | 0.1382 | 0.0088 | -0.0179 [-0.0191, -0.0168] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 0.8%, e∈(0, 30]: 2.1%, e∈(30, 60]: 3.7%, e∈(60, 120]: 4.6%, e∈(120, 180]: 4.9%, e∈(180, 240]: 5.0%

quantile coverage (nominal 5/10/25/50/75/90/95%): 4.4/8.9/23.5/50.5/76.6/91.0/95.5; pinball skill vs naive Gaussian 2.64%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 3.3% | 0.0965 | 0.0779 | 19.3% | 0.0018 |
| -10 bps | 9.7% | 0.2324 | 0.1984 | 14.6% | 0.0073 |
| -5 bps | 18.0% | 0.3756 | 0.3283 | 12.6% | 0.0132 |
| +5 bps | 17.8% | 0.3695 | 0.3238 | 12.4% | 0.0098 |
| +10 bps | 9.5% | 0.2303 | 0.1957 | 15.0% | 0.0042 |
| +20 bps | 3.2% | 0.0964 | 0.0771 | 20.0% | 0.0008 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 21.6% | 21.7% | 0.8262 | 0.7061 | 14.5% | 0.883 |
| ±10 bps | 10.3% | 10.5% | 0.4879 | 0.4116 | 15.6% | 0.955 |
| ±20 bps | 3.3% | 3.4% | 0.1955 | 0.1567 | 19.9% | 0.984 |

#### 5-minute close head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.4748 | 0.1572 | 0.0169 |  |
| + clock-aware vol model | 0.4662 | 0.1560 | 0.0142 | -0.0086 [-0.0097, -0.0076] |
| + no-move atom & fat-tailed law | 0.4634 | 0.1556 | 0.0048 | -0.0115 [-0.0128, -0.0103] |
| + drift (structural model) | 0.4621 | 0.1551 | 0.0047 | -0.0128 [-0.0141, -0.0115] |
| + calibration from prior | 0.4621 | 0.1551 | 0.0048 | -0.0128 [-0.0141, -0.0115] |
| direct GBM (rival) | 0.4643 | 0.1557 | 0.0082 | -0.0106 [-0.0119, -0.0092] |
| **final: blend** | 0.4621 | 0.1551 | 0.0064 | -0.0127 [-0.0140, -0.0113] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 0.2%, e∈(0, 30]: 0.5%, e∈(30, 60]: 1.1%, e∈(60, 120]: 1.9%, e∈(120, 180]: 2.9%, e∈(180, 240]: 4.3%, e∈(240, 270]: 6.3%, e∈(270, 299]: 11.2%

quantile coverage (nominal 5/10/25/50/75/90/95%): 4.4/8.9/23.5/50.7/76.4/90.7/95.3; pinball skill vs naive Gaussian 2.31%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 6.9% | 0.1858 | 0.1563 | 15.9% | 0.0075 |
| -10 bps | 17.5% | 0.3466 | 0.3091 | 10.8% | 0.0142 |
| -5 bps | 28.6% | 0.4584 | 0.4155 | 9.4% | 0.0150 |
| +5 bps | 28.5% | 0.4565 | 0.4147 | 9.1% | 0.0125 |
| +10 bps | 17.5% | 0.3552 | 0.3117 | 12.3% | 0.0092 |
| +20 bps | 6.7% | 0.1891 | 0.1565 | 17.2% | 0.0053 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 31.5% | 31.4% | 0.8692 | 0.7575 | 12.8% | 0.810 |
| ±10 bps | 18.3% | 18.3% | 0.6943 | 0.5995 | 13.7% | 0.875 |
| ±20 bps | 6.8% | 6.9% | 0.3738 | 0.3076 | 17.7% | 0.921 |