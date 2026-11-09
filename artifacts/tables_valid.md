
#### 30 s head - P(S >= P0) on valid

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.3437 | 0.1068 | 0.0153 |  |
| + clock-aware vol model | 0.3347 | 0.1064 | 0.0183 | -0.0090 [-0.0096, -0.0083] |
| + no-move atom & fat-tailed law | 0.3317 | 0.1056 | 0.0091 | -0.0120 [-0.0131, -0.0109] |
| + drift (structural model) | 0.3279 | 0.1042 | 0.0080 | -0.0158 [-0.0174, -0.0144] |
| + calibration from prior | 0.3275 | 0.1040 | 0.0061 | -0.0162 [-0.0179, -0.0147] |
| direct GBM (rival) | 0.3283 | 0.1042 | 0.0075 | -0.0154 [-0.0170, -0.0139] |
| **final: blend** | 0.3272 | 0.1039 | 0.0067 | -0.0164 [-0.0181, -0.0149] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 1.2%, e∈(0, 30]: 3.0%, e∈(30, 60]: 4.1%, e∈(60, 120]: 5.1%, e∈(120, 180]: 5.7%, e∈(180, 240]: 5.7%, e∈(240, 270]: 6.5%

quantile coverage (nominal 5/10/25/50/75/90/95%): 4.6/9.0/23.5/50.8/76.7/91.1/95.5; pinball skill vs naive Gaussian 2.59%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 3.3% | 0.0952 | 0.0757 | 20.5% | 0.0009 |
| -10 bps | 7.9% | 0.1979 | 0.1654 | 16.5% | 0.0048 |
| -5 bps | 13.4% | 0.3035 | 0.2586 | 14.8% | 0.0115 |
| +5 bps | 13.4% | 0.2993 | 0.2559 | 14.5% | 0.0087 |
| +10 bps | 7.8% | 0.1944 | 0.1636 | 15.9% | 0.0032 |
| +20 bps | 3.3% | 0.0934 | 0.0753 | 19.4% | 0.0012 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 18.2% | 18.0% | 0.7651 | 0.6293 | 17.7% | 0.906 |
| ±10 bps | 9.0% | 9.1% | 0.4418 | 0.3660 | 17.2% | 0.966 |
| ±20 bps | 3.4% | 3.5% | 0.1969 | 0.1569 | 20.3% | 0.990 |

#### 60 s head - P(S >= P0) on valid

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.4430 | 0.1428 | 0.0165 |  |
| + clock-aware vol model | 0.4322 | 0.1418 | 0.0175 | -0.0108 [-0.0118, -0.0098] |
| + no-move atom & fat-tailed law | 0.4310 | 0.1414 | 0.0084 | -0.0120 [-0.0131, -0.0110] |
| + drift (structural model) | 0.4286 | 0.1404 | 0.0079 | -0.0144 [-0.0157, -0.0131] |
| + calibration from prior | 0.4285 | 0.1404 | 0.0066 | -0.0145 [-0.0159, -0.0133] |
| direct GBM (rival) | 0.4296 | 0.1405 | 0.0079 | -0.0134 [-0.0148, -0.0122] |
| **final: blend** | 0.4284 | 0.1403 | 0.0073 | -0.0146 [-0.0160, -0.0134] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 0.6%, e∈(0, 30]: 1.6%, e∈(30, 60]: 2.3%, e∈(60, 120]: 3.7%, e∈(120, 180]: 4.1%, e∈(180, 240]: 4.6%

quantile coverage (nominal 5/10/25/50/75/90/95%): 4.6/9.0/23.4/50.3/76.7/91.2/95.5; pinball skill vs naive Gaussian 2.36%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 5.7% | 0.1589 | 0.1280 | 19.4% | 0.0025 |
| -10 bps | 13.2% | 0.2989 | 0.2577 | 13.8% | 0.0087 |
| -5 bps | 21.6% | 0.4170 | 0.3704 | 11.2% | 0.0160 |
| +5 bps | 21.6% | 0.4106 | 0.3665 | 10.7% | 0.0128 |
| +10 bps | 13.1% | 0.2912 | 0.2535 | 13.0% | 0.0061 |
| +20 bps | 5.6% | 0.1535 | 0.1260 | 17.9% | 0.0016 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 26.7% | 26.5% | 0.9055 | 0.7725 | 14.7% | 0.851 |
| ±10 bps | 14.5% | 14.6% | 0.6265 | 0.5316 | 15.1% | 0.925 |
| ±20 bps | 5.8% | 5.9% | 0.3202 | 0.2580 | 19.4% | 0.967 |

#### 5-minute close head - P(S >= P0) on valid

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.4830 | 0.1605 | 0.0163 |  |
| + clock-aware vol model | 0.4746 | 0.1589 | 0.0102 | -0.0084 [-0.0097, -0.0073] |
| + no-move atom & fat-tailed law | 0.4717 | 0.1587 | 0.0038 | -0.0113 [-0.0128, -0.0099] |
| + drift (structural model) | 0.4705 | 0.1583 | 0.0029 | -0.0125 [-0.0142, -0.0110] |
| + calibration from prior | 0.4705 | 0.1583 | 0.0029 | -0.0125 [-0.0142, -0.0110] |
| direct GBM (rival) | 0.4721 | 0.1587 | 0.0042 | -0.0110 [-0.0128, -0.0093] |
| **final: blend** | 0.4704 | 0.1582 | 0.0033 | -0.0126 [-0.0144, -0.0111] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 0.2%, e∈(0, 30]: 0.6%, e∈(30, 60]: 1.0%, e∈(60, 120]: 1.8%, e∈(120, 180]: 2.8%, e∈(180, 240]: 4.1%, e∈(240, 270]: 6.2%, e∈(270, 299]: 11.2%

quantile coverage (nominal 5/10/25/50/75/90/95%): 5.0/9.4/23.8/50.5/76.5/90.7/95.1; pinball skill vs naive Gaussian 2.40%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 10.8% | 0.2661 | 0.2234 | 16.0% | 0.0069 |
| -10 bps | 22.4% | 0.4104 | 0.3658 | 10.9% | 0.0129 |
| -5 bps | 32.8% | 0.4847 | 0.4419 | 8.8% | 0.0119 |
| +5 bps | 32.8% | 0.4820 | 0.4396 | 8.8% | 0.0102 |
| +10 bps | 22.4% | 0.4066 | 0.3631 | 10.7% | 0.0101 |
| +20 bps | 10.8% | 0.2616 | 0.2227 | 14.9% | 0.0036 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 35.7% | 35.5% | 0.8803 | 0.7625 | 13.4% | 0.791 |
| ±10 bps | 23.5% | 23.4% | 0.7925 | 0.6804 | 14.1% | 0.844 |
| ±20 bps | 11.0% | 10.9% | 0.5259 | 0.4333 | 17.6% | 0.889 |