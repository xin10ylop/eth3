
#### 10 s head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.2162 | 0.0630 | 0.0228 |  |
| + clock-aware vol model | 0.2142 | 0.0621 | 0.0194 | -0.0020 [-0.0027, -0.0012] |
| + no-move atom & fat-tailed law | 0.1998 | 0.0601 | 0.0044 | -0.0165 [-0.0178, -0.0152] |
| + drift (structural model) | 0.1945 | 0.0585 | 0.0033 | -0.0218 [-0.0232, -0.0204] |
| + calibration from prior | 0.1925 | 0.0579 | 0.0026 | -0.0237 [-0.0256, -0.0221] |
| direct GBM (rival) | 0.1921 | 0.0578 | 0.0010 | -0.0241 [-0.0262, -0.0223] |
| **final: blend** | 0.1918 | 0.0577 | 0.0013 | -0.0245 [-0.0265, -0.0227] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 3.8%, e∈(0, 30]: 10.0%, e∈(30, 60]: 12.0%, e∈(60, 120]: 12.1%, e∈(120, 180]: 12.4%, e∈(180, 240]: 11.9%, e∈(240, 270]: 12.2%, e∈(270, 299]: 13.9%

quantile coverage (nominal 5/10/25/50/75/90/95%): 5.2/9.7/25.7/54.6/76.5/90.3/94.8; pinball skill vs naive Gaussian 4.57%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 0.8% | 0.0257 | 0.0196 | 23.6% | 0.0005 |
| -10 bps | 2.3% | 0.0745 | 0.0560 | 24.8% | 0.0011 |
| -5 bps | 4.5% | 0.1508 | 0.1099 | 27.1% | 0.0011 |
| +5 bps | 4.4% | 0.1441 | 0.1061 | 26.4% | 0.0009 |
| +10 bps | 2.3% | 0.0717 | 0.0544 | 24.2% | 0.0012 |
| +20 bps | 0.8% | 0.0251 | 0.0189 | 24.6% | 0.0004 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 6.0% | 6.1% | 0.3941 | 0.2857 | 27.5% | 0.981 |
| ±10 bps | 2.5% | 2.6% | 0.1637 | 0.1232 | 24.7% | 0.997 |
| ±20 bps | 0.8% | 0.8% | 0.0523 | 0.0397 | 24.1% | 0.999 |

#### 15 s head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.2581 | 0.0766 | 0.0217 |  |
| + clock-aware vol model | 0.2522 | 0.0759 | 0.0196 | -0.0060 [-0.0066, -0.0053] |
| + no-move atom & fat-tailed law | 0.2417 | 0.0742 | 0.0045 | -0.0165 [-0.0178, -0.0152] |
| + drift (structural model) | 0.2364 | 0.0725 | 0.0031 | -0.0217 [-0.0232, -0.0203] |
| + calibration from prior | 0.2351 | 0.0721 | 0.0026 | -0.0230 [-0.0249, -0.0214] |
| direct GBM (rival) | 0.2350 | 0.0720 | 0.0013 | -0.0232 [-0.0252, -0.0213] |
| **final: blend** | 0.2345 | 0.0719 | 0.0013 | -0.0237 [-0.0256, -0.0219] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 2.7%, e∈(0, 30]: 7.5%, e∈(30, 60]: 9.4%, e∈(60, 120]: 10.0%, e∈(120, 180]: 10.2%, e∈(180, 240]: 9.9%, e∈(240, 270]: 10.1%, e∈(270, 299]: 10.9%

quantile coverage (nominal 5/10/25/50/75/90/95%): 5.1/9.8/24.9/53.0/76.2/90.2/94.9; pinball skill vs naive Gaussian 3.77%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 1.1% | 0.0337 | 0.0264 | 21.6% | 0.0008 |
| -10 bps | 3.2% | 0.0953 | 0.0748 | 21.5% | 0.0014 |
| -5 bps | 6.3% | 0.1867 | 0.1439 | 22.9% | 0.0017 |
| +5 bps | 6.2% | 0.1791 | 0.1394 | 22.2% | 0.0010 |
| +10 bps | 3.1% | 0.0923 | 0.0729 | 21.0% | 0.0015 |
| +20 bps | 1.1% | 0.0331 | 0.0256 | 22.7% | 0.0005 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 8.3% | 8.5% | 0.4800 | 0.3673 | 23.5% | 0.969 |
| ±10 bps | 3.5% | 3.6% | 0.2088 | 0.1640 | 21.5% | 0.995 |
| ±20 bps | 1.1% | 1.1% | 0.0687 | 0.0535 | 22.2% | 0.999 |

#### 30 s head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.3432 | 0.1053 | 0.0200 |  |
| + clock-aware vol model | 0.3315 | 0.1047 | 0.0190 | -0.0117 [-0.0124, -0.0111] |
| + no-move atom & fat-tailed law | 0.3266 | 0.1036 | 0.0058 | -0.0166 [-0.0178, -0.0155] |
| + drift (structural model) | 0.3223 | 0.1020 | 0.0039 | -0.0209 [-0.0223, -0.0196] |
| + calibration from prior | 0.3216 | 0.1017 | 0.0022 | -0.0216 [-0.0231, -0.0201] |
| direct GBM (rival) | 0.3220 | 0.1018 | 0.0022 | -0.0212 [-0.0229, -0.0197] |
| **final: blend** | 0.3211 | 0.1016 | 0.0020 | -0.0221 [-0.0237, -0.0205] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 1.4%, e∈(0, 30]: 4.4%, e∈(30, 60]: 6.2%, e∈(60, 120]: 7.3%, e∈(120, 180]: 7.4%, e∈(180, 240]: 7.3%, e∈(240, 270]: 7.3%

quantile coverage (nominal 5/10/25/50/75/90/95%): 4.9/9.7/24.3/51.3/75.9/90.3/95.1; pinball skill vs naive Gaussian 3.09%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 1.9% | 0.0555 | 0.0446 | 19.6% | 0.0013 |
| -10 bps | 5.6% | 0.1484 | 0.1224 | 17.5% | 0.0024 |
| -5 bps | 10.9% | 0.2674 | 0.2215 | 17.2% | 0.0014 |
| +5 bps | 10.7% | 0.2600 | 0.2169 | 16.6% | 0.0017 |
| +10 bps | 5.5% | 0.1448 | 0.1198 | 17.3% | 0.0025 |
| +20 bps | 1.8% | 0.0553 | 0.0439 | 20.7% | 0.0008 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 13.9% | 14.0% | 0.6525 | 0.5330 | 18.3% | 0.935 |
| ±10 bps | 6.1% | 6.2% | 0.3210 | 0.2638 | 17.8% | 0.984 |
| ±20 bps | 1.9% | 2.0% | 0.1135 | 0.0905 | 20.3% | 0.996 |

#### 60 s head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.4413 | 0.1410 | 0.0213 |  |
| + clock-aware vol model | 0.4278 | 0.1400 | 0.0186 | -0.0135 [-0.0144, -0.0125] |
| + no-move atom & fat-tailed law | 0.4258 | 0.1393 | 0.0071 | -0.0155 [-0.0165, -0.0144] |
| + drift (structural model) | 0.4231 | 0.1382 | 0.0057 | -0.0182 [-0.0195, -0.0170] |
| + calibration from prior | 0.4226 | 0.1380 | 0.0034 | -0.0187 [-0.0200, -0.0175] |
| direct GBM (rival) | 0.4236 | 0.1383 | 0.0044 | -0.0177 [-0.0190, -0.0165] |
| **final: blend** | 0.4223 | 0.1379 | 0.0041 | -0.0190 [-0.0203, -0.0178] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 0.8%, e∈(0, 30]: 2.3%, e∈(30, 60]: 3.9%, e∈(60, 120]: 4.9%, e∈(120, 180]: 5.2%, e∈(180, 240]: 5.3%

quantile coverage (nominal 5/10/25/50/75/90/95%): 4.6/9.4/24.3/50.6/75.9/90.5/95.3; pinball skill vs naive Gaussian 2.79%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 3.3% | 0.0965 | 0.0779 | 19.3% | 0.0018 |
| -10 bps | 9.7% | 0.2324 | 0.1978 | 14.9% | 0.0037 |
| -5 bps | 18.0% | 0.3756 | 0.3269 | 13.0% | 0.0024 |
| +5 bps | 17.8% | 0.3695 | 0.3230 | 12.6% | 0.0020 |
| +10 bps | 9.5% | 0.2303 | 0.1954 | 15.1% | 0.0034 |
| +20 bps | 3.2% | 0.0964 | 0.0771 | 20.0% | 0.0009 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 21.6% | 21.7% | 0.8262 | 0.7026 | 15.0% | 0.883 |
| ±10 bps | 10.3% | 10.5% | 0.4879 | 0.4104 | 15.9% | 0.955 |
| ±20 bps | 3.3% | 3.4% | 0.1955 | 0.1564 | 20.0% | 0.984 |

#### 5-minute close head - P(S >= P0) on test

| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |
|---|---|---|---|---|
| naive Gaussian random walk (trailing vol) | 0.4748 | 0.1572 | 0.0169 |  |
| + clock-aware vol model | 0.4655 | 0.1559 | 0.0127 | -0.0094 [-0.0105, -0.0083] |
| + no-move atom & fat-tailed law | 0.4631 | 0.1556 | 0.0022 | -0.0117 [-0.0131, -0.0105] |
| + drift (structural model) | 0.4618 | 0.1551 | 0.0033 | -0.0130 [-0.0145, -0.0116] |
| + calibration from prior | 0.4619 | 0.1552 | 0.0053 | -0.0129 [-0.0144, -0.0115] |
| direct GBM (rival) | 0.4632 | 0.1554 | 0.0062 | -0.0116 [-0.0131, -0.0101] |
| **final: blend** | 0.4617 | 0.1550 | 0.0060 | -0.0132 [-0.0146, -0.0117] |

log-loss skill vs naive by elapsed time: e∈(-1, 0]: 0.2%, e∈(0, 30]: 0.5%, e∈(30, 60]: 1.1%, e∈(60, 120]: 1.9%, e∈(120, 180]: 3.0%, e∈(180, 240]: 4.5%, e∈(240, 270]: 6.5%, e∈(270, 299]: 11.6%

quantile coverage (nominal 5/10/25/50/75/90/95%): 4.4/9.1/24.1/50.7/76.0/90.5/95.3; pinball skill vs naive Gaussian 2.36%

| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |
|---|---|---|---|---|---|
| -20 bps | 6.9% | 0.1858 | 0.1562 | 16.0% | 0.0051 |
| -10 bps | 17.5% | 0.3466 | 0.3083 | 11.1% | 0.0095 |
| -5 bps | 28.6% | 0.4584 | 0.4146 | 9.5% | 0.0105 |
| +5 bps | 28.5% | 0.4565 | 0.4149 | 9.1% | 0.0082 |
| +10 bps | 17.5% | 0.3552 | 0.3116 | 12.3% | 0.0043 |
| +20 bps | 6.7% | 0.1891 | 0.1564 | 17.3% | 0.0036 |

| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |
|---|---|---|---|---|---|---|
| ±5 bps | 31.5% | 31.4% | 0.8692 | 0.7557 | 13.1% | 0.810 |
| ±10 bps | 18.3% | 18.3% | 0.6943 | 0.5978 | 13.9% | 0.875 |
| ±20 bps | 6.8% | 6.9% | 0.3738 | 0.3072 | 17.8% | 0.920 |