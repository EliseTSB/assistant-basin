# Part 122 score

## Gates

| Gate | Result | Observed | Threshold |
| --- | --- | ---: | ---: |
| 1 token match | MISS | 743/936 within 2 tokens; 936/936 complete | 90% |
| 2 Part 95 positive control | PASS | 11/16 max wins; 16/16 complete | 9/16 |
| 3 random control T | PASS | 12/117 read; 117/117 complete | 15% |
| 3 random control F | PASS | 11/117 read; 117/117 complete | 15% |
| 3 random control C | PASS | 13/117 read; 117/117 complete | 15% |

## Activation tables: nothing is read from them because gate 1 failed

### Unit and frame results

| Unit | Frame | Haiku | DS | Max wins | Losses | Ties | Read (7/8) | Max persona | Max ordinary | Mean wins | Mean persona | Mean ordinary | Mean delta |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p92-00 | T | concept | unclassified | 8/8 | 0 | 0 | True | 0.305 | 0.058 | 6 | 0.025 | 0.003 | 0.022 |
| p92-01 | T | concept | unclassified | 8/8 | 0 | 0 | True | 0.871 | 0.054 | 8 | 0.103 | 0.001 | 0.102 |
| p92-02 | T | slot | slot | 3/8 | 5 | 0 | False | 0.037 | 0.042 | 3 | -0.000 | 0.001 | -0.002 |
| p92-03 | T | slot | slot | 4/8 | 4 | 0 | False | 0.123 | 0.125 | 5 | 0.006 | 0.006 | 0.000 |
| p92-04 | T | slot | slot | 3/8 | 5 | 0 | False | 0.050 | 0.059 | 2 | 0.003 | 0.004 | -0.001 |
| p92-05 | T | slot | slot | 2/8 | 6 | 0 | False | 0.032 | 0.061 | 5 | -0.005 | -0.011 | 0.006 |
| p92-07 | T | slot | slot | 4/8 | 4 | 0 | False | 0.030 | 0.029 | 5 | 0.001 | 0.001 | 0.001 |
| p92-08 | T | concept | concept | 7/8 | 1 | 0 | True | 0.065 | 0.042 | 3 | 0.002 | 0.003 | -0.001 |
| p92-09 | T | slot | slot | 6/8 | 2 | 0 | False | 0.244 | 0.082 | 5 | 0.032 | 0.005 | 0.027 |
| p92-10 | T | slot | slot | 3/8 | 5 | 0 | False | 0.037 | 0.041 | 3 | 0.003 | 0.003 | 0.001 |
| p92-11 | T | concept | slot | 8/8 | 0 | 0 | True | 0.645 | 0.072 | 7 | 0.050 | -0.001 | 0.051 |
| p92-13 | T | slot | slot | 8/8 | 0 | 0 | True | 0.134 | 0.056 | 5 | 0.012 | 0.002 | 0.010 |
| p92-15 | T | concept | concept | 8/8 | 0 | 0 | True | 3.121 | 0.063 | 8 | 0.500 | -0.004 | 0.504 |
| p92-18 | T | slot | slot | 8/8 | 0 | 0 | True | 0.117 | 0.055 | 1 | -0.010 | -0.004 | -0.006 |
| p92-20 | T | slot | slot | 3/8 | 5 | 0 | False | 0.029 | 0.026 | 5 | -0.003 | -0.003 | 0.000 |
| p92-24 | T | slot | slot | 1/8 | 7 | 0 | False | 0.024 | 0.036 | 0 | -0.020 | -0.005 | -0.015 |
| p92-25 | T | slot | slot | 8/8 | 0 | 0 | True | 0.382 | 0.197 | 8 | 0.050 | 0.013 | 0.037 |
| p92-28 | T | concept | slot | 8/8 | 0 | 0 | True | 0.265 | 0.149 | 8 | 0.106 | 0.021 | 0.085 |
| p92-29 | T | concept | concept | 8/8 | 0 | 0 | True | 0.089 | 0.033 | 7 | 0.004 | -0.000 | 0.004 |
| p92-30 | T | concept | slot | 7/8 | 1 | 0 | True | 0.130 | 0.072 | 7 | 0.010 | 0.005 | 0.005 |
| p92-32 | T | slot | slot | 5/8 | 3 | 0 | False | 0.268 | 0.187 | 5 | 0.010 | 0.011 | -0.002 |
| p92-33 | T | slot | slot | 1/8 | 6 | 1 | False | 0.061 | 0.108 | 2 | -0.009 | -0.002 | -0.007 |
| p92-35 | T | slot | slot | 4/8 | 4 | 0 | False | 0.044 | 0.042 | 7 | -0.000 | -0.002 | 0.002 |
| p92-36 | T | slot | slot | 4/8 | 4 | 0 | False | 0.045 | 0.039 | 6 | 0.001 | -0.004 | 0.006 |
| p92-37 | T | slot | slot | 0/8 | 2 | 6 | False | 0.187 | 0.200 | 0 | -0.002 | 0.011 | -0.012 |
| p92-38 | T | concept | slot | 8/8 | 0 | 0 | True | 1.268 | 0.094 | 8 | 0.173 | 0.021 | 0.152 |
| p92-39 | T | concept | concept | 8/8 | 0 | 0 | True | 1.797 | 0.057 | 8 | 0.239 | 0.005 | 0.234 |
| p92-40 | T | concept | slot | 8/8 | 0 | 0 | True | 0.482 | 0.067 | 1 | -0.030 | -0.000 | -0.029 |
| p92-41 | T | concept | unclassified | 8/8 | 0 | 0 | True | 0.451 | 0.135 | 8 | 0.061 | 0.014 | 0.047 |
| p92-42 | T | slot | slot | 8/8 | 0 | 0 | True | 0.247 | 0.028 | 8 | 0.025 | -0.001 | 0.026 |
| p92-43 | T | concept | concept | 5/8 | 3 | 0 | False | 0.123 | 0.105 | 4 | 0.003 | 0.007 | -0.004 |
| p92-45 | T | concept | concept | 8/8 | 0 | 0 | True | 0.977 | 0.035 | 8 | 0.089 | -0.000 | 0.090 |
| p92-46 | T | slot | slot | 3/8 | 5 | 0 | False | 0.038 | 0.042 | 6 | -0.004 | -0.006 | 0.002 |
| p92-47 | T | slot | slot | 2/8 | 6 | 0 | False | 0.021 | 0.025 | 4 | -0.001 | -0.000 | -0.000 |
| p92-48 | T | concept | concept | 5/8 | 3 | 0 | False | 0.047 | 0.041 | 5 | -0.004 | -0.005 | 0.001 |
| p92-49 | T | concept | slot | 0/8 | 8 | 0 | False | 0.162 | 0.271 | 2 | 0.003 | 0.019 | -0.016 |
| p92-50 | T | slot | slot | 4/8 | 4 | 0 | False | 0.043 | 0.030 | 0 | -0.006 | -0.002 | -0.004 |
| p92-51 | T | concept | concept | 8/8 | 0 | 0 | True | 5.328 | 0.247 | 8 | 0.736 | 0.008 | 0.727 |
| p92-52 | T | concept | slot | 1/8 | 7 | 0 | False | 0.065 | 0.092 | 0 | -0.003 | 0.007 | -0.010 |
| p92-00 | F | concept | unclassified | 3/8 | 5 | 0 | False | 0.050 | 0.047 | 2 | -0.003 | -0.001 | -0.002 |
| p92-01 | F | concept | unclassified | 8/8 | 0 | 0 | True | 0.450 | 0.083 | 8 | 0.039 | 0.002 | 0.038 |
| p92-02 | F | slot | slot | 3/8 | 5 | 0 | False | 0.041 | 0.048 | 3 | 0.001 | 0.002 | -0.001 |
| p92-03 | F | slot | slot | 6/8 | 2 | 0 | False | 0.127 | 0.100 | 5 | 0.004 | 0.005 | -0.001 |
| p92-04 | F | slot | slot | 3/8 | 5 | 0 | False | 0.053 | 0.056 | 2 | 0.002 | 0.002 | -0.000 |
| p92-05 | F | slot | slot | 2/8 | 5 | 1 | False | 0.040 | 0.054 | 6 | -0.006 | -0.014 | 0.007 |
| p92-07 | F | slot | slot | 1/8 | 7 | 0 | False | 0.015 | 0.022 | 3 | -0.000 | 0.000 | -0.000 |
| p92-08 | F | concept | concept | 6/8 | 1 | 1 | False | 0.073 | 0.043 | 1 | -0.004 | 0.000 | -0.004 |
| p92-09 | F | slot | slot | 8/8 | 0 | 0 | True | 0.238 | 0.081 | 7 | 0.024 | 0.001 | 0.023 |
| p92-10 | F | slot | slot | 3/8 | 5 | 0 | False | 0.036 | 0.038 | 5 | 0.000 | 0.000 | 0.000 |
| p92-11 | F | concept | slot | 8/8 | 0 | 0 | True | 0.387 | 0.071 | 5 | 0.012 | 0.000 | 0.012 |
| p92-13 | F | slot | slot | 6/8 | 2 | 0 | False | 0.172 | 0.071 | 6 | 0.020 | 0.007 | 0.013 |
| p92-15 | F | concept | concept | 8/8 | 0 | 0 | True | 2.439 | 0.054 | 8 | 0.368 | -0.008 | 0.376 |
| p92-18 | F | slot | slot | 5/8 | 3 | 0 | False | 0.078 | 0.057 | 0 | -0.013 | -0.000 | -0.012 |
| p92-20 | F | slot | slot | 4/8 | 4 | 0 | False | 0.024 | 0.025 | 5 | -0.002 | -0.002 | 0.000 |
| p92-24 | F | slot | slot | 4/8 | 4 | 0 | False | 0.026 | 0.032 | 0 | -0.013 | -0.005 | -0.008 |
| p92-25 | F | slot | slot | 8/8 | 0 | 0 | True | 0.425 | 0.074 | 8 | 0.018 | -0.003 | 0.021 |
| p92-28 | F | concept | slot | 8/8 | 0 | 0 | True | 0.281 | 0.137 | 8 | 0.103 | 0.016 | 0.086 |
| p92-29 | F | concept | concept | 8/8 | 0 | 0 | True | 0.098 | 0.029 | 8 | 0.005 | -0.001 | 0.005 |
| p92-30 | F | concept | slot | 4/8 | 4 | 0 | False | 0.101 | 0.062 | 1 | -0.001 | 0.004 | -0.004 |
| p92-32 | F | slot | slot | 6/8 | 2 | 0 | False | 0.230 | 0.202 | 3 | 0.005 | 0.022 | -0.017 |
| p92-33 | F | slot | slot | 1/8 | 5 | 2 | False | 0.066 | 0.096 | 3 | -0.008 | -0.006 | -0.002 |
| p92-35 | F | slot | slot | 4/8 | 4 | 0 | False | 0.046 | 0.055 | 6 | 0.002 | -0.001 | 0.002 |
| p92-36 | F | slot | slot | 3/8 | 5 | 0 | False | 0.039 | 0.040 | 3 | 0.002 | -0.003 | 0.005 |
| p92-37 | F | slot | slot | 0/8 | 0 | 8 | False | 0.184 | 0.184 | 6 | 0.001 | 0.000 | 0.001 |
| p92-38 | F | concept | slot | 8/8 | 0 | 0 | True | 1.471 | 0.091 | 8 | 0.118 | 0.018 | 0.100 |
| p92-39 | F | concept | concept | 8/8 | 0 | 0 | True | 1.649 | 0.056 | 8 | 0.088 | 0.003 | 0.085 |
| p92-40 | F | concept | slot | 8/8 | 0 | 0 | True | 0.391 | 0.074 | 1 | -0.026 | 0.002 | -0.027 |
| p92-41 | F | concept | unclassified | 8/8 | 0 | 0 | True | 0.291 | 0.138 | 8 | 0.036 | 0.016 | 0.020 |
| p92-42 | F | slot | slot | 8/8 | 0 | 0 | True | 0.149 | 0.031 | 8 | 0.011 | -0.001 | 0.012 |
| p92-43 | F | concept | concept | 5/8 | 3 | 0 | False | 0.465 | 0.443 | 1 | -0.015 | 0.010 | -0.026 |
| p92-45 | F | concept | concept | 8/8 | 0 | 0 | True | 0.963 | 0.030 | 8 | 0.087 | -0.001 | 0.088 |
| p92-46 | F | slot | slot | 6/8 | 1 | 1 | False | 0.053 | 0.047 | 5 | -0.001 | -0.003 | 0.002 |
| p92-47 | F | slot | slot | 6/8 | 2 | 0 | False | 0.029 | 0.028 | 3 | -0.001 | -0.001 | 0.000 |
| p92-48 | F | concept | concept | 2/8 | 2 | 4 | False | 0.040 | 0.040 | 1 | -0.005 | -0.003 | -0.002 |
| p92-49 | F | concept | slot | 2/8 | 6 | 0 | False | 0.209 | 0.290 | 3 | 0.017 | 0.011 | 0.006 |
| p92-50 | F | slot | slot | 5/8 | 3 | 0 | False | 0.045 | 0.034 | 1 | -0.005 | -0.001 | -0.004 |
| p92-51 | F | concept | concept | 8/8 | 0 | 0 | True | 4.369 | 0.153 | 8 | 0.601 | -0.002 | 0.603 |
| p92-52 | F | concept | slot | 3/8 | 5 | 0 | False | 0.067 | 0.077 | 4 | 0.005 | 0.004 | 0.000 |
| p92-00 | C | concept | unclassified | 4/8 | 4 | 0 | False | 0.050 | 0.047 | 2 | -0.001 | 0.001 | -0.002 |
| p92-01 | C | concept | unclassified | 8/8 | 0 | 0 | True | 0.409 | 0.077 | 8 | 0.041 | 0.000 | 0.041 |
| p92-02 | C | slot | slot | 4/8 | 4 | 0 | False | 0.040 | 0.044 | 3 | -0.002 | -0.000 | -0.001 |
| p92-03 | C | slot | slot | 6/8 | 2 | 0 | False | 0.107 | 0.094 | 0 | -0.008 | 0.003 | -0.011 |
| p92-04 | C | slot | slot | 5/8 | 3 | 0 | False | 0.060 | 0.054 | 3 | 0.002 | 0.003 | -0.000 |
| p92-05 | C | slot | slot | 1/8 | 6 | 1 | False | 0.038 | 0.058 | 4 | -0.007 | -0.011 | 0.004 |
| p92-07 | C | slot | slot | 1/8 | 7 | 0 | False | 0.012 | 0.020 | 1 | -0.001 | 0.000 | -0.001 |
| p92-08 | C | concept | concept | 6/8 | 1 | 1 | False | 0.064 | 0.037 | 1 | -0.006 | -0.001 | -0.004 |
| p92-09 | C | slot | slot | 8/8 | 0 | 0 | True | 0.214 | 0.071 | 7 | 0.015 | 0.004 | 0.011 |
| p92-10 | C | slot | slot | 2/8 | 6 | 0 | False | 0.034 | 0.035 | 5 | 0.001 | -0.000 | 0.001 |
| p92-11 | C | concept | slot | 8/8 | 0 | 0 | True | 0.393 | 0.073 | 5 | 0.008 | -0.001 | 0.009 |
| p92-13 | C | slot | slot | 7/8 | 1 | 0 | True | 0.184 | 0.067 | 6 | 0.018 | 0.006 | 0.012 |
| p92-15 | C | concept | concept | 8/8 | 0 | 0 | True | 1.936 | 0.064 | 8 | 0.257 | -0.024 | 0.281 |
| p92-18 | C | slot | slot | 5/8 | 3 | 0 | False | 0.084 | 0.065 | 0 | -0.013 | 0.001 | -0.014 |
| p92-20 | C | slot | slot | 4/8 | 4 | 0 | False | 0.024 | 0.023 | 4 | -0.002 | -0.002 | 0.000 |
| p92-24 | C | slot | slot | 5/8 | 3 | 0 | False | 0.027 | 0.030 | 0 | -0.011 | -0.005 | -0.006 |
| p92-25 | C | slot | slot | 8/8 | 0 | 0 | True | 0.497 | 0.074 | 8 | 0.022 | -0.002 | 0.024 |
| p92-28 | C | concept | slot | 1/8 | 0 | 7 | False | 0.517 | 0.500 | 8 | 0.125 | 0.042 | 0.083 |
| p92-29 | C | concept | concept | 8/8 | 0 | 0 | True | 0.126 | 0.028 | 8 | 0.011 | 0.001 | 0.010 |
| p92-30 | C | concept | slot | 6/8 | 2 | 0 | False | 0.100 | 0.060 | 1 | -0.004 | 0.001 | -0.005 |
| p92-32 | C | slot | slot | 5/8 | 3 | 0 | False | 0.294 | 0.234 | 5 | 0.042 | 0.029 | 0.012 |
| p92-33 | C | slot | slot | 3/8 | 5 | 0 | False | 0.053 | 0.087 | 4 | -0.013 | -0.012 | -0.001 |
| p92-35 | C | slot | slot | 4/8 | 4 | 0 | False | 0.052 | 0.051 | 5 | 0.002 | -0.000 | 0.003 |
| p92-36 | C | slot | slot | 5/8 | 3 | 0 | False | 0.044 | 0.042 | 3 | 0.003 | -0.002 | 0.004 |
| p92-37 | C | slot | slot | 4/8 | 4 | 0 | False | 0.052 | 0.044 | 8 | -0.000 | -0.006 | 0.006 |
| p92-38 | C | concept | slot | 8/8 | 0 | 0 | True | 1.348 | 0.084 | 8 | 0.114 | 0.013 | 0.101 |
| p92-39 | C | concept | concept | 8/8 | 0 | 0 | True | 1.510 | 0.058 | 8 | 0.090 | 0.006 | 0.084 |
| p92-40 | C | concept | slot | 8/8 | 0 | 0 | True | 0.378 | 0.063 | 5 | 0.000 | -0.002 | 0.002 |
| p92-41 | C | concept | unclassified | 8/8 | 0 | 0 | True | 0.322 | 0.136 | 7 | 0.038 | 0.016 | 0.022 |
| p92-42 | C | slot | slot | 8/8 | 0 | 0 | True | 0.133 | 0.034 | 8 | 0.008 | -0.000 | 0.008 |
| p92-43 | C | concept | concept | 3/8 | 5 | 0 | False | 0.592 | 0.531 | 1 | -0.002 | 0.022 | -0.024 |
| p92-45 | C | concept | concept | 8/8 | 0 | 0 | True | 0.875 | 0.031 | 8 | 0.076 | -0.000 | 0.077 |
| p92-46 | C | slot | slot | 3/8 | 4 | 1 | False | 0.043 | 0.040 | 4 | -0.008 | -0.010 | 0.001 |
| p92-47 | C | slot | slot | 5/8 | 3 | 0 | False | 0.040 | 0.036 | 5 | 0.002 | 0.001 | 0.001 |
| p92-48 | C | concept | concept | 3/8 | 4 | 1 | False | 0.027 | 0.032 | 1 | -0.009 | -0.006 | -0.003 |
| p92-49 | C | concept | slot | 4/8 | 4 | 0 | False | 0.283 | 0.305 | 6 | 0.029 | 0.014 | 0.015 |
| p92-50 | C | slot | slot | 4/8 | 4 | 0 | False | 0.038 | 0.027 | 1 | -0.002 | 0.000 | -0.002 |
| p92-51 | C | concept | concept | 8/8 | 0 | 0 | True | 3.096 | 0.159 | 8 | 0.403 | -0.000 | 0.403 |
| p92-52 | C | concept | slot | 2/8 | 6 | 0 | False | 0.092 | 0.107 | 2 | 0.018 | 0.019 | -0.000 |

### Rates by Haiku kind and agreed kind

| Frame | Haiku kind | Reads / complete | Rate | Agreed reads / complete | Agreed rate |
| --- | --- | ---: | ---: | ---: | ---: |
| T | concept | 14/18 (n=18) | 0.778 | 6/8 (n=8) | 0.750 |
| T | slot | 4/21 (n=21) | 0.190 | 4/21 (n=21) | 0.190 |
| F | concept | 11/18 (n=18) | 0.611 | 5/8 (n=8) | 0.625 |
| F | slot | 3/21 (n=21) | 0.143 | 3/21 (n=21) | 0.143 |
| C | concept | 10/18 (n=18) | 0.556 | 5/8 (n=8) | 0.625 |
| C | slot | 4/21 (n=21) | 0.190 | 4/21 (n=21) | 0.190 |

### Controls

| Frame | Random reads / cells | Random rate | Shuffled reads / units | Shuffled rate |
| --- | ---: | ---: | ---: | ---: |
| T | 12/117 (n=117) | 0.103 | 8/39 (n=39) | 0.205 |
| F | 11/117 (n=117) | 0.094 | 6/39 (n=39) | 0.154 |
| C | 13/117 (n=117) | 0.111 | 6/39 (n=39) | 0.154 |

### Predictions

| # | Confidence | Result | Verdict |
| --- | --- | --- | --- |
| 122.1 | medium | {'concept_reads': 14, 'complete_units': 18, 'expected_units': 18, 'threshold': 9} | not read (gate 1) |
| 122.2 | medium | {'slot_reads': 4, 'complete_units': 21, 'expected_units': 21, 'threshold': 5} | not read (gate 1) |
| 122.3 | medium | {'concept_rate': 0.7777777777777778, 'slot_rate': 0.19047619047619047, 'fisher_two_sided_p': 0.0003632882773957398, 'threshold': 0.05, 'direction': 'concept > slot'} | not read (gate 1) |
| 122.4 | low | {'slot_T_reads': 4, 'slot_C_reads': 4, 'gain': 0, 'threshold': 5, 'expected_units_per_frame': 21} | not read (gate 1) |
| 122.5 | low | {'slot_gain': 0, 'slot_gain_share': 0.0, 'concept_gain': -4, 'concept_gain_share': -0.2222222222222222, 'comparison': 'slot share > concept share'} | not read (gate 1) |
| 122.6 | medium | {'by_frame': {'T': {'reads': 8, 'complete_units': 39, 'expected_units': 39, 'rate': 0.20512820512820512, 'threshold': 0.1, 'passed': False}, 'F': {'reads': 6, 'complete_units': 39, 'expected_units': 39, 'rate': 0.15384615384615385, 'threshold': 0.1, 'passed': False}, 'C': {'reads': 6, 'complete_units': 39, 'expected_units': 39, 'rate': 0.15384615384615385, 'threshold': 0.1, 'passed': False}}, 'threshold': 0.1} | not read (gate 1) |
| 122.7 | low | {'jlens_units': 11, 'complete_units': 11, 'reads': 10, 'threshold': 7} | not read (gate 1) |

### Registered reading

Nothing is read from these tables because gate1 failed.
