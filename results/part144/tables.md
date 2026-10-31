# Part 144 tables

Bootstrap intervals are 95% percentile intervals over units, with 10,000 resamples and seed 144.

## Shift by cell

| Cell | Set/unit/arm/read | n | Median log-odds shift | Q1 | Q3 | Bootstrap 95% |
|---|---|---:|---:|---:|---:|---|
| base-plain | R|neuron|p_first|full | 96 | -1.46079 | -2.01175 | -1.08723 | -1.64397 to -1.29645 |
| base-plain | R|neuron|p_m|full | 96 | -0.780408 | -1.20774 | -0.290589 | -0.907854 to -0.641751 |
| base-plain | R|neuron|p_sorry|full | 96 | 2.47928 | 1.92355 | 3.3392 | 2.18035 to 2.8001 |
| base-plain | R|neuron|p_first|prompt-only | 96 | -1.46079 | -2.01175 | -1.08723 | -1.64397 to -1.29645 |
| base-plain | R|neuron|p_m|prompt-only | 96 | -0.580148 | -0.93939 | -0.119223 | -0.738996 to -0.380696 |
| base-plain | R|neuron|p_sorry|prompt-only | 96 | 2.55971 | 1.93963 | 3.19119 | 2.33797 to 2.75061 |
| base-plain | R|neuron|p_first|answer-only | 96 | 0 | 0 | 0 | 0 to 0 |
| base-plain | R|neuron|p_m|answer-only | 96 | -0.0918162 | -0.201028 | 0.00382565 | -0.134863 to -0.0585957 |
| base-plain | R|neuron|p_sorry|answer-only | 96 | 1.13776 | 0.755726 | 1.59353 | 0.990353 to 1.35388 |
| base-plain | R|random|p_first|full | 34 | -1.45938 | -1.87025 | -1.14347 | -1.7135 to -1.18918 |
| base-plain | R|random|p_m|full | 34 | -0.443798 | -0.974379 | -0.232825 | -0.911895 to -0.287608 |
| base-plain | R|random|p_sorry|full | 34 | 2.83937 | 2.10684 | 3.31702 | 2.49046 to 3.14817 |
| base-plain | R|random|p_first|prompt-only | 34 | -1.45938 | -1.87025 | -1.14347 | -1.7135 to -1.18918 |
| base-plain | R|random|p_m|prompt-only | 34 | -0.374586 | -0.795653 | 0.0358889 | -0.758703 to -0.151158 |
| base-plain | R|random|p_sorry|prompt-only | 34 | 2.51073 | 2.24864 | 3.06988 | 2.25446 to 2.86497 |
| base-plain | R|random|p_first|answer-only | 34 | 0 | 0 | 0 | 0 to 0 |
| base-plain | R|random|p_m|answer-only | 34 | 0.0115562 | -0.112957 | 0.146486 | -0.0724522 to 0.0870891 |
| base-plain | R|random|p_sorry|answer-only | 34 | 0.984404 | 0.525653 | 1.32184 | 0.620895 to 1.30301 |
| base-plain | S|neuron|p_first|full | 96 | -1.36209 | -1.91531 | -0.855258 | -1.57311 to -1.16165 |
| base-plain | S|neuron|p_m|full | 96 | -0.776103 | -1.46648 | -0.385369 | -1.01033 to -0.53851 |
| base-plain | S|neuron|p_sorry|full | 96 | 2.07158 | 1.34945 | 2.60653 | 1.86125 to 2.37181 |
| base-plain | S|neuron|p_first|prompt-only | 96 | -1.36209 | -1.91531 | -0.855258 | -1.57311 to -1.16165 |
| base-plain | S|neuron|p_m|prompt-only | 96 | -0.553916 | -1.21764 | -0.266661 | -0.809195 to -0.435599 |
| base-plain | S|neuron|p_sorry|prompt-only | 96 | 2.12941 | 1.59684 | 2.59678 | 1.94088 to 2.28859 |
| base-plain | S|neuron|p_first|answer-only | 96 | 0 | 0 | 0 | 0 to 0 |
| base-plain | S|neuron|p_m|answer-only | 96 | -0.0958189 | -0.233228 | 0.0424604 | -0.183071 to -0.0529819 |
| base-plain | S|neuron|p_sorry|answer-only | 96 | 0.717677 | 0.387563 | 1.04555 | 0.618007 to 0.880367 |
| base-plain | F|neuron|p_first|full | 128 | -1.26742 | -1.67016 | -0.995068 | -1.43398 to -1.1995 |
| base-plain | F|neuron|p_m|full | 128 | -0.760846 | -1.41777 | -0.312029 | -1.04296 to -0.599444 |
| base-plain | F|neuron|p_sorry|full | 128 | 1.86736 | 1.02208 | 2.72482 | 1.60474 to 2.24633 |
| base-plain | F|neuron|p_first|prompt-only | 128 | -1.26742 | -1.67016 | -0.995068 | -1.43398 to -1.1995 |
| base-plain | F|neuron|p_m|prompt-only | 128 | -0.581267 | -1.03213 | -0.215297 | -0.791886 to -0.46017 |
| base-plain | F|neuron|p_sorry|prompt-only | 128 | 1.8836 | 1.38844 | 2.55746 | 1.67739 to 2.11164 |
| base-plain | F|neuron|p_first|answer-only | 128 | 0 | 0 | 0 | 0 to 0 |
| base-plain | F|neuron|p_m|answer-only | 128 | -0.112318 | -0.256543 | 0.0362065 | -0.145376 to -0.0586223 |
| base-plain | F|neuron|p_sorry|answer-only | 128 | 0.727094 | 0.295035 | 1.27054 | 0.578308 to 0.839268 |
| base-plain | F|random|p_first|full | 128 | -1.35381 | -1.65736 | -0.96419 | -1.42904 to -1.24061 |
| base-plain | F|random|p_m|full | 128 | -0.794705 | -1.45658 | -0.320326 | -1.00696 to -0.665051 |
| base-plain | F|random|p_sorry|full | 128 | 2.04626 | 1.52729 | 2.6833 | 1.87418 to 2.26256 |
| base-plain | F|random|p_first|prompt-only | 128 | -1.35381 | -1.65736 | -0.96419 | -1.42904 to -1.24061 |
| base-plain | F|random|p_m|prompt-only | 128 | -0.656442 | -1.1309 | -0.174476 | -0.787526 to -0.514173 |
| base-plain | F|random|p_sorry|prompt-only | 128 | 2.15344 | 1.73604 | 2.62837 | 2.02809 to 2.24652 |
| base-plain | F|random|p_first|answer-only | 128 | 0 | 0 | 0 | 0 to 0 |
| base-plain | F|random|p_m|answer-only | 128 | -0.0990652 | -0.248269 | 0.0663683 | -0.158418 to -0.0592227 |
| base-plain | F|random|p_sorry|answer-only | 128 | 0.785332 | 0.317411 | 1.25654 | 0.691705 to 0.97609 |
| base-chat | R|neuron|p_first|full | 96 | -1.17777 | -1.88051 | -0.707274 | -1.46411 to -0.868592 |
| base-chat | R|neuron|p_m|full | 96 | -1.13284 | -1.46907 | -0.796421 | -1.2171 to -1.02327 |
| base-chat | R|neuron|p_sorry|full | 96 | 2.53525 | 1.70449 | 3.37134 | 2.27226 to 2.88407 |
| base-chat | R|neuron|p_first|prompt-only | 96 | -1.17777 | -1.88051 | -0.707274 | -1.46411 to -0.868592 |
| base-chat | R|neuron|p_m|prompt-only | 96 | -0.926427 | -1.25232 | -0.596011 | -1.06219 to -0.835006 |
| base-chat | R|neuron|p_sorry|prompt-only | 96 | 2.93048 | 2.57197 | 3.48479 | 2.77139 to 3.11726 |
| base-chat | R|neuron|p_first|answer-only | 96 | 0 | 0 | 0 | 0 to 0 |
| base-chat | R|neuron|p_m|answer-only | 96 | -0.224559 | -0.360373 | -0.0799307 | -0.296093 to -0.163695 |
| base-chat | R|neuron|p_sorry|answer-only | 96 | 1.65678 | 1.23555 | 2.26313 | 1.4703 to 1.85891 |
| base-chat | R|random|p_first|full | 34 | -1.32196 | -2.06092 | -0.752197 | -1.76371 to -0.910835 |
| base-chat | R|random|p_m|full | 34 | -1.01661 | -1.35848 | -0.675397 | -1.26772 to -0.751715 |
| base-chat | R|random|p_sorry|full | 34 | 2.82726 | 2.17417 | 3.15617 | 2.29934 to 3.08859 |
| base-chat | R|random|p_first|prompt-only | 34 | -1.32196 | -2.06092 | -0.752197 | -1.76371 to -0.910835 |
| base-chat | R|random|p_m|prompt-only | 34 | -0.843624 | -1.13351 | -0.472805 | -0.998064 to -0.634512 |
| base-chat | R|random|p_sorry|prompt-only | 34 | 2.94058 | 2.44486 | 3.33898 | 2.56126 to 3.1837 |
| base-chat | R|random|p_first|answer-only | 34 | 0 | 0 | 0 | 0 to 0 |
| base-chat | R|random|p_m|answer-only | 34 | -0.211952 | -0.309805 | -0.0290828 | -0.275302 to -0.0974662 |
| base-chat | R|random|p_sorry|answer-only | 34 | 1.44764 | 1.13654 | 1.72051 | 1.26313 to 1.66329 |
| base-chat | S|neuron|p_first|full | 96 | -1.22375 | -1.79659 | -0.714024 | -1.46598 to -1.02894 |
| base-chat | S|neuron|p_m|full | 96 | -1.22071 | -1.65282 | -0.870486 | -1.39945 to -1.08032 |
| base-chat | S|neuron|p_sorry|full | 96 | 1.75811 | 1.07246 | 2.67725 | 1.61921 to 2.12616 |
| base-chat | S|neuron|p_first|prompt-only | 96 | -1.22375 | -1.79659 | -0.714024 | -1.46598 to -1.02894 |
| base-chat | S|neuron|p_m|prompt-only | 96 | -0.972058 | -1.26504 | -0.685407 | -1.05698 to -0.864454 |
| base-chat | S|neuron|p_sorry|prompt-only | 96 | 2.3194 | 1.77613 | 2.87376 | 2.13889 to 2.62963 |
| base-chat | S|neuron|p_first|answer-only | 96 | 0 | 0 | 0 | 0 to 0 |
| base-chat | S|neuron|p_m|answer-only | 96 | -0.166227 | -0.336351 | -0.0424336 | -0.226159 to -0.118792 |
| base-chat | S|neuron|p_sorry|answer-only | 96 | 1.06453 | 0.685754 | 1.55089 | 0.876215 to 1.22174 |
| base-chat | F|neuron|p_first|full | 128 | -1.29846 | -1.98165 | -0.8016 | -1.4824 to -1.13554 |
| base-chat | F|neuron|p_m|full | 128 | -1.21644 | -1.5595 | -0.721542 | -1.34542 to -1.03784 |
| base-chat | F|neuron|p_sorry|full | 128 | 1.99771 | 1.01983 | 2.73443 | 1.74394 to 2.24054 |
| base-chat | F|neuron|p_first|prompt-only | 128 | -1.29846 | -1.98165 | -0.8016 | -1.4824 to -1.13554 |
| base-chat | F|neuron|p_m|prompt-only | 128 | -0.973551 | -1.40284 | -0.653185 | -1.08732 to -0.875939 |
| base-chat | F|neuron|p_sorry|prompt-only | 128 | 2.46689 | 1.93484 | 3.05104 | 2.32581 to 2.69052 |
| base-chat | F|neuron|p_first|answer-only | 128 | 0 | 0 | 0 | 0 to 0 |
| base-chat | F|neuron|p_m|answer-only | 128 | -0.156163 | -0.322368 | -0.0328348 | -0.22895 to -0.111286 |
| base-chat | F|neuron|p_sorry|answer-only | 128 | 0.989347 | 0.494072 | 1.57218 | 0.850549 to 1.20532 |
| base-chat | F|random|p_first|full | 128 | -1.2748 | -1.77097 | -0.765639 | -1.41666 to -1.08911 |
| base-chat | F|random|p_m|full | 128 | -1.20162 | -1.66849 | -0.747417 | -1.32473 to -1.026 |
| base-chat | F|random|p_sorry|full | 128 | 2.22513 | 1.45279 | 2.99725 | 1.98067 to 2.44243 |
| base-chat | F|random|p_first|prompt-only | 128 | -1.2748 | -1.77097 | -0.765639 | -1.41666 to -1.08911 |
| base-chat | F|random|p_m|prompt-only | 128 | -0.960688 | -1.32514 | -0.58829 | -1.0929 to -0.84521 |
| base-chat | F|random|p_sorry|prompt-only | 128 | 2.61905 | 2.12902 | 3.03596 | 2.48148 to 2.77987 |
| base-chat | F|random|p_first|answer-only | 128 | 0 | 0 | 0 | 0 to 0 |
| base-chat | F|random|p_m|answer-only | 128 | -0.194082 | -0.326242 | -0.00945609 | -0.243124 to -0.144095 |
| base-chat | F|random|p_sorry|answer-only | 128 | 1.18441 | 0.806703 | 1.62691 | 1.01732 to 1.38856 |
| sft-chat | R|neuron|p_first|full | 96 | -1.27541 | -1.7765 | -0.852902 | -1.44743 to -1.09411 |
| sft-chat | R|neuron|p_m|full | 96 | 2.0028 | 1.67855 | 2.34554 | 1.922 to 2.16962 |
| sft-chat | R|neuron|p_sorry|full | 96 | 5.6688 | 5.23274 | 6.34406 | 5.52694 to 5.98106 |
| sft-chat | R|neuron|p_first|prompt-only | 96 | -1.27541 | -1.7765 | -0.852902 | -1.44743 to -1.09411 |
| sft-chat | R|neuron|p_m|prompt-only | 96 | 1.69846 | 1.249 | 2.05299 | 1.53185 to 1.83686 |
| sft-chat | R|neuron|p_sorry|prompt-only | 96 | 4.942 | 4.20206 | 5.72753 | 4.65571 to 5.10038 |
| sft-chat | R|neuron|p_first|answer-only | 96 | 0 | 0 | 0 | 0 to 0 |
| sft-chat | R|neuron|p_m|answer-only | 96 | 0.68092 | 0.354677 | 1.05204 | 0.617787 to 0.789654 |
| sft-chat | R|neuron|p_sorry|answer-only | 96 | 1.12872 | 0.266053 | 2.02764 | 0.744854 to 1.41433 |
| sft-chat | R|random|p_first|full | 34 | -1.54051 | -2.07205 | -0.926433 | -1.84662 to -1.33176 |
| sft-chat | R|random|p_m|full | 34 | 1.92902 | 1.61159 | 2.20919 | 1.72253 to 2.12069 |
| sft-chat | R|random|p_sorry|full | 34 | 5.8734 | 5.26691 | 6.26472 | 5.48184 to 6.18367 |
| sft-chat | R|random|p_first|prompt-only | 34 | -1.54051 | -2.07205 | -0.926433 | -1.84662 to -1.33176 |
| sft-chat | R|random|p_m|prompt-only | 34 | 1.64348 | 1.17487 | 1.90477 | 1.3626 to 1.82592 |
| sft-chat | R|random|p_sorry|prompt-only | 34 | 4.84712 | 4.44732 | 5.43077 | 4.60796 to 5.23021 |
| sft-chat | R|random|p_first|answer-only | 34 | 0 | 0 | 0 | 0 to 0 |
| sft-chat | R|random|p_m|answer-only | 34 | 0.65079 | 0.366066 | 1.14279 | 0.429897 to 0.990818 |
| sft-chat | R|random|p_sorry|answer-only | 34 | 0.804879 | 0.200528 | 1.84121 | 0.420368 to 1.53541 |
| sft-chat | S|neuron|p_first|full | 96 | -1.16168 | -1.70811 | -0.494662 | -1.35792 to -0.879269 |
| sft-chat | S|neuron|p_m|full | 96 | 0.832791 | 0.250266 | 1.25443 | 0.511125 to 0.958898 |
| sft-chat | S|neuron|p_sorry|full | 96 | 3.26533 | 2.1337 | 4.15613 | 2.85101 to 3.73584 |
| sft-chat | S|neuron|p_first|prompt-only | 96 | -1.16168 | -1.70811 | -0.494662 | -1.35792 to -0.879269 |
| sft-chat | S|neuron|p_m|prompt-only | 96 | 0.666592 | 0.240009 | 1.06755 | 0.422355 to 0.804121 |
| sft-chat | S|neuron|p_sorry|prompt-only | 96 | 2.78529 | 1.68401 | 3.47977 | 2.28493 to 3.09843 |
| sft-chat | S|neuron|p_first|answer-only | 96 | 0 | 0 | 0 | 0 to 0 |
| sft-chat | S|neuron|p_m|answer-only | 96 | 0.49719 | 0.186405 | 0.801581 | 0.367796 to 0.620861 |
| sft-chat | S|neuron|p_sorry|answer-only | 96 | 0.355469 | -0.366705 | 1.16559 | 0.151875 to 0.751648 |
| sft-chat | F|neuron|p_first|full | 128 | -1.10123 | -1.63264 | -0.496905 | -1.25571 to -0.916922 |
| sft-chat | F|neuron|p_m|full | 128 | 1.06209 | 0.460202 | 1.64728 | 0.849651 to 1.29109 |
| sft-chat | F|neuron|p_sorry|full | 128 | 3.14052 | 1.96285 | 4.37955 | 2.42612 to 3.70201 |
| sft-chat | F|neuron|p_first|prompt-only | 128 | -1.10123 | -1.63264 | -0.496905 | -1.25571 to -0.916922 |
| sft-chat | F|neuron|p_m|prompt-only | 128 | 0.91537 | 0.38085 | 1.49649 | 0.713255 to 1.18402 |
| sft-chat | F|neuron|p_sorry|prompt-only | 128 | 2.76335 | 1.48512 | 4.22564 | 2.35672 to 3.29128 |
| sft-chat | F|neuron|p_first|answer-only | 128 | 0 | 0 | 0 | 0 to 0 |
| sft-chat | F|neuron|p_m|answer-only | 128 | 0.555486 | 0.187431 | 0.918999 | 0.488785 to 0.623495 |
| sft-chat | F|neuron|p_sorry|answer-only | 128 | 0.614588 | -0.345797 | 1.60309 | 0.213198 to 0.868136 |
| sft-chat | F|random|p_first|full | 128 | -1.06863 | -1.63653 | -0.653842 | -1.28946 to -0.919361 |
| sft-chat | F|random|p_m|full | 128 | 1.15322 | 0.557728 | 1.68081 | 0.95461 to 1.34258 |
| sft-chat | F|random|p_sorry|full | 128 | 3.44155 | 2.44425 | 4.50337 | 3.11452 to 3.78134 |
| sft-chat | F|random|p_first|prompt-only | 128 | -1.06863 | -1.63653 | -0.653842 | -1.28946 to -0.919361 |
| sft-chat | F|random|p_m|prompt-only | 128 | 0.967589 | 0.392639 | 1.53253 | 0.801395 to 1.14323 |
| sft-chat | F|random|p_sorry|prompt-only | 128 | 2.90928 | 2.03545 | 4.04238 | 2.73357 to 3.19148 |
| sft-chat | F|random|p_first|answer-only | 128 | 0 | 0 | 0 | 0 to 0 |
| sft-chat | F|random|p_m|answer-only | 128 | 0.498833 | 0.123244 | 0.942427 | 0.365327 to 0.618517 |
| sft-chat | F|random|p_sorry|answer-only | 128 | 0.369083 | -0.356813 | 1.46639 | 0.127204 to 0.596866 |
| dpo-chat | R|neuron|p_first|full | 96 | -0.727896 | -1.2385 | -0.180103 | -0.921642 to -0.541448 |
| dpo-chat | R|neuron|p_m|full | 96 | 1.66464 | 1.35408 | 2.02169 | 1.49334 to 1.80993 |
| dpo-chat | R|neuron|p_sorry|full | 96 | 7.45004 | 6.95521 | 8.15041 | 7.2231 to 7.74637 |
| dpo-chat | R|neuron|p_first|prompt-only | 96 | -0.727896 | -1.2385 | -0.180103 | -0.921642 to -0.541448 |
| dpo-chat | R|neuron|p_m|prompt-only | 96 | 1.42182 | 1.02342 | 1.76177 | 1.25609 to 1.51231 |
| dpo-chat | R|neuron|p_sorry|prompt-only | 96 | 6.25379 | 5.37946 | 7.18354 | 5.793 to 6.78505 |
| dpo-chat | R|neuron|p_first|answer-only | 96 | 0 | 0 | 0 | 0 to 0 |
| dpo-chat | R|neuron|p_m|answer-only | 96 | 0.42363 | 0.175052 | 0.727224 | 0.361408 to 0.514271 |
| dpo-chat | R|neuron|p_sorry|answer-only | 96 | 1.19856 | 0.462272 | 2.12627 | 0.887945 to 1.45839 |
| dpo-chat | R|random|p_first|full | 34 | -0.766702 | -1.22003 | -0.170468 | -1.10208 to -0.500584 |
| dpo-chat | R|random|p_m|full | 34 | 1.52728 | 1.36662 | 1.71177 | 1.44162 to 1.6675 |
| dpo-chat | R|random|p_sorry|full | 34 | 7.32794 | 7.01792 | 7.86315 | 7.08966 to 7.72168 |
| dpo-chat | R|random|p_first|prompt-only | 34 | -0.766702 | -1.22003 | -0.170468 | -1.10208 to -0.500584 |
| dpo-chat | R|random|p_m|prompt-only | 34 | 1.28193 | 0.964834 | 1.54782 | 1.0599 to 1.48402 |
| dpo-chat | R|random|p_sorry|prompt-only | 34 | 6.12582 | 5.88196 | 6.86043 | 5.97401 to 6.61706 |
| dpo-chat | R|random|p_first|answer-only | 34 | 0 | 0 | 0 | 0 to 0 |
| dpo-chat | R|random|p_m|answer-only | 34 | 0.436176 | 0.0727189 | 0.67883 | 0.199424 to 0.545422 |
| dpo-chat | R|random|p_sorry|answer-only | 34 | 0.948997 | 0.227369 | 1.74923 | 0.437171 to 1.45962 |
| dpo-chat | S|neuron|p_first|full | 96 | -0.419551 | -1.02029 | 0.177679 | -0.656749 to -0.16979 |
| dpo-chat | S|neuron|p_m|full | 96 | 0.657897 | 0.0784023 | 1.03354 | 0.319723 to 0.794344 |
| dpo-chat | S|neuron|p_sorry|full | 96 | 4.6185 | 3.21306 | 5.68448 | 4.09773 to 4.96719 |
| dpo-chat | S|neuron|p_first|prompt-only | 96 | -0.419551 | -1.02029 | 0.177679 | -0.656749 to -0.16979 |
| dpo-chat | S|neuron|p_m|prompt-only | 96 | 0.476122 | -0.0400175 | 0.879286 | 0.276797 to 0.63563 |
| dpo-chat | S|neuron|p_sorry|prompt-only | 96 | 3.72752 | 2.56015 | 4.61544 | 3.0664 to 4.13911 |
| dpo-chat | S|neuron|p_first|answer-only | 96 | 0 | 0 | 0 | 0 to 0 |
| dpo-chat | S|neuron|p_m|answer-only | 96 | 0.3189 | 0.062321 | 0.589003 | 0.234385 to 0.419339 |
| dpo-chat | S|neuron|p_sorry|answer-only | 96 | 0.669308 | -0.268846 | 1.42785 | 0.372843 to 1.07431 |
| dpo-chat | F|neuron|p_first|full | 128 | -0.428806 | -0.955628 | 0.0241162 | -0.614515 to -0.341816 |
| dpo-chat | F|neuron|p_m|full | 128 | 0.810283 | 0.403533 | 1.56884 | 0.608049 to 1.04112 |
| dpo-chat | F|neuron|p_sorry|full | 128 | 4.34425 | 2.99243 | 5.76654 | 3.68628 to 5.0227 |
| dpo-chat | F|neuron|p_first|prompt-only | 128 | -0.428806 | -0.955628 | 0.0241162 | -0.614515 to -0.341816 |
| dpo-chat | F|neuron|p_m|prompt-only | 128 | 0.639489 | 0.207929 | 1.39256 | 0.500305 to 0.906791 |
| dpo-chat | F|neuron|p_sorry|prompt-only | 128 | 3.85318 | 2.32462 | 5.3486 | 3.25528 to 4.0584 |
| dpo-chat | F|neuron|p_first|answer-only | 128 | 0 | 0 | 0 | 0 to 0 |
| dpo-chat | F|neuron|p_m|answer-only | 128 | 0.408727 | 0.119949 | 0.606905 | 0.310126 to 0.424644 |
| dpo-chat | F|neuron|p_sorry|answer-only | 128 | 0.58303 | -0.103579 | 1.54516 | 0.424986 to 0.980124 |
| dpo-chat | F|random|p_first|full | 128 | -0.386038 | -0.91455 | 0.0432695 | -0.539115 to -0.248235 |
| dpo-chat | F|random|p_m|full | 128 | 0.94267 | 0.351821 | 1.46611 | 0.745569 to 1.13402 |
| dpo-chat | F|random|p_sorry|full | 128 | 4.57794 | 3.26093 | 5.90773 | 4.31861 to 4.96268 |
| dpo-chat | F|random|p_first|prompt-only | 128 | -0.386038 | -0.91455 | 0.0432695 | -0.539115 to -0.248235 |
| dpo-chat | F|random|p_m|prompt-only | 128 | 0.729547 | 0.220783 | 1.28913 | 0.579931 to 0.932968 |
| dpo-chat | F|random|p_sorry|prompt-only | 128 | 3.74625 | 2.6279 | 5.04258 | 3.36542 to 4.09178 |
| dpo-chat | F|random|p_first|answer-only | 128 | 0 | 0 | 0 | 0 to 0 |
| dpo-chat | F|random|p_m|answer-only | 128 | 0.353764 | -0.00128497 | 0.726213 | 0.241062 to 0.451469 |
| dpo-chat | F|random|p_sorry|answer-only | 128 | 0.632502 | -0.157196 | 1.60026 | 0.273512 to 0.906729 |
| instruct-chat | R|neuron|p_first|full | 96 | -0.853005 | -1.34826 | -0.422991 | -0.993967 to -0.716917 |
| instruct-chat | R|neuron|p_m|full | 96 | 1.63403 | 1.28689 | 1.97611 | 1.4524 to 1.76426 |
| instruct-chat | R|neuron|p_sorry|full | 96 | 8.15348 | 7.57172 | 8.72172 | 7.90272 to 8.31991 |
| instruct-chat | R|neuron|p_first|prompt-only | 96 | -0.853005 | -1.34826 | -0.422991 | -0.993967 to -0.716917 |
| instruct-chat | R|neuron|p_m|prompt-only | 96 | 1.41434 | 1.02293 | 1.75251 | 1.23952 to 1.50644 |
| instruct-chat | R|neuron|p_sorry|prompt-only | 96 | 6.74371 | 5.9748 | 7.8456 | 6.36243 to 7.23066 |
| instruct-chat | R|neuron|p_first|answer-only | 96 | 0 | 0 | 0 | 0 to 0 |
| instruct-chat | R|neuron|p_m|answer-only | 96 | 0.36716 | 0.112814 | 0.61931 | 0.321995 to 0.428295 |
| instruct-chat | R|neuron|p_sorry|answer-only | 96 | 1.43321 | 0.575121 | 2.27781 | 1.07903 to 1.67494 |
| instruct-chat | R|random|p_first|full | 34 | -1.06386 | -1.40877 | -0.354292 | -1.30014 to -0.671017 |
| instruct-chat | R|random|p_m|full | 34 | 1.52831 | 1.35339 | 1.80872 | 1.38027 to 1.59095 |
| instruct-chat | R|random|p_sorry|full | 34 | 7.92534 | 7.61346 | 8.60246 | 7.71689 to 8.32207 |
| instruct-chat | R|random|p_first|prompt-only | 34 | -1.06386 | -1.40877 | -0.354292 | -1.30014 to -0.671017 |
| instruct-chat | R|random|p_m|prompt-only | 34 | 1.30759 | 0.976673 | 1.57878 | 1.07167 to 1.47294 |
| instruct-chat | R|random|p_sorry|prompt-only | 34 | 6.74306 | 6.43284 | 7.45199 | 6.51946 to 7.34796 |
| instruct-chat | R|random|p_first|answer-only | 34 | 0 | 0 | 0 | 0 to 0 |
| instruct-chat | R|random|p_m|answer-only | 34 | 0.426675 | 0.0742671 | 0.644407 | 0.258564 to 0.495018 |
| instruct-chat | R|random|p_sorry|answer-only | 34 | 1.16492 | 0.464549 | 1.98161 | 0.692043 to 1.59984 |
| instruct-chat | S|neuron|p_first|full | 96 | -0.676421 | -1.14796 | -0.0240518 | -0.888938 to -0.404691 |
| instruct-chat | S|neuron|p_m|full | 96 | 0.619778 | 0.120041 | 1.04591 | 0.402846 to 0.760716 |
| instruct-chat | S|neuron|p_sorry|full | 96 | 5.28412 | 3.81613 | 6.33447 | 4.60923 to 5.58279 |
| instruct-chat | S|neuron|p_first|prompt-only | 96 | -0.676421 | -1.14796 | -0.0240518 | -0.888938 to -0.404691 |
| instruct-chat | S|neuron|p_m|prompt-only | 96 | 0.466878 | -0.0253533 | 0.872823 | 0.37888 to 0.607748 |
| instruct-chat | S|neuron|p_sorry|prompt-only | 96 | 4.3424 | 3.11651 | 5.34458 | 3.67615 to 4.72673 |
| instruct-chat | S|neuron|p_first|answer-only | 96 | 0 | 0 | 0 | 0 to 0 |
| instruct-chat | S|neuron|p_m|answer-only | 96 | 0.29327 | 0.0614159 | 0.549148 | 0.185535 to 0.372791 |
| instruct-chat | S|neuron|p_sorry|answer-only | 96 | 0.876343 | 0.0557464 | 1.65058 | 0.528191 to 1.21903 |
| instruct-chat | F|neuron|p_first|full | 128 | -0.614586 | -1.15339 | -0.119413 | -0.765231 to -0.504114 |
| instruct-chat | F|neuron|p_m|full | 128 | 0.816993 | 0.387511 | 1.54437 | 0.637646 to 1.04178 |
| instruct-chat | F|neuron|p_sorry|full | 128 | 5.06186 | 3.49228 | 6.4519 | 4.39231 to 5.58696 |
| instruct-chat | F|neuron|p_first|prompt-only | 128 | -0.614586 | -1.15339 | -0.119413 | -0.765231 to -0.504114 |
| instruct-chat | F|neuron|p_m|prompt-only | 128 | 0.625483 | 0.204381 | 1.33219 | 0.470405 to 0.884339 |
| instruct-chat | F|neuron|p_sorry|prompt-only | 128 | 4.41454 | 2.64229 | 5.88407 | 3.77248 to 4.64778 |
| instruct-chat | F|neuron|p_first|answer-only | 128 | 0 | 0 | 0 | 0 to 0 |
| instruct-chat | F|neuron|p_m|answer-only | 128 | 0.36479 | 0.0589182 | 0.552935 | 0.280216 to 0.426175 |
| instruct-chat | F|neuron|p_sorry|answer-only | 128 | 0.782951 | -0.0527487 | 1.65547 | 0.681494 to 1.1636 |
| instruct-chat | F|random|p_first|full | 128 | -0.603275 | -1.08979 | -0.137561 | -0.738588 to -0.493783 |
| instruct-chat | F|random|p_m|full | 128 | 0.926135 | 0.351319 | 1.43038 | 0.808082 to 1.09349 |
| instruct-chat | F|random|p_sorry|full | 128 | 5.31707 | 3.85604 | 6.4572 | 4.86933 to 5.64186 |
| instruct-chat | F|random|p_first|prompt-only | 128 | -0.603275 | -1.08979 | -0.137561 | -0.738588 to -0.493783 |
| instruct-chat | F|random|p_m|prompt-only | 128 | 0.744851 | 0.19988 | 1.18021 | 0.569014 to 0.876135 |
| instruct-chat | F|random|p_sorry|prompt-only | 128 | 4.38065 | 3.17528 | 5.554 | 3.83536 to 4.77923 |
| instruct-chat | F|random|p_first|answer-only | 128 | 0 | 0 | 0 | 0 to 0 |
| instruct-chat | F|random|p_m|answer-only | 128 | 0.295334 | -0.00583352 | 0.617216 | 0.18535 to 0.435365 |
| instruct-chat | F|random|p_sorry|answer-only | 128 | 0.774937 | 0.0454577 | 1.80362 | 0.435346 to 1.10968 |
| instruct-plain | R|neuron|p_first|full | 96 | -2.02765 | -2.90236 | -1.4402 | -2.53116 to -1.80821 |
| instruct-plain | R|neuron|p_m|full | 96 | 0.193259 | -0.412658 | 0.613658 | -0.0608496 to 0.363719 |
| instruct-plain | R|neuron|p_sorry|full | 96 | 1.92123 | 0.994028 | 2.42074 | 1.58847 to 2.15991 |
| instruct-plain | R|neuron|p_first|prompt-only | 96 | -2.02765 | -2.90236 | -1.4402 | -2.53116 to -1.80821 |
| instruct-plain | R|neuron|p_m|prompt-only | 96 | 0.17056 | -0.16063 | 0.716116 | 0.00513097 to 0.294963 |
| instruct-plain | R|neuron|p_sorry|prompt-only | 96 | 1.79621 | 1.13258 | 2.48108 | 1.54003 to 2.03365 |
| instruct-plain | R|neuron|p_first|answer-only | 96 | 0 | 0 | 0 | 0 to 0 |
| instruct-plain | R|neuron|p_m|answer-only | 96 | 0.430652 | 0.253811 | 0.620527 | 0.374661 to 0.502715 |
| instruct-plain | R|neuron|p_sorry|answer-only | 96 | 1.44073 | 0.939356 | 1.94955 | 1.29818 to 1.64245 |
| instruct-plain | R|random|p_first|full | 34 | -2.11068 | -2.88671 | -1.66292 | -2.60989 to -1.81084 |
| instruct-plain | R|random|p_m|full | 34 | 0.241809 | -0.326666 | 0.563098 | -0.155667 to 0.442152 |
| instruct-plain | R|random|p_sorry|full | 34 | 1.58229 | 0.245947 | 2.87996 | 0.803782 to 2.41336 |
| instruct-plain | R|random|p_first|prompt-only | 34 | -2.11068 | -2.88671 | -1.66292 | -2.60989 to -1.81084 |
| instruct-plain | R|random|p_m|prompt-only | 34 | 0.355203 | -0.192216 | 0.702997 | -0.0403293 to 0.499496 |
| instruct-plain | R|random|p_sorry|prompt-only | 34 | 1.75661 | 1.20031 | 2.51114 | 1.55976 to 2.20008 |
| instruct-plain | R|random|p_first|answer-only | 34 | 0 | 0 | 0 | 0 to 0 |
| instruct-plain | R|random|p_m|answer-only | 34 | 0.414175 | 0.180606 | 0.622655 | 0.323203 to 0.527927 |
| instruct-plain | R|random|p_sorry|answer-only | 34 | 1.27009 | 0.553709 | 1.84616 | 0.806587 to 1.71891 |
| instruct-plain | S|neuron|p_first|full | 96 | -1.8388 | -2.74074 | -1.04493 | -2.07563 to -1.6028 |
| instruct-plain | S|neuron|p_m|full | 96 | -0.127502 | -0.913179 | 0.289989 | -0.357682 to 0.0509891 |
| instruct-plain | S|neuron|p_sorry|full | 96 | 0.197661 | -0.654816 | 1.4906 | -0.0641321 to 0.862764 |
| instruct-plain | S|neuron|p_first|prompt-only | 96 | -1.8388 | -2.74074 | -1.04493 | -2.07563 to -1.6028 |
| instruct-plain | S|neuron|p_m|prompt-only | 96 | -0.102749 | -0.630474 | 0.366908 | -0.354132 to 0.068746 |
| instruct-plain | S|neuron|p_sorry|prompt-only | 96 | 0.908936 | 0.221504 | 1.82471 | 0.710244 to 1.02673 |
| instruct-plain | S|neuron|p_first|answer-only | 96 | 0 | 0 | 0 | 0 to 0 |
| instruct-plain | S|neuron|p_m|answer-only | 96 | 0.307979 | 0.0600813 | 0.541981 | 0.24357 to 0.36586 |
| instruct-plain | S|neuron|p_sorry|answer-only | 96 | 0.79854 | 0.399912 | 1.45674 | 0.660215 to 0.982131 |
| instruct-plain | F|neuron|p_first|full | 128 | -1.84706 | -2.72666 | -1.05108 | -2.17978 to -1.54906 |
| instruct-plain | F|neuron|p_m|full | 128 | -0.0898914 | -0.57225 | 0.465567 | -0.222305 to 0.0277328 |
| instruct-plain | F|neuron|p_sorry|full | 128 | 0.245391 | -0.613259 | 1.45966 | 0.0160861 to 0.556618 |
| instruct-plain | F|neuron|p_first|prompt-only | 128 | -1.84706 | -2.72666 | -1.05108 | -2.17978 to -1.54906 |
| instruct-plain | F|neuron|p_m|prompt-only | 128 | 0.0182605 | -0.397408 | 0.440529 | -0.130588 to 0.189165 |
| instruct-plain | F|neuron|p_sorry|prompt-only | 128 | 0.72166 | -0.223036 | 1.57143 | 0.470512 to 0.938561 |
| instruct-plain | F|neuron|p_first|answer-only | 128 | 0 | 0 | 0 | 0 to 0 |
| instruct-plain | F|neuron|p_m|answer-only | 128 | 0.281109 | 0.0565968 | 0.538168 | 0.202863 to 0.346708 |
| instruct-plain | F|neuron|p_sorry|answer-only | 128 | 0.887156 | 0.336916 | 1.55681 | 0.703662 to 1.07788 |
| instruct-plain | F|random|p_first|full | 128 | -1.87996 | -2.73685 | -1.20396 | -2.17522 to -1.5259 |
| instruct-plain | F|random|p_m|full | 128 | -0.0444051 | -0.60074 | 0.50621 | -0.19743 to 0.19662 |
| instruct-plain | F|random|p_sorry|full | 128 | 0.759058 | -0.208105 | 1.77104 | 0.504321 to 1.14091 |
| instruct-plain | F|random|p_first|prompt-only | 128 | -1.87996 | -2.73685 | -1.20396 | -2.17522 to -1.5259 |
| instruct-plain | F|random|p_m|prompt-only | 128 | 0.119584 | -0.355791 | 0.583522 | -0.00440981 to 0.278538 |
| instruct-plain | F|random|p_sorry|prompt-only | 128 | 1.06033 | 0.263278 | 1.73562 | 0.79618 to 1.27942 |
| instruct-plain | F|random|p_first|answer-only | 128 | 0 | 0 | 0 | 0 to 0 |
| instruct-plain | F|random|p_m|answer-only | 128 | 0.349277 | 0.0706126 | 0.587739 | 0.246259 to 0.391171 |
| instruct-plain | F|random|p_sorry|answer-only | 128 | 1.07204 | 0.471319 | 1.45435 | 0.857487 to 1.20799 |

## README main-table intervals

Set F intervals combine neurons and matched random directions.

| Cell | Full P(first) | Full P('m') | Full P(' sorry') | Prompt-only P(' sorry') | Answer-only P(' sorry') |
|---|---:|---:|---:|---:|---:|
| base-plain | -1.32456 (-1.41386 to -1.24451) | -0.784929 (-0.960245 to -0.702435) | 1.98019 (1.84523 to 2.20372) | 2.05558 (1.9216 to 2.17588) | 0.756304 (0.687158 to 0.857116) |
| base-chat | -1.28418 (-1.40955 to -1.1612) | -1.20719 (-1.30487 to -1.09147) | 2.11396 (1.96289 to 2.29962) | 2.56217 (2.45347 to 2.69338) | 1.09159 (0.983022 to 1.23714) |
| sft-chat | -1.08053 (-1.23996 to -0.95306) | 1.12436 (0.945401 to 1.27307) | 3.34402 (3.02432 to 3.58623) | 2.85599 (2.60913 to 3.07293) | 0.493449 (0.232664 to 0.645301) |
| dpo-chat | -0.409512 (-0.53601 to -0.350613) | 0.890034 (0.723167 to 1.03223) | 4.48118 (4.21611 to 4.88778) | 3.80784 (3.44518 to 4.03353) | 0.605197 (0.442672 to 0.88577) |
| instruct-chat | -0.610694 (-0.694361 to -0.533465) | 0.876169 (0.79256 to 1.00869) | 5.20283 (4.78565 to 5.49042) | 4.39446 (3.90437 to 4.62383) | 0.778264 (0.683084 to 1.07337) |
| instruct-plain | -1.86846 (-2.11292 to -1.60164) | -0.0600815 (-0.164776 to 0.046519) | 0.532744 (0.30759 to 0.745582) | 0.854161 (0.715759 to 1.0724) | 0.997117 (0.826632 to 1.09368) |

## Plain probabilities

| Cell | P(first) | P('m') | P(' sorry') |
|---|---:|---:|---:|
| base-plain | 0.41344476 | 0.30151308 | 0.0028879477 |
| base-chat | 0.10331618 | 0.43112305 | 0.0048468355 |
| sft-chat | 0.82923192 | 0.19126442 | 0.005755953 |
| dpo-chat | 0.63897866 | 0.15397978 | 0.00052913133 |
| instruct-chat | 0.65931201 | 0.13887511 | 0.00023909577 |
| instruct-plain | 0.4282825 | 0.18342414 | 0.054604799 |

## Additivity gaps

| Cell | Set/read | n | Median log-odds gap | Q1 | Q3 |
|---|---|---:|---:|---:|---:|
| base-plain | R|p_first | 130 | 0 | 0 | 0 |
| base-plain | R|p_m | 130 | -0.123307 | -0.261145 | 0.0444293 |
| base-plain | R|p_sorry | 130 | -1.02029 | -1.58837 | -0.512092 |
| base-plain | S|p_first | 96 | 0 | 0 | 0 |
| base-plain | S|p_m | 96 | -0.0868051 | -0.22076 | 0.05982 |
| base-plain | S|p_sorry | 96 | -0.732554 | -1.19249 | -0.374401 |
| base-plain | F|p_first | 256 | 0 | 0 | 0 |
| base-plain | F|p_m | 256 | -0.0636308 | -0.219157 | 0.0703834 |
| base-plain | F|p_sorry | 256 | -0.825248 | -1.26166 | -0.432657 |
| base-chat | R|p_first | 130 | 0 | 0 | 0 |
| base-chat | R|p_m | 130 | 0.0215474 | -0.183561 | 0.261934 |
| base-chat | R|p_sorry | 130 | -1.85312 | -2.71027 | -1.38816 |
| base-chat | S|p_first | 96 | 0 | 0 | 0 |
| base-chat | S|p_m | 96 | -0.0855845 | -0.265241 | 0.108868 |
| base-chat | S|p_sorry | 96 | -1.43177 | -2.12074 | -1.0476 |
| base-chat | F|p_first | 256 | 0 | 0 | 0 |
| base-chat | F|p_m | 256 | -0.0473565 | -0.250442 | 0.164769 |
| base-chat | F|p_sorry | 256 | -1.48491 | -2.15773 | -0.985708 |
| sft-chat | R|p_first | 130 | 0 | 0 | 0 |
| sft-chat | R|p_m | 130 | -0.356438 | -0.782497 | 0.0689007 |
| sft-chat | R|p_sorry | 130 | -0.382183 | -1.01137 | 0.527409 |
| sft-chat | S|p_first | 96 | 0 | 0 | 0 |
| sft-chat | S|p_m | 96 | -0.334897 | -0.884713 | 0.00460723 |
| sft-chat | S|p_sorry | 96 | -0.0649943 | -0.947833 | 0.709401 |
| sft-chat | F|p_first | 256 | 0 | 0 | 0 |
| sft-chat | F|p_m | 256 | -0.353846 | -0.769398 | -0.00062553 |
| sft-chat | F|p_sorry | 256 | -0.110394 | -0.984019 | 0.782925 |
| dpo-chat | R|p_first | 130 | 0 | 0 | 0 |
| dpo-chat | R|p_m | 130 | -0.146752 | -0.419443 | 0.282662 |
| dpo-chat | R|p_sorry | 130 | 0.0664399 | -0.785948 | 1.02226 |
| dpo-chat | S|p_first | 96 | 0 | 0 | 0 |
| dpo-chat | S|p_m | 96 | -0.150698 | -0.529056 | 0.0777822 |
| dpo-chat | S|p_sorry | 96 | 0.0370292 | -0.816977 | 1.01644 |
| dpo-chat | F|p_first | 256 | 0 | 0 | 0 |
| dpo-chat | F|p_m | 256 | -0.160884 | -0.495506 | 0.1188 |
| dpo-chat | F|p_sorry | 256 | -0.0203209 | -0.790388 | 0.950806 |
| instruct-chat | R|p_first | 130 | 0 | 0 | 0 |
| instruct-chat | R|p_m | 130 | -0.0913776 | -0.361539 | 0.206126 |
| instruct-chat | R|p_sorry | 130 | -0.0241881 | -0.999464 | 0.926032 |
| instruct-chat | S|p_first | 96 | 0 | 0 | 0 |
| instruct-chat | S|p_m | 96 | -0.11206 | -0.486803 | 0.0894037 |
| instruct-chat | S|p_sorry | 96 | -0.0130755 | -0.897596 | 0.976877 |
| instruct-chat | F|p_first | 256 | 0 | 0 | 0 |
| instruct-chat | F|p_m | 256 | -0.0885509 | -0.431316 | 0.148733 |
| instruct-chat | F|p_sorry | 256 | -0.0419403 | -0.916988 | 0.851509 |
| instruct-plain | R|p_first | 130 | 0 | 0 | 0 |
| instruct-plain | R|p_m | 130 | -0.481126 | -1.01554 | -0.161464 |
| instruct-plain | R|p_sorry | 130 | -1.37551 | -2.30344 | -0.600458 |
| instruct-plain | S|p_first | 96 | 0 | 0 | 0 |
| instruct-plain | S|p_m | 96 | -0.477734 | -0.90048 | -0.0940054 |
| instruct-plain | S|p_sorry | 96 | -1.39233 | -2.10933 | -0.477855 |
| instruct-plain | F|p_first | 256 | 0 | 0 | 0 |
| instruct-plain | F|p_m | 256 | -0.396773 | -0.728154 | -0.0224834 |
| instruct-plain | F|p_sorry | 256 | -1.23488 | -2.06374 | -0.309138 |
