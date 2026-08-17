# Phase-2 calibration coverage audit

Read-only enumeration of the existing data. Denominators: per GPU the full config grid is |TP| x |Freq| x |IL| x |OL| (rate is a sweep dimension used to *derive* decode capacity, not part of a config). `mem_freq` maps 1:1 to `gpu_freq`, so it is not an independent axis.

## Step 1 — Calibration space (unique values per axis)

### GPU = L40S
- **TP** (3): [1, 2, 4]
- **FREQ** (10): [210, 480, 735, 990, 1245, 1500, 1755, 2010, 2265, 2520]
- **IL** (11): [32, 128, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
- **OL** (4): [32, 128, 512, 1024]
- **RATE** (5): [1.0, 10.0, 20.0, 30.0, 50.0]

### GPU = L4
- **TP** (4): [1, 2, 4, 8]
- **FREQ** (10): [210, 360, 570, 780, 990, 1200, 1410, 1620, 1830, 2040]
- **IL** (11): [32, 128, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
- **OL** (4): [32, 128, 512, 1024]
- **RATE** (5): [1.0, 10.0, 20.0, 30.0, 50.0]

## Step 2 — Coverage matrix (TP x Freq): #IL, #OL, #rate-points, #configs

### GPU = L40S

| TP | Freq | #IL | #OL | #(IL,OL) | rate-pts (min..max) | #configs |
|---|---|---|---|---|---|---|
| 1 | 210 | 7 | 4 | 11 | 1..1 | 11 |
| 1 | 480 | 7 | 4 | 14 | 1..1 | 14 |
| 1 | 735 | 7 | 4 | 13 | 1..2 | 15 |
| 1 | 990 | 7 | 4 | 14 | 1..2 | 16 |
| 1 | 1245 | 7 | 4 | 14 | 1..2 | 16 |
| 1 | 1500 | 7 | 4 | 9 | 1..2 | 10 |
| 1 | 1755 | 6 | 4 | 6 | 1..1 | 6 |
| 1 | 2010 | 6 | 4 | 6 | 1..1 | 6 |
| 1 | 2265 | 6 | 4 | 6 | 1..1 | 6 |
| 1 | 2520 | 11 | 4 | 44 | 3..3 | 132 |
| 2 | 210 | 5 | 4 | 5 | 1..2 | 7 |
| 2 | 480 | 5 | 4 | 6 | 1..2 | 8 |
| 2 | 735 | 5 | 4 | 6 | 1..2 | 9 |
| 2 | 990 | 5 | 4 | 8 | 1..2 | 10 |
| 2 | 1245 | 6 | 4 | 8 | 1..2 | 10 |
| 2 | 1500 | 5 | 4 | 5 | 1..2 | 7 |
| 2 | 1755 | 5 | 4 | 5 | 1..3 | 8 |
| 2 | 2010 | 5 | 4 | 6 | 1..2 | 8 |
| 2 | 2265 | 5 | 4 | 5 | 1..2 | 7 |
| 2 | 2520 | 6 | 4 | 9 | 2..4 | 20 |
| 4 | 210 | 5 | 2 | 6 | 1..4 | 15 |
| 4 | 480 | 5 | 4 | 11 | 1..4 | 22 |
| 4 | 735 | 7 | 4 | 11 | 1..5 | 21 |
| 4 | 990 | 6 | 3 | 8 | 1..4 | 17 |
| 4 | 1245 | 7 | 3 | 9 | 1..4 | 19 |
| 4 | 1500 | 5 | 2 | 6 | 1..4 | 15 |
| 4 | 1755 | 5 | 2 | 6 | 1..4 | 15 |
| 4 | 2010 | 5 | 2 | 6 | 1..4 | 15 |
| 4 | 2265 | 6 | 3 | 7 | 1..4 | 16 |
| 4 | 2520 | 7 | 4 | 10 | 2..5 | 28 |

### GPU = L4

| TP | Freq | #IL | #OL | #(IL,OL) | rate-pts (min..max) | #configs |
|---|---|---|---|---|---|---|
| 1 | 210 | 3 | 4 | 8 | 1..2 | 10 |
| 1 | 360 | 4 | 4 | 7 | 1..1 | 7 |
| 1 | 570 | 1 | 1 | 1 | 1..1 | 1 |
| 1 | 780 | 0 | 0 | 0 | - | 0 |
| 1 | 990 | 0 | 0 | 0 | - | 0 |
| 1 | 1200 | 1 | 1 | 1 | 1..1 | 1 |
| 1 | 1410 | 0 | 0 | 0 | - | 0 |
| 1 | 1620 | 0 | 0 | 0 | - | 0 |
| 1 | 1830 | 0 | 0 | 0 | - | 0 |
| 1 | 2040 | 11 | 4 | 44 | 3..3 | 132 |
| 2 | 210 | 3 | 1 | 3 | 2..2 | 6 |
| 2 | 360 | 4 | 2 | 5 | 1..2 | 8 |
| 2 | 570 | 3 | 1 | 3 | 2..2 | 6 |
| 2 | 780 | 3 | 3 | 5 | 1..2 | 8 |
| 2 | 990 | 3 | 1 | 3 | 2..2 | 6 |
| 2 | 1200 | 3 | 1 | 3 | 2..2 | 6 |
| 2 | 1410 | 3 | 1 | 3 | 2..2 | 6 |
| 2 | 1620 | 5 | 3 | 5 | 1..2 | 8 |
| 2 | 1830 | 3 | 1 | 3 | 2..2 | 6 |
| 2 | 2040 | 5 | 3 | 9 | 2..4 | 22 |
| 4 | 210 | 6 | 3 | 9 | 1..4 | 17 |
| 4 | 360 | 7 | 3 | 10 | 1..4 | 18 |
| 4 | 570 | 7 | 3 | 11 | 1..4 | 19 |
| 4 | 780 | 6 | 3 | 10 | 1..4 | 18 |
| 4 | 990 | 5 | 2 | 7 | 1..4 | 15 |
| 4 | 1200 | 5 | 2 | 8 | 1..4 | 17 |
| 4 | 1410 | 5 | 2 | 7 | 1..4 | 15 |
| 4 | 1620 | 7 | 4 | 12 | 1..4 | 20 |
| 4 | 1830 | 5 | 2 | 7 | 1..4 | 15 |
| 4 | 2040 | 5 | 4 | 11 | 2..5 | 29 |
| 8 | 210 | 5 | 1 | 5 | 1..4 | 12 |
| 8 | 360 | 6 | 2 | 7 | 1..4 | 14 |
| 8 | 570 | 7 | 3 | 8 | 1..4 | 16 |
| 8 | 780 | 5 | 3 | 6 | 1..4 | 13 |
| 8 | 990 | 4 | 1 | 4 | 1..4 | 11 |
| 8 | 1200 | 4 | 1 | 4 | 1..4 | 11 |
| 8 | 1410 | 4 | 1 | 4 | 1..4 | 11 |
| 8 | 1620 | 6 | 3 | 10 | 1..4 | 18 |
| 8 | 1830 | 5 | 2 | 7 | 1..4 | 15 |
| 8 | 2040 | 5 | 4 | 11 | 2..5 | 29 |

## Step 3 — Shape (IL,OL) coverage per (GPU,TP,Freq)

Total possible (IL,OL) pairs per cell = |IL| x |OL| = 11 x 4 = 44.

### GPU = L40S

| TP | Freq | existing pairs | missing | pairs |
|---|---|---|---|---|
| 1 | 210 | 11 | 33 | (32,32), (32,128), (128,32), (128,128), (128,1024), (512,32), (512,512), (1024,128), (2048,32), (4096,128), (8192,32) |
| 1 | 480 | 14 | 30 | (32,32), (32,128), (32,512), (32,1024), (128,32), (128,512), (128,1024), (512,32), (512,128), (512,512), (1024,128), (2048,32), (4096,128), (8192,32) |
| 1 | 735 | 13 | 31 | (32,32), (32,128), (32,512), (128,32), (128,1024), (512,32), (512,512), (512,1024), (1024,32), (1024,128), (2048,32), (4096,128), (8192,32) |
| 1 | 990 | 14 | 30 | (32,128), (32,512), (128,32), (128,128), (128,512), (128,1024), (512,32), (512,512), (512,1024), (1024,128), (1024,512), (2048,32), (4096,128), (8192,32) |
| 1 | 1245 | 14 | 30 | (32,128), (32,512), (32,1024), (128,128), (128,512), (128,1024), (512,32), (512,512), (1024,128), (1024,512), (2048,32), (2048,128), (4096,128), (8192,32) |
| 1 | 1500 | 9 | 35 | (32,32), (128,32), (128,1024), (512,512), (1024,32), (1024,128), (2048,32), (4096,128), (8192,32) |
| 1 | 1755 | 6 | 38 | (128,1024), (512,512), (1024,128), (2048,32), (4096,128), (8192,32) |
| 1 | 2010 | 6 | 38 | (128,1024), (512,512), (1024,128), (2048,32), (4096,128), (8192,32) |
| 1 | 2265 | 6 | 38 | (128,1024), (512,512), (1024,128), (2048,32), (4096,128), (8192,32) |
| 1 | 2520 | 44 | 0 | (32,32), (32,128), (32,512), (32,1024), (128,32), (128,128), (128,512), (128,1024), (512,32), (512,128), (512,512), (512,1024), (1024,32), (1024,128), (1024,512), (1024,1024), (2048,32), (2048,128), (2048,512), (2048,1024), (3072,32), (3072,128), (3072,512), (3072,1024), (4096,32), (4096,128), (4096,512), (4096,1024), (5120,32), (5120,128), (5120,512), (5120,1024), (6144,32), (6144,128), (6144,512), (6144,1024), (7168,32), (7168,128), (7168,512), (7168,1024), (8192,32), (8192,128), (8192,512), (8192,1024) |
| 2 | 210 | 5 | 39 | (128,1024), (512,512), (1024,32), (2048,32), (4096,128) |
| 2 | 480 | 6 | 38 | (128,1024), (512,512), (1024,32), (1024,1024), (2048,32), (4096,128) |
| 2 | 735 | 6 | 38 | (128,128), (128,1024), (512,512), (1024,32), (2048,32), (4096,128) |
| 2 | 990 | 8 | 36 | (128,1024), (512,128), (512,512), (512,1024), (1024,32), (1024,512), (2048,32), (4096,128) |
| 2 | 1245 | 8 | 36 | (32,32), (32,1024), (128,32), (128,1024), (512,512), (1024,32), (2048,32), (4096,128) |
| 2 | 1500 | 5 | 39 | (128,1024), (512,512), (1024,32), (2048,32), (4096,128) |
| 2 | 1755 | 5 | 39 | (128,1024), (512,512), (1024,32), (2048,32), (4096,128) |
| 2 | 2010 | 6 | 38 | (128,1024), (512,32), (512,512), (1024,32), (2048,32), (4096,128) |
| 2 | 2265 | 5 | 39 | (128,1024), (512,512), (1024,32), (2048,32), (4096,128) |
| 2 | 2520 | 9 | 35 | (128,1024), (512,512), (1024,32), (1024,128), (2048,32), (4096,32), (4096,128), (8192,32), (8192,128) |
| 4 | 210 | 6 | 38 | (1024,32), (1024,128), (2048,32), (3072,32), (4096,32), (8192,32) |
| 4 | 480 | 11 | 33 | (1024,32), (1024,128), (1024,512), (1024,1024), (2048,32), (2048,128), (2048,512), (2048,1024), (3072,32), (4096,32), (8192,32) |
| 4 | 735 | 11 | 33 | (128,512), (512,32), (512,128), (1024,32), (1024,128), (2048,32), (2048,512), (2048,1024), (3072,32), (4096,32), (8192,32) |
| 4 | 990 | 8 | 36 | (32,128), (1024,32), (1024,128), (2048,32), (2048,1024), (3072,32), (4096,32), (8192,32) |
| 4 | 1245 | 9 | 35 | (128,32), (128,128), (512,512), (1024,32), (1024,128), (2048,32), (3072,32), (4096,32), (8192,32) |
| 4 | 1500 | 6 | 38 | (1024,32), (1024,128), (2048,32), (3072,32), (4096,32), (8192,32) |
| 4 | 1755 | 6 | 38 | (1024,32), (1024,128), (2048,32), (3072,32), (4096,32), (8192,32) |
| 4 | 2010 | 6 | 38 | (1024,32), (1024,128), (2048,32), (3072,32), (4096,32), (8192,32) |
| 4 | 2265 | 7 | 37 | (128,1024), (1024,32), (1024,128), (2048,32), (3072,32), (4096,32), (8192,32) |
| 4 | 2520 | 10 | 34 | (128,1024), (512,512), (1024,32), (1024,128), (2048,32), (3072,32), (4096,32), (4096,128), (8192,32), (8192,128) |

### GPU = L4

| TP | Freq | existing pairs | missing | pairs |
|---|---|---|---|---|
| 1 | 210 | 8 | 36 | (32,32), (32,128), (32,512), (32,1024), (128,32), (128,128), (128,512), (512,32) |
| 1 | 360 | 7 | 37 | (32,32), (32,128), (32,512), (128,1024), (512,32), (512,128), (1024,32) |
| 1 | 570 | 1 | 43 | (512,512) |
| 1 | 1200 | 1 | 43 | (128,32) |
| 1 | 2040 | 44 | 0 | (32,32), (32,128), (32,512), (32,1024), (128,32), (128,128), (128,512), (128,1024), (512,32), (512,128), (512,512), (512,1024), (1024,32), (1024,128), (1024,512), (1024,1024), (2048,32), (2048,128), (2048,512), (2048,1024), (3072,32), (3072,128), (3072,512), (3072,1024), (4096,32), (4096,128), (4096,512), (4096,1024), (5120,32), (5120,128), (5120,512), (5120,1024), (6144,32), (6144,128), (6144,512), (6144,1024), (7168,32), (7168,128), (7168,512), (7168,1024), (8192,32), (8192,128), (8192,512), (8192,1024) |
| 2 | 210 | 3 | 41 | (1024,32), (2048,32), (4096,32) |
| 2 | 360 | 5 | 39 | (32,32), (32,128), (1024,32), (2048,32), (4096,32) |
| 2 | 570 | 3 | 41 | (1024,32), (2048,32), (4096,32) |
| 2 | 780 | 5 | 39 | (1024,32), (1024,128), (1024,512), (2048,32), (4096,32) |
| 2 | 990 | 3 | 41 | (1024,32), (2048,32), (4096,32) |
| 2 | 1200 | 3 | 41 | (1024,32), (2048,32), (4096,32) |
| 2 | 1410 | 3 | 41 | (1024,32), (2048,32), (4096,32) |
| 2 | 1620 | 5 | 39 | (128,128), (512,1024), (1024,32), (2048,32), (4096,32) |
| 2 | 1830 | 3 | 41 | (1024,32), (2048,32), (4096,32) |
| 2 | 2040 | 9 | 35 | (512,32), (512,512), (1024,32), (2048,32), (2048,512), (4096,32), (4096,128), (8192,32), (8192,128) |
| 4 | 210 | 9 | 35 | (32,32), (32,128), (512,32), (512,512), (1024,32), (2048,32), (2048,512), (4096,32), (8192,32) |
| 4 | 360 | 10 | 34 | (32,32), (128,128), (512,32), (512,128), (512,512), (1024,32), (2048,32), (2048,512), (4096,32), (8192,32) |
| 4 | 570 | 11 | 33 | (32,512), (128,32), (128,512), (512,32), (512,512), (512,1024), (1024,32), (2048,32), (2048,512), (4096,32), (8192,32) |
| 4 | 780 | 10 | 34 | (32,512), (32,1024), (512,32), (512,512), (512,1024), (1024,32), (2048,32), (2048,512), (4096,32), (8192,32) |
| 4 | 990 | 7 | 37 | (512,32), (512,512), (1024,32), (2048,32), (2048,512), (4096,32), (8192,32) |
| 4 | 1200 | 8 | 36 | (512,32), (512,512), (1024,32), (1024,512), (2048,32), (2048,512), (4096,32), (8192,32) |
| 4 | 1410 | 7 | 37 | (512,32), (512,512), (1024,32), (2048,32), (2048,512), (4096,32), (8192,32) |
| 4 | 1620 | 12 | 32 | (32,512), (128,512), (128,1024), (512,32), (512,512), (1024,32), (1024,128), (1024,1024), (2048,32), (2048,512), (4096,32), (8192,32) |
| 4 | 1830 | 7 | 37 | (512,32), (512,512), (1024,32), (2048,32), (2048,512), (4096,32), (8192,32) |
| 4 | 2040 | 11 | 33 | (512,32), (512,512), (1024,32), (2048,32), (2048,512), (4096,32), (4096,128), (8192,32), (8192,128), (8192,512), (8192,1024) |
| 8 | 210 | 5 | 39 | (128,32), (1024,32), (2048,32), (4096,32), (8192,32) |
| 8 | 360 | 7 | 37 | (32,32), (32,128), (128,128), (1024,32), (2048,32), (4096,32), (8192,32) |
| 8 | 570 | 8 | 36 | (32,32), (128,128), (512,128), (512,1024), (1024,32), (2048,32), (4096,32), (8192,32) |
| 8 | 780 | 6 | 38 | (32,512), (1024,32), (2048,32), (2048,128), (4096,32), (8192,32) |
| 8 | 990 | 4 | 40 | (1024,32), (2048,32), (4096,32), (8192,32) |
| 8 | 1200 | 4 | 40 | (1024,32), (2048,32), (4096,32), (8192,32) |
| 8 | 1410 | 4 | 40 | (1024,32), (2048,32), (4096,32), (8192,32) |
| 8 | 1620 | 10 | 34 | (128,32), (128,512), (512,32), (512,512), (1024,32), (1024,128), (2048,32), (2048,128), (4096,32), (8192,32) |
| 8 | 1830 | 7 | 37 | (512,32), (512,512), (1024,32), (2048,32), (2048,512), (4096,32), (8192,32) |
| 8 | 2040 | 11 | 33 | (512,32), (512,512), (1024,32), (2048,32), (2048,512), (4096,32), (4096,128), (8192,32), (8192,128), (8192,512), (8192,1024) |

## Step 4 — Rate-sweep quality & decode-capacity grade per (GPU,TP,Freq,IL,OL)

### GPU = L40S (over 280 measured configs)
- Confirmed (saw saturation knee): **52** (19%)
- Lower-bound only (stable seen, no knee): **124** (44%)
- No capacity (never stable at any tested rate): **104** (37%)
- Configs where BOTH stable & unstable rates observed: 73 | only one region: 207

### GPU = L4 (over 261 measured configs)
- Confirmed (saw saturation knee): **71** (27%)
- Lower-bound only (stable seen, no knee): **85** (33%)
- No capacity (never stable at any tested rate): **105** (40%)
- Configs where BOTH stable & unstable rates observed: 81 | only one region: 180

## Step 5 — Frequency coverage per (GPU,TP,IL,OL)

How many of the 10 clock levels were profiled for each shape.

### GPU = L40S
- #shapes (TP,IL,OL) measured: 83
- #freqs-per-shape histogram: {1: 49, 2: 5, 3: 5, 4: 2, 5: 2, 6: 3, 10: 17}
- shapes with >=8 freqs (DVFS-usable): 17 -> [(1, 128, 1024), (1, 512, 512), (1, 1024, 128), (1, 2048, 32), (1, 4096, 128), (1, 8192, 32), (2, 128, 1024), (2, 512, 512), (2, 1024, 32), (2, 2048, 32), (2, 4096, 128), (4, 1024, 32), (4, 1024, 128), (4, 2048, 32), (4, 3072, 32), (4, 4096, 32), (4, 8192, 32)]

### GPU = L4
- #shapes (TP,IL,OL) measured: 104
- #freqs-per-shape histogram: {1: 67, 2: 15, 3: 8, 10: 14}
- shapes with >=8 freqs (DVFS-usable): 14 -> [(2, 1024, 32), (2, 2048, 32), (2, 4096, 32), (4, 512, 32), (4, 512, 512), (4, 1024, 32), (4, 2048, 32), (4, 2048, 512), (4, 4096, 32), (4, 8192, 32), (8, 1024, 32), (8, 2048, 32), (8, 4096, 32), (8, 8192, 32)]

## Step 6 — TP coverage per (GPU,IL,OL,Freq)

### GPU = L40S (|TP|=3)
- (IL,OL,Freq) cells measured: 177
- TP-set histogram: [1]:65; [1, 2]:28; [4]:27; [1, 2, 4]:22; [1, 4]:22; [2, 4]:9; [2]:4

### GPU = L4 (|TP|=4)
- (IL,OL,Freq) cells measured: 144
- TP-set histogram: [1]:40; [4]:28; [2, 4, 8]:26; [4, 8]:19; [1, 2, 4, 8]:11; [1, 4]:6; [8]:6; [2]:4; [1, 4, 8]:2; [1, 2, 8]:1; [1, 8]:1

## Step 7 — TPOT-SLO bucket coverage (from decode_capacity.csv)

### GPU = L40S
| SLO(ms) | confirmed | lower_bound | missing |
|---|---|---|---|
| 50 | 33 | 89 | 158 |
| 100 | 42 | 112 | 126 |
| 200 | 52 | 123 | 105 |
| 500 | 52 | 124 | 104 |

### GPU = L4
| SLO(ms) | confirmed | lower_bound | missing |
|---|---|---|---|
| 50 | 35 | 51 | 175 |
| 100 | 47 | 60 | 154 |
| 200 | 67 | 71 | 123 |
| 500 | 70 | 83 | 108 |

## Step 8 — Final coverage report

### GPU = L40S
- Full config grid |TP|x|Freq|x|IL|x|OL| = 3x10x11x4 = **1320**
- Measured configs (any rate): **280** = 21% of grid
- (1) measured capacity (confirmed): **4%** of full grid (52 cells)
- (2) conservative lower-bound only: **9%** of full grid (124 cells)
- usable (confirmed+lower_bound) of MEASURED configs: 63%; of FULL grid: 13%
- (3) DENSE: IL axis (all 11 values appear), OL axis (all 4 appear), TP axis (all 3 appear), rate grid (5 pts) at the shapes that were swept.
- (4) SPARSE: FREQUENCY per shape — 49/83 (TP,IL,OL) shapes have a SINGLE clock; only 17 have >=8.

### GPU = L4
- Full config grid |TP|x|Freq|x|IL|x|OL| = 4x10x11x4 = **1760**
- Measured configs (any rate): **261** = 15% of grid
- (1) measured capacity (confirmed): **4%** of full grid (71 cells)
- (2) conservative lower-bound only: **5%** of full grid (85 cells)
- usable (confirmed+lower_bound) of MEASURED configs: 60%; of FULL grid: 9%
- (3) DENSE: IL axis (all 11 values appear), OL axis (all 4 appear), TP axis (all 4 appear), rate grid (5 pts) at the shapes that were swept.
- (4) SPARSE: FREQUENCY per shape — 67/104 (TP,IL,OL) shapes have a SINGLE clock; only 14 have >=8.

### Top coverage holes ranked by scheduler-coverage impact

Impact = DVFS cells (shape x 10 freq) the hole denies the decode gate, weighted toward diverse OL (the gate needs decode capacity across OL, but full freq sweeps sit almost entirely at OL=32).

1. **[L4]** OL=128: 24/29 shapes single-clock -> ~216 DVFS cells missing
2. **[L4]** OL=512: 16/25 shapes single-clock -> ~144 DVFS cells missing
3. **[L4]** OL=1024: 16/19 shapes single-clock -> ~144 DVFS cells missing
4. **[L40S]** OL=128: 15/23 shapes single-clock -> ~135 DVFS cells missing
5. **[L40S]** OL=32: 12/25 shapes single-clock -> ~108 DVFS cells missing
6. **[L40S]** OL=1024: 12/18 shapes single-clock -> ~108 DVFS cells missing
7. **[L4]** OL=32: 11/31 shapes single-clock -> ~99 DVFS cells missing
8. **[L40S]** OL=512: 10/17 shapes single-clock -> ~90 DVFS cells missing

(Only 8 GPU x OL groups exist, so the list stops at 8 — this is the natural
granularity of the frequency hole. The single dominating structural hole is:
**full-frequency sweeps live almost entirely at OL=32** — L4 has 12 of 14 full-sweep
shapes at OL=32, L40S 9 of 17 — so every non-OL=32 decode shape is DVFS-starved.)

## Cross-cutting completeness (rate & TP), for the marginal-value estimate

| GPU | full-5 rate sweep | knee bracketed (stable+unstable seen) | (IL,OL,Freq) cells with ALL TPs |
|---|---|---|---|
| L40S | 3/280 (1%) | 73/280 (26%) | 22/177 (12%) |
| L4 | 4/261 (2%) | 81/261 (31%) | 11/144 (8%) |

- **Rate sweep is the confidence bottleneck:** only 26-31% of measured configs bracket
  the saturation knee, which is exactly why most configs are "lower-bound only" or "none"
  rather than "confirmed". Rate depth per config is mostly 1-3 of the 5 available points.
- **TP is sparse:** only 8-12% of (IL,OL,Freq) cells carry all TP degrees.
- **IL is dense** (all 11 values present wherever a shape was swept); **OL** has all 4
  values but its FREQUENCY dimension is starved off OL=32.

## Marginal value per dimension (coverage gain per unit effort) — NO experiments proposed

Two orthogonal levers dominate; they fix different failures:

1. **Rate-sweep completion — highest marginal value for capacity CONFIDENCE.**
   Extending the rate sweep upward on ALREADY-measured (TP,Freq,IL,OL) configs converts
   `lower_bound`/`none` -> `confirmed` with NO new shapes/freqs/TPs. It reuses existing
   config points, so effort is low, and it directly raises the 26-31% knee-bracket rate.
   It does NOT enlarge the DVFS grid.
2. **Frequency completion — highest marginal value for DVFS GRID coverage.**
   ~49 (L40S) / ~67 (L4) single-clock shapes cannot support DVFS at all; each additional
   clock adds one usable cell, and the whole energy-optimization premise is DVFS. Highest
   value when TARGETED at non-OL=32 shapes (holes #1-8 above), since full sweeps today sit
   at OL=32.
3. **OL x Frequency — high, but it is really lever #2 aimed at OL in {128,512,1024}.**
   Decode capacity depends strongly on OL, and the freq sweep is missing precisely there.
4. **TP completion — medium.** Only 8-12% complete, and it matters for routing/TP choice,
   but decode capacity is not TP-monotone (no safe cross-TP borrow), so each TP is its own
   independent measurement with no leverage onto others.
5. **IL completion — negligible.** IL is already dense; near-zero marginal gain.

**To move scheduler coverage from ~9-13% of the full grid toward ~90%, the binding
dimensions are FREQUENCY (grid size) and RATE-SWEEP DEPTH (confidence), in that priority;
OL is the axis along which frequency completion pays off most; TP is secondary; IL adds
essentially nothing.**
