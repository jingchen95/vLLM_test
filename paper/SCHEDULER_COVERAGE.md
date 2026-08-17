# Scheduler-coverage analysis (verified)

## Data-integrity checks

- [PASS] total lookup counts reconcile
- [PASS] exact + lower_bound + missing = total
- [PASS] coverage percentages sum to 100%
- [PASS] greedy selected cells are unique
- [PASS] cumulative coverage is monotonic
- [PASS] no cumulative percentage exceeds 100%
- [PASS] top-k shape shares use one consistent key (GPU,IL,OL,TP)
- [PASS] blank/unreachable SLO records explicitly handled

Totals: decode lookups **N=95139**, prefill lookups 414, windows 36. Source partition: measured=2936, lower_bound=3590, missing=88613.
On-grid lookups (IL,OL exactly on the profiling grid): 76615 (80.5%); off-grid means are snapped to the nearest cell for the greedy (planning) model.

## Definitions

- **hardware cell** = (GPU, IL, OL, TP, Freq); one rate sweep -> capacity at all SLO buckets, so SLO rows do NOT create extra hardware cells.
- **query shape** = (GPU, IL, OL, TP); the single top-k grouping key.
- coverage read from the real logged resolution; percentages are of N.

## Task 1 — Query distribution

- **gpu**: l4:84%, l40s:16%
- **tp**: 1:89%, 2:7%, 4:4%
- **il**: 1024:70%, 2048:15%, 512:8%, 4096:4%, 128:2%, 3072:1%
- **ol**: 512:69%, 32:15%, 128:12%, 1024:3%
- **freq**: 210:11%, 990:10%, 1410:9%, 360:9%, 1620:9%, 1830:9%, 2040:9%, 570:8%, 780:8%, 1200:8%

Top query shapes **(GPU,IL,OL,TP)** — the consistent key:
  - l4 IL=1024 OL=512 TP=1: 47089 (49.5%)
  - l4 IL=2048 OL=32 TP=1: 10278 (10.8%)
  - l40s IL=1024 OL=512 TP=1: 9630 (10.1%)
  - l4 IL=512 OL=128 TP=1: 3958 (4.2%)
  - l4 IL=1024 OL=128 TP=1: 3599 (3.8%)
  - l4 IL=1024 OL=512 TP=2: 2460 (2.6%)
  - l4 IL=1024 OL=512 TP=4: 2380 (2.5%)
  - l40s IL=2048 OL=32 TP=1: 2156 (2.3%)
  - l4 IL=4096 OL=1024 TP=1: 2000 (2.1%)
  - l4 IL=128 OL=512 TP=1: 1389 (1.5%)

Shape shares (GPU,IL,OL,TP): top-1=49.5%, top-3=70.4%, top-5=78.4%, top-10=89.3%  (47 distinct shapes)
For reference only, by (IL,OL) alone: top-1=64.7%, top-3=86.0%, top-5=94.1%, top-10=100.0% (10 distinct). The report uses (GPU,IL,OL,TP) everywhere below.

## Task 2 — Lookup-weighted coverage

- exact (measured): **2936 = 3.1%**
- lower_bound: **3590 = 3.8%**
- missing: **88613 = 93.1%**
- **covered (exact+lower_bound) = 6526 = 6.9%**
- of the missing: 57843 need a NEW shape, 30770 are frequency-completable on an existing shape, 0 unreachable (req_slo<50ms), 0 had no SLO bucket at replay (shape absent; still coverable by profiling).

## Task 3 — (TP x Freq) query weight per (GPU,IL,OL) shape (top 6)

### l4 IL=1024 OL=512 — 51929 queries (55%), 100% to missing cells
| TP\Freq | 210 | 360 | 570 | 780 | 990 | 1200 | 1410 | 1620 | 1830 | 2040 |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 4739x | 4735x | 4735x | 4600x | 4600x | 4600x | 4770x | 4770x | 4770x | 4770x |
| **2** | 230x | 230x | 230x | 230x | 230x | 230x | 270x | 270x | 270x | 270x |
| **4** | 230x | 230x | 230x | 230x | 230x | 230~ | 250x | 250x | 250x | 250x |
(x=missing, ~=lower-bound, blank=measured)

### l4 IL=2048 OL=32 — 11818 queries (12%), 85% to missing cells
| TP\Freq | 210 | 360 | 570 | 780 | 990 | 1200 | 1410 | 1620 | 1830 | 2040 |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 990x | 974x | 954x | 952x | 950x | 950x | 1148x | 1120x | 1120x | 1120 |
| **2** | 80x | 80x | 80x | 80x | 80x | 80x | 120x | 120x | 120x | 120 |
| **4** | 50x | 50x | 50 | 50 | 50 | 50 | 70 | 70 | 70 | 70 |
(x=missing, ~=lower-bound, blank=measured)

### l40s IL=1024 OL=512 — 9630 queries (10%), 71% to missing cells
| TP\Freq | 210 | 480 | 735 | 990 | 1245 | 1500 | 1755 | 2010 | 2265 | 2520 |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 1026x | 1026x | 1026x | 936~ | 936~ | 936x | 936x | 936x | 936x | 936 |
(x=missing, ~=lower-bound, blank=measured)

### l4 IL=512 OL=128 — 5178 queries (5%), 84% to missing cells
| TP\Freq | 210 | 360 | 570 | 780 | 990 | 1200 | 1410 | 1620 | 1830 | 2040 |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 482x | 422~ | 412x | 400x | 388x | 388x | 392x | 358x | 358x | 358 |
| **2** | 104x | 104x | 102x | 102x | 100x | 94x | 94x | 94x | 94x | 94x |
| **4** | 26x | 26~ | 26x | 24x | 24x | 24x | 22x | 22x | 22x | 22x |
(x=missing, ~=lower-bound, blank=measured)

### l4 IL=1024 OL=128 — 4391 queries (5%), 92% to missing cells
| TP\Freq | 210 | 360 | 570 | 780 | 990 | 1200 | 1410 | 1620 | 1830 | 2040 |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 843x | 567x | 335x | 288x | 264x | 262x | 266x | 258x | 258x | 258 |
| **2** | 64x | 64x | 64x | 64~ | 64x | 64x | 64x | 64x | 64x | 64x |
| **4** | 16x | 16x | 16x | 16x | 16x | 16x | 14x | 14~ | 14x | 14x |
(x=missing, ~=lower-bound, blank=measured)

### l40s IL=2048 OL=32 — 2256 queries (2%), 13% to missing cells
| TP\Freq | 210 | 480 | 735 | 990 | 1245 | 1500 | 1755 | 2010 | 2265 | 2520 |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 216x | 216~ | 216~ | 216~ | 216~ | 216~ | 216~ | 216~ | 214~ | 214 |
| **2** | 10x | 10x | 10x | 10x | 10x | 10x | 10 | 10x | 10x | 10 |
(x=missing, ~=lower-bound, blank=measured)

## Task 4 — Marginal value: greedy hardware-cell selection

Greedy over currently-MISSING coverable lookups (88613), each assigned to its nearest hardware cell; a cell is credited only for its own lookups (no double counting). Ceiling = 100.0% (the rest are unreachable at the current SLO grid).

| +cells | incr. this-batch | cum. NEW covered | cum. TOTAL coverage |
|---|---|---|---|
| 10 | 47089 (49.5%) | 47089 (49.5%) | 56.4% |
| 20 | 10336 (10.9%) | 57425 (60.4%) | 67.2% |
| 50 | 14281 (15.0%) | 71706 (75.4%) | 82.2% |
| 100 | 8928 (9.4%) | 80634 (84.8%) | 91.6% |

Cells to reach total coverage: 80% -> **42**, 90% -> **86**, 95% -> **140** (None = ceiling below target).

Unique hardware cells in the greedy list: 428. Each, once profiled, populates 4 SLO-table rows (x4), i.e. 1712 table rows from 428 hardware sweeps — SLO rows are NOT independent cells.

Highest-value cells first (GPU, IL, OL, TP, Freq | missing lookups | cum TOTAL):
  1. l4 IL=1024 OL=512 TP=1 f=1410 | 4770 (5.0%) | 11.9%
  2. l4 IL=1024 OL=512 TP=1 f=1620 | 4770 (5.0%) | 16.9%
  3. l4 IL=1024 OL=512 TP=1 f=1830 | 4770 (5.0%) | 21.9%
  4. l4 IL=1024 OL=512 TP=1 f=2040 | 4770 (5.0%) | 26.9%
  5. l4 IL=1024 OL=512 TP=1 f=210 | 4739 (5.0%) | 31.9%
  6. l4 IL=1024 OL=512 TP=1 f=360 | 4735 (5.0%) | 36.9%
  7. l4 IL=1024 OL=512 TP=1 f=570 | 4735 (5.0%) | 41.8%
  8. l4 IL=1024 OL=512 TP=1 f=780 | 4600 (4.8%) | 46.7%
  9. l4 IL=1024 OL=512 TP=1 f=990 | 4600 (4.8%) | 51.5%
  10. l4 IL=1024 OL=512 TP=1 f=1200 | 4600 (4.8%) | 56.4%
  11. l4 IL=2048 OL=32 TP=1 f=1410 | 1148 (1.2%) | 57.6%
  12. l4 IL=2048 OL=32 TP=1 f=1620 | 1120 (1.2%) | 58.7%
  13. l4 IL=2048 OL=32 TP=1 f=1830 | 1120 (1.2%) | 59.9%
  14. l40s IL=1024 OL=512 TP=1 f=210 | 1026 (1.1%) | 61.0%
  15. l40s IL=1024 OL=512 TP=1 f=480 | 1026 (1.1%) | 62.1%

## Task 5 — Frequency-completion-only gain

- Missing lookups fixable by FREQUENCY completion of an already-measured shape: **30770 (32.3%)**
- Missing lookups needing a NEW (GPU,IL,OL,TP) shape: 57843 (60.8%)
- Frequency completion alone lifts coverage to **39.2%** (vs 82.2% for 50 targeted cells).

## Task 6 — Rate-completion-only gain (lower_bound -> confirmed)

- Lower-bound lookups upgradeable to confirmed by extra RATE sweeps only (no new IL/OL/TP/Freq): **3590 (3.8%)** across 13 cells.

## Task 6B — Window-weighted coverage (separate from lookup-weighted)

- windows: **36**
- fully covered (every decode lookup covered): 0 (0%)
- >=1 usable decode candidate (>=1 covered decode lookup): 31 (86%)
- NO usable decode candidate (all decode lookups missing): 5 (14%)
- >=1 usable decode AND >=1 prefill hit (approx 'complete feasible candidate'; same-candidate pairing would need candidate-id logging): 28 (78%)

## Task 7 — Final answers

1. Covered (lookup-weighted): **6.9%**.
2. Missing: **93.1%**.
3. Spend on the greedy hardware cells above (hot shapes x full freq at TP=1); blanket freq-completion of measured shapes only reaches 39%.
4. Budget: +10 -> 56%, +20 -> 67%, +50 -> 82%, +100 -> 92%.
5. Cells for 80/90/95% = 42/86/140.
