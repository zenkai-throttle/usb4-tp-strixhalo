# Raw data

Each directory holds `run.log` (the script's output), `r_*.json` (llama-server `/completion` answers with timings), and bench output. Host names are anonymized (`unit1`, `unit2`). Speeds are `predicted_per_second` for 160 tokens, temperature 0.

| Directory | What it is | Headline numbers (ja / en tok/s) |
|---|---|---|
| `20261103-2226-front20g-sync` | upstream b12038 + 2-line sync fix, default 128 µs throttling | one box 12.19 / 12.15, tensor 9.06 / 9.09 (step 1) |
| `20261104-1142-usb4front-itr0-lazy` | throttling 0 + "do not stall" patch | tensor 14.05 / 14.09 (step 3) |
| `20261104-1512-usb4front-itr0-lazy-prof2-vkdirect` | + direct UMA read | tensor 15.10 / 15.04 (step 4), MTP depth 2: 23.48 / 23.42 |
| `20261106-1128-w237a-depth-usb4front` | MTP depth 0-4, one box and two boxes (one line) | see `run.log` |
| `20261106-1324-two-line-split` | one line vs two lines, no MTP and MTP depth 2 | 15.0 -> 15.5 |
| `20261106-1330-two-line-mtp` | one line vs two lines, MTP depth 2-4 | depth 3: 24.4 -> 25.9 (Japanese/English average) |

The Japanese and English prompts reach different acceptance rates with MTP (for example depth 3 on one box: 22.4 vs 26.1), so each side is averaged in the README.
