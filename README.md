# Two Strix Halo boxes over USB4 beat one: llama.cpp tensor parallel, 128 µs interrupt throttling, and two lines

Measurements, patches and a tiny kernel module for running **one 27B model across two MINISFORUM MS-S1 MAX (Ryzen AI Max+ 395, 128 GB)** over a plain USB4 cable (`thunderbolt_net`, TCP), with llama.cpp RPC `-sm tensor` (tensor parallel).

Short version: two boxes were **26% slower** than one. The cause was not the link width. The Linux `thunderbolt` driver sets the USB4 host router's interrupt throttling to **128 µs**, and a generation step does about 128 small all-reduces. Setting it to 0, plus a few small llama.cpp changes, gave **+23%** over one box, and **about +7%** over one box with the best speculative-decoding depth when the partial sums are split over two USB4 lines.

This is a measurement log by one person with AI coding assistants (Claude Code, Gemini, ChatGPT). Experiment design, judging results and operating the machines were done by a human. Each number below is a single run (160 generated tokens, temperature 0) unless noted, and differences of a few percent are within noise.

## Results

Generation speed, tok/s (Japanese prompt / English prompt). Model: Qwen3.8-27B UD-Q4_K_XL (16.34 GiB). One box: 12.2-12.3 / 12.1-12.3 depending on the run (each run's own value is in `data/README.md`); the percentages below use 12.3 (Japanese prompt).

| Step | What changed | 2 boxes, tensor parallel | vs one box |
|---|---|---|---|
| 1 | upstream llama.cpp b12038 + the 2-line sync fix from the PR #26610 thread (it crashes without it) | 9.06 / 9.09 | -26% |
| 2 | our patch: do not stall the GPU while waiting for the peer | 9.75 / 9.76 | -21% |
| 3 | USB4 host router interrupt throttling 128 µs -> 0 (`module/`) | 14.05 / 14.09 | +14% |
| 4 | our patch: read small tensors from uncached UMA memory directly instead of making a copy | 15.10 / 15.04 | **+23%** |
| 5 | our patch: send half of each partial sum on each of two USB4 lines | 15.46 | +26% |

Round trip over `thunderbolt_net` (TCP, p50): 4 KB 130 -> 43 µs, 8 KB 131 -> 53 µs, 20 KB (one decode step's partial sum) **224 -> 80 µs**. The same 20 KB takes about 77 µs on 10 GbE.

### With speculative decoding (MTP)

Speculative decoding with MTP heads is not new: it was merged into llama.cpp in May 2026 ([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)), and roughly 2x single-box speedups on Strix Halo are already reported publicly. We measured it too (one box: 12.3 -> 22.6 tok/s at depth 2). What we add is the **two-box** numbers below. We did not find public two-box MTP measurements, but absence of search hits is not proof that none exist.

Depth sweep, each side at its own best depth:

| MTP depth | 1 box | 2 boxes, 1 line | 2 boxes, 2 lines |
|---|---|---|---|
| 0 (none) | 12.3 | 15.0 | 15.5 |
| 2 | 22.6 | 23.4 | 25.6 |
| 3 | 24.3 | 24.3 | **25.9** |
| 4 | 23.3 | 21.6 | 24.2 |

- With one line, the best two-box depth only **ties** the best one-box depth. Speculative decoding sends 40-80 KB per step, and one line is not enough.
- Splitting each partial sum over two lines gives +3% without MTP and +6% to +9% with it, measured against one line in the same run (depth 2: 23.50 -> 25.60, depth 3: 24.45 -> 25.94, depth 4: 22.83 -> 24.19). The "1 line" column in the table above comes from a separate depth-sweep run. With two lines, two boxes are about **+7% over one box at its best depth** (25.9 vs 24.3). The output text is identical to the one-line run.
- Each extra verified token costs a two-box setup about 1.4-1.6x more per step than a one-box setup (the all-reduce payload grows). Two boxes do not get cheaper "by dividing the work".
- bf16 partial sums do not add anything once two lines are used (+0.2%).

### What did not work

- Layer split (`-sm layer`): 11.2 vs 12.3 on one box. It is a relay, so only one box works at a time.
- Prompt processing (prefill) does not get faster with tensor parallel over this link (about 340 tok/s at pp2048, vs 354 on one box).
- Raising the MTU, the number of connections, `dma_credits`, `clx=0` or NHI IOMMU identity mapping did not move the ~9.4 Gbit/s per-NHI TCP ceiling. Two NHIs at once reach 10.1 + 10.1 = 20.2 Gbit/s.

## Caveats

- 1 box vs 2 boxes with temperature 0: the text diverges after some tokens (tensor parallel adds in a different order). We have not measured whether numbers or names change. The two-line split does not change the output compared with one line.
- Single runs, one machine pair, one model, one cable. Please send numbers if yours differ.
- Front USB4 ports only (2 lanes x 10 Gb/s = 20 Gb/s). We did not use the 80 Gbps port.

## Setup

- 2x MINISFORUM MS-S1 MAX (BIOS 1.09), Ubuntu 26.04, kernel 7.0.0-42, Secure Boot on
- llama.cpp b12038 (`a7a490a49`), Vulkan (RADV) + RPC
- Two front USB4 ports on each box connected to the other box (`thunderbolt0`, `thunderbolt1`), 10.77.x.x addresses on the cable

## 1. The kernel module (`module/`)

Sets `REG_INT_THROTTLING_RATE` (offset `0x38c00`, 16 vectors, 256 ns units) to `val` on the AMD USB4 host routers `1022:158d` / `1022:158e`, and puts the old values back when unloaded. Linux sets 500 (128 µs) in `nhi_enable_int_throttling()`.

```bash
cd module && make                     # needs kernel headers
sudo insmod tb_itr.ko val=0           # 0 = no throttling
sudo rmmod tb_itr                     # restores the old value
```

- **Use at your own risk.** It writes hardware registers. We only tested it on these two boxes. A reboot also restores the default.
- With Secure Boot on, sign the module with your own MOK key (`kmodsign`). Writing the register from `/sys/.../resource0` is blocked by kernel lockdown.
- Do this on both boxes. Then check the 20 KB round trip: it should drop from ~220 µs to ~80 µs.

## 2. The llama.cpp patch (`patches/llama-b12038.diff`)

One diff against `a7a490a49` (b12038). It contains the sync fix, the "do not stall" change, the direct UMA read, the second line, and some profiling switches. Everything is off by default and switched on by environment variables.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout a7a490a4905c36349f1074d3f5322291304a6dd3
git apply /path/to/patches/llama-b12038.diff
cmake -B build -G Ninja -DGGML_VULKAN=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-bench ggml-rpc-server
```

| Variable | Effect |
|---|---|
| `GGML_RPC_LAZY_SYNC=1`, `GGML_VK_HOST_BARRIER=1` | do not stall the GPU while waiting for the peer |
| `GGML_VK_UMA_DIRECT_READ_MAX=65536` | read tensors up to 64 KB from uncached UMA memory directly |
| `GGML_RPC_PEER2_HOST=<ip>` | second line: rank 0 listens on port+1, rank 1 connects to this address |
| `GGML_RPC_SPLIT_MIN` / `GGML_RPC_SPLIT_MAX` | payload range (bytes) that is split over two lines (default 8192-65536) |
| `GGML_RPC_BF16_MIN` | send partial sums as bf16 from this size |
| `GGML_RPC_PROF2=1` | print the average split of one all-reduce every 2000 calls |

## 3. Running it

On each box, with the module loaded (`val=0`) and the cables connected:

```bash
export GGML_RPC_LAZY_SYNC=1 GGML_VK_HOST_BARRIER=1 GGML_VK_UMA_DIRECT_READ_MAX=65536
# add on both rpc-servers for the second line (use box 1's address on line 2):
# export GGML_RPC_PEER2_HOST=10.77.3.1

# box 1 and box 2 (each on its own address on line 1):
build/bin/ggml-rpc-server -H 10.77.1.1 -p 50052 -c     # box 1
build/bin/ggml-rpc-server -H 10.77.1.2 -p 50052 -c     # box 2

# box 1:
build/bin/llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl 99 -fa on \
  --rpc 10.77.1.1:50052,10.77.1.2:50052 -dev RPC0,RPC1 -sm tensor \
  --spec-type draft-mtp --spec-draft-n-max 3
```

If the Vulkan allocations fail, we also set `GGML_VK_SUBALLOCATION_BLOCK_SIZE`, `GGML_VK_FORCE_MAX_BUFFER_SIZE` and `GGML_VK_FORCE_MAX_ALLOCATION_SIZE` to 12884901888.

## Raw data (`data/`)

Run logs and JSON answers for the steps above. See `data/README.md` for which directory is which.

## Related work

Before publishing we searched again (English pages, 2026-11-20 and 2026-11-21). We did not find a published one-box vs two-box decode comparison on this hardware with stock `thunderbolt_net` TCP, but absence of search hits is not proof that none exist. Related public work:

- The 128 µs value was added to the Linux `thunderbolt` driver in 2017 ([patch](https://lkml.iu.edu/hypermail/linux/kernel/1709.2/01071.html)). The USB4STREAM series (Apr-May 2026, [v1](https://ratatoskr.run/netdev/2026/04/8893681/t), [v2](https://ratatoskr.run/linux-usb/2026/05/8994875/t)) lets each service driver set its own throttling from Linux 7.2 (commit `c51777370ac2`); the network driver keeps 128 µs (`TBNET_THROTTLING`). Our boxes run 7.0, so we used a module instead.
- A GitHub issue ([fak #11910](https://github.com/anthony-chaudhary/fak/issues/11910), Sep 2026) commits code that reprograms the same register (`0x38c00`) from 128 µs to 8 µs for dual Strix Halo with RoCEv2; we found no measured one-box vs two-box results there.
- [thunderbolt-ibverbs PR #75](https://github.com/hellas-ai/thunderbolt-ibverbs/pull/75) (Sep 2026, open) sets the throttle on Apple NHI hardware (65 -> 15 µs for 2 B messages); the stock-NHI case is left out of that PR.
- [Strix-halo-dual-optimized](https://github.com/ayysasha/Strix-halo-dual-optimized) cuts TCP latency over `thunderbolt_net` from ~65 µs to ~8 µs with busy polling and power settings (vLLM, TP=2).
- [strix-halo-usb4-driver](https://github.com/hyukudan/strix-halo-usb4-driver) and [strix-halo-engine-lab](https://github.com/hyukudan/strix-halo-engine-lab) use their own USB4 stream driver (throttling set to 8192 ns, optional polling) and their own engine for TP2 across two Strix Halo boxes.

## The story behind it

The original write-up is in Japanese (note, episode 9 of a series). note offers an automatic English translation, which we have not checked, so please read it with care: https://note.com/zenkai_throttle/n/n13515d4bc8dd

## Questions we would like answered

- Does the same interrupt-throttling change help other hardware that uses the Linux `thunderbolt` driver for host-to-host networking?
- Do you see a different ceiling than ~9.4 Gbit/s per host router with TCP over `thunderbolt_net`?
- Can the 80 Gbps port work as a third line next to the two front lines?

## License

- `patches/`: the diff modifies llama.cpp, so it is under the MIT license (see `LICENSE-MIT`).
- `module/tb_itr.c`: GPL-2.0 (SPDX header in the file).
- Text and data: free to reuse with attribution.
