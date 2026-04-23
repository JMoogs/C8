# C8

A cycle-accurate CHIP-8 interpreter that runs entirely in the terminal, written in Rust.

## Overview

C8 implements the full 35-instruction CHIP-8 ISA with accurate 60 Hz timer behaviour and flicker-free terminal output. Four OS threads run concurrently: the main loop executes instructions at a configurable rate, a dedicated timer thread decrements the delay and sound timers at exactly 60 Hz, a render thread repaints the terminal at 60 Hz independently of CPU speed, and an input thread blocks on keyboard events — all coordinated through atomics and a single mutex with no contention on the hot path.

A compile-time feature flag selects between the original CHIP-8 instruction set and the CHIP-48/SCHIP variant, producing two distinct binaries from the same source.

## Features

- **Cycle-accurate 60 Hz timers** — delay and sound timers tick in a dedicated background thread via `Arc<AtomicU8>`, decoupled from instruction rate so timer-sensitive games behave correctly at any speed
- **Decoupled 60 Hz render thread** — the display runs at exactly 60 Hz regardless of CPU rate; any number of pixel writes within a 16.67 ms frame are coalesced into a single terminal redraw, preventing I/O from bottlenecking the interpreter at high IPS
- **Dirty-buffer rendering** — pixel writes during sprite ops mark the frame dirty; the render thread skips the pass entirely when nothing has changed
- **Live IPS measurement** — measures actual instructions executed per second against the configured target, surfaced at runtime to expose OS scheduling jitter
- **Runtime speed control** — default 700 IPS, adjustable live with arrow keys (±100 / ±1000 IPS) via a shared `Arc<AtomicU64>` with no locking on the interpreter path
- **Dual ISA via compile-time feature flag** — `--features new_instructions` builds the CHIP-48/SCHIP variant: in-place shifts, register-relative `BNNN` jumps, non-incrementing index for store/load
- **Sprite collision detection** — XOR pixel drawing with `VF` flag set on collision, per spec
- **Full instruction coverage** — BCD decode, font character lookup, growable subroutine stack, `FxNN` I/O ops
- **Sound** — terminal bell (`BEL`) fires while the sound timer is active; suppressible with `--disable-sound`
- **26-test suite** — unit tests cover decode correctness and arithmetic VF flag behaviour, including regression tests for the carry and borrow edge cases

## Tech Stack

| Crate | Role |
|---|---|
| `crossterm` | Cross-platform terminal raw mode and keyboard event polling |
| `clap` | CLI argument parsing via derive macros |
| `rand` | Pseudorandom number generation for the `CXNN` instruction |
| `std::sync` | `Arc<AtomicU8>`, `Arc<AtomicU64>`, `Arc<Mutex<…>>` for cross-thread state |

## Getting Started

**Prerequisites:** Rust toolchain (`cargo`). Install via [rustup.rs](https://rustup.rs).

```bash
# Clone
git clone https://github.com/JMoogs/C8
cd C8

# Build — original CHIP-8 instruction set
cargo build --release

# Build — CHIP-48/SCHIP instruction set
cargo build --release --features new_instructions

# Run a ROM
./target/release/c8 path/to/rom.ch8

# Run with runtime speed controls enabled
./target/release/c8 --enable-speed-controls path/to/rom.ch8

# All options
./target/release/c8 --help

# Run tests
cargo test
```

**Requires a terminal with at least 64 columns × 35 rows** (64×32 for the display + 2 rows for the IPS stats line).

### Controls

The CHIP-8 hex keypad maps to the left block of a QWERTY keyboard:

```
CHIP-8   →   Keyboard
1 2 3 C      1 2 3 4
4 5 6 D      q w e r
7 8 9 E      a s d f
A 0 B F      z x c v
```

| Key | Action |
|---|---|
| `Esc` | Exit |
| `←` / `→` | Speed ±100 IPS (requires `--enable-speed-controls`) |
| `↑` / `↓` | Speed ±1000 IPS (requires `--enable-speed-controls`) |

ROMs for testing are widely available — the [Chip-8 Test Suite](https://github.com/Timendus/chip8-test-suite) is a good starting point for verifying instruction correctness.
