A CHIP-8 interepreter that runs in a terminal.

## Usage
Run `c8 --help` for a list of options.

## Building

Install cargo

For the original instruction set use:
```bash
cargo build --release
```

For the new instruction set used by some newer games use:
```bash
cargo build --release --features new_instructions
```

