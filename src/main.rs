mod display;
mod opcode;
mod vm;

use std::{
    io::Write,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use clap::Parser;
use crossterm::ExecutableCommand;

use vm::VM;

/// A CHIP-8 interpreter
///
/// Implements the full CHIP-8 ISA with cycle-accurate 60 Hz timers and a
/// decoupled render thread. Build with `--features new_instructions` for the
/// CHIP-48/SCHIP variant (in-place shifts, register-relative BNNN).
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Speed of the interpreter in instructions per second
    #[arg(short, long, default_value_t = 700)]
    speed: u32,

    /// Enable runtime speed controls
    ///
    /// Left/right arrow keys adjust speed by ±100 IPS.
    /// Up/down arrow keys adjust speed by ±1000 IPS.
    #[arg(short, long)]
    enable_speed_controls: bool,

    /// Disable sound
    #[arg(short, long)]
    disable_sound: bool,

    /// Path to the CHIP-8 ROM file
    program: String,
}

// Four-thread design:
//   Main thread   — fetch / decode / execute at a configurable IPS rate.
//   Timer thread  — decrements delay and sound timers at exactly 60 Hz;
//                   uses AtomicU8 so reads on the interpreter path are lock-free.
//   Render thread — repaints the terminal at 60 Hz, fully decoupled from CPU rate;
//                   coalesces all pixel writes within one frame into one redraw.
//   Input thread  — blocks on crossterm events; writes keypad state through a
//                   Mutex<[bool; 16]> and speed through a shared AtomicU64.
fn main() {
    let Args {
        speed,
        program,
        disable_sound,
        enable_speed_controls,
    } = Args::parse();

    let speed: Arc<AtomicU64> = Arc::new((speed as u64).into());

    let mut vm = VM::new();
    vm.load_program(&program);
    vm.start_timers(disable_sound);
    vm.display.spawn_render_thread();

    crossterm::terminal::enable_raw_mode().unwrap();
    // Hide cursor once at startup; the Esc handler restores it on exit.
    std::io::stdout()
        .execute(crossterm::cursor::Hide)
        .unwrap();

    vm.start_read_inputs(enable_speed_controls, Arc::clone(&speed));

    // Track measured IPS separately from the target rate. These diverge when
    // the OS sleep is imprecise or the system is under load. Printed at row 34
    // (two rows below the 32-row CHIP-8 display) once per second.
    let instr_count: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    {
        let instr_count = Arc::clone(&instr_count);
        let speed = Arc::clone(&speed);
        thread::spawn(move || loop {
            thread::sleep(Duration::from_secs(1));
            let measured = instr_count.swap(0, Ordering::Relaxed);
            let target = speed.load(Ordering::Relaxed);
            print!(
                "\x1B[34;1H\x1B[2K IPS: {:>6} | target: {:>6}",
                measured, target
            );
            std::io::stdout().flush().unwrap();
        });
    }

    loop {
        vm.run_step();
        instr_count.fetch_add(1, Ordering::Relaxed);
        thread::sleep(Duration::from_micros(
            1_000_000 / speed.load(Ordering::Relaxed),
        ));
    }
}
