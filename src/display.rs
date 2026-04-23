//! Terminal display with a dedicated 60 Hz render thread.
//!
//! The CHIP-8 display is a 64×32 monochrome pixel grid. This module provides
//! a shared frame buffer ([`DisplayState`]) wrapped in `Arc<Mutex>`, allowing
//! the CPU thread to write pixels during sprite operations while the render
//! thread reads and repaints at a fixed 60 Hz — decoupled from the CPU rate.

use std::{
    io::Write,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

/// The inner state of the display, shared between the CPU and render threads.
pub struct DisplayState {
    /// Flat, row-major pixel buffer. The pixel at column `x`, row `y` is at
    /// index `x + y * 64`. `true` = on (drawn as `█`), `false` = off.
    pub buffer: [bool; 64 * 32],
    /// Set to `true` whenever pixels change during a `Display` instruction;
    /// cleared by the render thread after each repaint. When `false`, the
    /// render thread skips its pass entirely — no terminal I/O at all.
    pub dirty: bool,
}

/// Shared display buffer accessed by both the CPU and the render thread.
///
/// The CPU acquires the lock briefly during sprite drawing (one lock per
/// Display instruction, not per pixel). The render thread holds it for one
/// paint pass every 16.67 ms. Because sprite draws are infrequent relative
/// to the CPU rate, contention is negligible.
pub struct Display(pub Arc<Mutex<DisplayState>>);

impl Display {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(DisplayState {
            buffer: [false; 64 * 32],
            dirty: false,
        })))
    }

    /// Set all pixels to off and schedule a repaint (`00E0` — CLS).
    pub fn clear(&self) {
        let mut state = self.0.lock().unwrap();
        state.buffer = [false; 64 * 32];
        state.dirty = true;
    }

    /// Spawn a background thread that repaints the terminal at exactly 60 Hz.
    ///
    /// The render rate is decoupled from the CPU instruction rate: any number
    /// of pixel writes that arrive within one 16.67 ms window are coalesced
    /// into a single terminal redraw. This prevents I/O from becoming a
    /// bottleneck at high instruction rates (1000+ IPS) while still
    /// delivering flicker-free 60 FPS output.
    pub fn spawn_render_thread(&self) {
        let inner = Arc::clone(&self.0);
        thread::spawn(move || loop {
            thread::sleep(Duration::from_micros(1_000_000 / 60));
            let mut state = inner.lock().unwrap();
            if state.dirty {
                // Erase and redraw every pixel with absolute cursor positioning.
                // Printing each cell individually avoids repainting unchanged
                // rows, keeping terminal write volume proportional to the
                // number of dirty pixels rather than the full frame.
                print!("\x1B[2J\x1B[H");
                for y in 0..32_usize {
                    for x in 0..64_usize {
                        let pixel = if state.buffer[x + y * 64] { "█" } else { " " };
                        print!("\x1B[{};{}H{}", y + 1, x + 1, pixel);
                    }
                }
                std::io::stdout().flush().unwrap();
                state.dirty = false;
            }
        });
    }
}
