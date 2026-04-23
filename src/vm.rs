//! CHIP-8 virtual machine: memory, registers, and the fetch-decode-execute loop.
//!
//! The [`VM`] struct owns all interpreter state. Execution is single-threaded
//! on the main thread; timer decrement and input polling run concurrently in
//! background threads that share only the fields they need via atomics and a
//! single mutex — the hot interpreter path acquires no locks during a typical
//! instruction cycle.

use std::{
    io::Write,
    sync::{
        atomic::{AtomicU64, AtomicU8, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

use crossterm::{
    event::{Event, KeyCode, KeyEventKind},
    ExecutableCommand,
};

use crate::{
    display::Display,
    opcode::{decode, Opcode},
};

/// The CHIP-8 virtual machine.
///
/// Owns the full interpreter state: memory, registers, stack, timers, display,
/// and keypad. Background threads for timers, rendering, and input are spawned
/// via the `start_*` methods and communicate through the shared fields.
pub struct VM {
    /// 4 KB flat address space. Layout: fontset at `0x050–0x09F`, program ROM
    /// loaded at `0x200`, everything below `0x200` reserved.
    memory: Memory,
    /// Pixel frame buffer shared with the dedicated render thread.
    pub display: Display,
    /// Program counter — address of the next instruction to fetch.
    pc: u16,
    /// Index register `I` — used by memory, sprite, and font instructions.
    index: u16,
    /// Subroutine return-address stack. The original hardware supported 16
    /// levels; this implementation uses a growable `Vec` for safety.
    stack: Vec<u16>,
    /// Decremented at 60 Hz by the timer thread. Programs poll this to
    /// implement delays; read lock-free from the interpreter via `Ordering::SeqCst`.
    delay_timer: Arc<AtomicU8>,
    /// Decremented at 60 Hz by the timer thread. Non-zero causes the terminal
    /// bell character to fire once per tick.
    sound_timer: Arc<AtomicU8>,
    /// General-purpose registers `V0–VF`. `VF` (index `0xF`) acts as a flag
    /// register and is overwritten by arithmetic, shift, and sprite draw
    /// instructions; programs avoid using it as ordinary storage.
    v: [u8; 16],
    /// 16-key hex keypad; state shared with the input thread via `Arc<Mutex>`.
    keypad: Keypad,
}

impl VM {
    pub fn new() -> Self {
        Self {
            memory: Memory::new(),
            display: Display::new(),
            // Programs are loaded at 0x200; the interpreter ROM occupied 0x000–0x1FF
            // on original hardware and the fontset lives at 0x050–0x09F.
            pc: 0x200,
            index: 0,
            stack: Vec::new(),
            delay_timer: Arc::new(0.into()),
            sound_timer: Arc::new(0.into()),
            v: [0; 16],
            keypad: Keypad::new(),
        }
    }

    pub fn load_program(&mut self, file: &str) {
        self.memory.load(file);
    }

    /// Spawn the timer thread, which decrements both timers at exactly 60 Hz.
    ///
    /// The CHIP-8 spec requires timers to count down at 60 Hz regardless of
    /// the CPU instruction rate. A dedicated thread with a fixed 16.67 ms sleep
    /// achieves this without coupling timer accuracy to instruction throughput.
    ///
    /// When `no_sound` is false, the terminal BEL character (`\x07`) is printed
    /// each tick while the sound timer is non-zero, producing an audible beep.
    pub fn start_timers(&self, no_sound: bool) {
        let delay_timer = Arc::clone(&self.delay_timer);
        let sound_timer = Arc::clone(&self.sound_timer);
        thread::spawn(move || loop {
            thread::sleep(Duration::from_micros(1_000_000 / 60));
            // Ring the terminal bell while the sound timer is active.
            if sound_timer.load(Ordering::SeqCst) > 0 && !no_sound {
                print!("\x07");
                std::io::stdout().flush().unwrap();
            }
            let _ = delay_timer.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                if x > 0 { Some(x - 1) } else { None }
            });
            let _ = sound_timer.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                if x > 0 { Some(x - 1) } else { None }
            });
        });
    }

    /// Spawn the input thread, which maps keyboard events to CHIP-8 keypad state.
    ///
    /// The CHIP-8 keypad is a 4×4 hex grid (`0x0–0xF`). This maps it to the
    /// left block of a QWERTY keyboard:
    ///
    /// ```text
    /// CHIP-8   Keyboard
    /// 1 2 3 C  →  1 2 3 4
    /// 4 5 6 D  →  q w e r
    /// 7 8 9 E  →  a s d f
    /// A 0 B F  →  z x c v
    /// ```
    ///
    /// When `enable_speed_controls` is true, arrow keys adjust the interpreter
    /// rate (±100 / ±1000 IPS). `Esc` restores the terminal and exits.
    pub fn start_read_inputs(&self, enable_speed_controls: bool, speed: Arc<AtomicU64>) {
        let keypad = self.keypad.clone();
        thread::spawn(move || loop {
            match crossterm::event::read().unwrap() {
                Event::Key(event) => {
                    let key = event.code;
                    let reg = match key {
                        KeyCode::Char('1') => 0x0,
                        KeyCode::Char('2') => 0x1,
                        KeyCode::Char('3') => 0x2,
                        KeyCode::Char('4') => 0x3,
                        KeyCode::Char('q') => 0x4,
                        KeyCode::Char('w') => 0x5,
                        KeyCode::Char('e') => 0x6,
                        KeyCode::Char('r') => 0x7,
                        KeyCode::Char('a') => 0x8,
                        KeyCode::Char('s') => 0x9,
                        KeyCode::Char('d') => 0xA,
                        KeyCode::Char('f') => 0xB,
                        KeyCode::Char('z') => 0xC,
                        KeyCode::Char('x') => 0xD,
                        KeyCode::Char('c') => 0xE,
                        KeyCode::Char('v') => 0xF,
                        KeyCode::Left => {
                            if enable_speed_controls {
                                let s = speed.load(Ordering::SeqCst);
                                speed.store(s.saturating_sub(100).max(1), Ordering::SeqCst);
                            }
                            continue;
                        }
                        KeyCode::Right => {
                            if enable_speed_controls {
                                let s = speed.load(Ordering::SeqCst);
                                speed.store(s.saturating_add(100), Ordering::SeqCst);
                            }
                            continue;
                        }
                        KeyCode::Up => {
                            if enable_speed_controls {
                                let s = speed.load(Ordering::SeqCst);
                                speed.store(s.saturating_add(1000), Ordering::SeqCst);
                            }
                            continue;
                        }
                        KeyCode::Down => {
                            if enable_speed_controls {
                                let s = speed.load(Ordering::SeqCst);
                                speed.store(s.saturating_sub(1000).max(1), Ordering::SeqCst);
                            }
                            continue;
                        }
                        KeyCode::Esc => {
                            crossterm::terminal::disable_raw_mode().unwrap();
                            std::io::stdout()
                                .execute(crossterm::cursor::Show)
                                .unwrap();
                            std::process::exit(0);
                        }
                        _ => continue,
                    };

                    if event.kind == KeyEventKind::Press {
                        keypad.0.lock().unwrap()[reg as usize] = true;
                    } else if event.kind == KeyEventKind::Release {
                        keypad.0.lock().unwrap()[reg as usize] = false;
                    }
                }
                _ => {}
            }
        });
    }

    /// Advance the interpreter by one fetch-decode-execute cycle.
    pub fn run_step(&mut self) {
        let opcode = decode(self.fetch());
        self.execute(opcode);
    }

    /// Fetch the next 16-bit opcode from memory and advance the program counter.
    ///
    /// Two consecutive bytes are combined big-endian: `memory[PC] << 8 | memory[PC+1]`.
    /// PC advances by 2 after each fetch. Skip instructions add a further 2
    /// (total +4) to step over the following instruction without executing it.
    fn fetch(&mut self) -> u16 {
        let pc = self.pc as usize;
        let opcode = (self.memory.0[pc] as u16) << 8 | self.memory.0[pc + 1] as u16;
        self.pc += 2;
        opcode
    }

    /// Execute a decoded instruction, mutating VM state in place.
    ///
    /// Each arm implements exactly the semantics of the corresponding opcode.
    /// VF write ordering is significant for arithmetic instructions: the flag
    /// is written *after* the result register so that programs which use `VF`
    /// as an operand observe the carry/borrow rather than the truncated sum.
    fn execute(&mut self, opcode: Opcode) {
        match opcode {
            Opcode::ClearScreen => self.display.clear(),
            Opcode::Jump(addr) => self.pc = addr,
            Opcode::SetReg((reg, val)) => self.v[reg as usize] = val,
            Opcode::AddToReg((reg, val)) => {
                self.v[reg as usize] = self.v[reg as usize].wrapping_add(val);
            }
            Opcode::SetIndex(val) => self.index = val,
            Opcode::Display(reg_x, reg_y, n) => {
                self.v[0xF] = 0;
                // Hold the display lock for the entire sprite draw to avoid
                // partial frames being rendered mid-instruction.
                let mut display = self.display.0.lock().unwrap();
                let mut y = self.v[reg_y as usize] % 32;
                for i in 0..n {
                    if y >= 32 {
                        break;
                    }
                    let mut x = self.v[reg_x as usize] % 64;
                    let byte = self.memory.0[self.index as usize + i as usize];
                    for j in 0..8 {
                        if x >= 64 {
                            break;
                        }
                        let idx = x as usize + y as usize * 64;
                        let sprite_pixel = (byte >> (7 - j)) & 1 == 1;
                        if sprite_pixel {
                            // XOR: if both set, pixel turns off and VF flags collision.
                            if display.buffer[idx] {
                                self.v[0xF] = 1;
                                display.buffer[idx] = false;
                            } else {
                                display.buffer[idx] = true;
                            }
                            display.dirty = true;
                        }
                        x += 1;
                    }
                    y += 1;
                }
            }
            Opcode::SubroutineCall(addr) => {
                self.stack.push(self.pc);
                self.pc = addr;
            }
            Opcode::SubroutineReturn => {
                self.pc = self.stack.pop().expect("Stack underflow");
            }
            Opcode::SkipEqVal((reg, val)) => {
                if self.v[reg as usize] == val {
                    self.pc += 2;
                }
            }
            Opcode::SkipNeqVal((reg, val)) => {
                if self.v[reg as usize] != val {
                    self.pc += 2;
                }
            }
            Opcode::SkipEqReg((reg1, reg2)) => {
                if self.v[reg1 as usize] == self.v[reg2 as usize] {
                    self.pc += 2;
                }
            }
            Opcode::SkipNeqReg((reg1, reg2)) => {
                if self.v[reg1 as usize] != self.v[reg2 as usize] {
                    self.pc += 2;
                }
            }
            Opcode::CopyReg { src, dest } => {
                self.v[dest as usize] = self.v[src as usize];
            }
            Opcode::BinaryOr { main, other } => {
                self.v[main as usize] |= self.v[other as usize];
            }
            Opcode::BinaryAnd { main, other } => {
                self.v[main as usize] &= self.v[other as usize];
            }
            Opcode::LogicalXor { main, other } => {
                self.v[main as usize] ^= self.v[other as usize];
            }
            Opcode::AddRegisters { main, other } => {
                let res = self.v[main as usize] as u16 + self.v[other as usize] as u16;
                // Write result before VF so that carry is correct even when main == 0xF.
                self.v[main as usize] = res as u8;
                self.v[0xF] = if res > 0xFF { 1 } else { 0 };
            }
            Opcode::SubtractRegisters { main, other } => {
                // VF = NOT borrow: 1 if Vx >= Vy, 0 if Vx < Vy.
                // Snapshot operands before writing so VF is correct when main == 0xF.
                let vx = self.v[main as usize];
                let vy = self.v[other as usize];
                self.v[main as usize] = vx.wrapping_sub(vy);
                self.v[0xF] = if vx >= vy { 1 } else { 0 };
            }
            Opcode::ReverseSubtractRegisters { main, other } => {
                // VF = NOT borrow: 1 if Vy >= Vx, 0 if Vy < Vx.
                let vx = self.v[main as usize];
                let vy = self.v[other as usize];
                self.v[main as usize] = vy.wrapping_sub(vx);
                self.v[0xF] = if vy >= vx { 1 } else { 0 };
            }
            #[cfg(not(feature = "new_instructions"))]
            Opcode::ShiftLeft { main, other } => {
                let shifted_out_bit = self.v[other as usize] >> 7;
                self.v[main as usize] = self.v[other as usize] << 1;
                self.v[0xF] = shifted_out_bit;
            }
            #[cfg(feature = "new_instructions")]
            Opcode::ShiftLeft(main) => {
                self.v[main as usize] <<= 1;
            }
            #[cfg(not(feature = "new_instructions"))]
            Opcode::ShiftRight { main, other } => {
                let shifted_out_bit = self.v[other as usize] & 1;
                self.v[main as usize] = self.v[other as usize] >> 1;
                self.v[0xF] = shifted_out_bit;
            }
            #[cfg(feature = "new_instructions")]
            Opcode::ShiftRight(main) => {
                self.v[main as usize] >>= 1;
            }
            #[cfg(not(feature = "new_instructions"))]
            Opcode::JumpWithOffset(offset) => {
                self.pc = self.v[0x0] as u16 + offset;
            }
            #[cfg(feature = "new_instructions")]
            Opcode::JumpWithOffset((reg, offset)) => {
                self.pc = self.v[reg as usize] as u16 + offset as u16;
            }
            Opcode::Random(reg, val) => {
                self.v[reg as usize] = rand::random::<u8>() & val;
            }
            Opcode::SkipIfKeyPressed(reg) => {
                if self.keypad.0.lock().unwrap()[self.v[reg as usize] as usize] {
                    self.pc += 2;
                }
            }
            Opcode::SkipIfKeyNotPressed(reg) => {
                if !self.keypad.0.lock().unwrap()[self.v[reg as usize] as usize] {
                    self.pc += 2;
                }
            }
            Opcode::SetToDelayTimer(reg) => {
                self.v[reg as usize] = self.delay_timer.load(Ordering::SeqCst);
            }
            Opcode::SetDelayTimer(reg) => {
                self.delay_timer.store(self.v[reg as usize], Ordering::SeqCst);
            }
            Opcode::SetSoundTimer(reg) => {
                self.sound_timer.store(self.v[reg as usize], Ordering::SeqCst);
            }
            Opcode::AddToIndex(reg) => {
                self.index += self.v[reg as usize] as u16;
            }
            Opcode::WaitForAnyKeyPress(reg) => {
                // Re-execute this instruction on the next cycle until a key is held.
                let mut key_pressed = false;
                for (i, key) in self.keypad.0.lock().unwrap().iter().enumerate() {
                    if *key {
                        self.v[reg as usize] = i as u8;
                        key_pressed = true;
                        break;
                    }
                }
                if !key_pressed {
                    self.pc -= 2;
                }
            }
            Opcode::FontCharacter(reg) => {
                // Each glyph is 5 bytes; fontset starts at 0x050.
                let character = self.v[reg as usize] & 0xF;
                self.index = 0x050 + (character as u16 * 5);
            }
            Opcode::BinaryCodedDecimal(reg) => {
                let num = self.v[reg as usize];
                self.memory.0[self.index as usize] = num / 100;
                self.memory.0[self.index as usize + 1] = (num / 10) % 10;
                self.memory.0[self.index as usize + 2] = num % 10;
            }
            Opcode::StoreMemory(max_reg) => {
                #[cfg(not(feature = "new_instructions"))]
                for reg in 0..=max_reg {
                    self.memory.0[self.index as usize] = self.v[reg as usize];
                    self.index += 1;
                }
                #[cfg(feature = "new_instructions")]
                for reg in 0..=max_reg {
                    self.memory.0[self.index as usize + reg as usize] = self.v[reg as usize];
                }
            }
            Opcode::LoadMemory(max_reg) => {
                #[cfg(not(feature = "new_instructions"))]
                for reg in 0..=max_reg {
                    self.v[reg as usize] = self.memory.0[self.index as usize];
                    self.index += 1;
                }
                #[cfg(feature = "new_instructions")]
                for reg in 0..=max_reg {
                    self.v[reg as usize] = self.memory.0[self.index as usize + reg as usize];
                }
            }
            Opcode::Unknown(x) => eprintln!("Unknown opcode: {x:#06X}"),
        }
    }
}

/// Flat 4 KB address space mirroring original CHIP-8 hardware.
///
/// Memory layout:
/// - `0x000–0x04F` — Reserved (held the interpreter ROM on real hardware).
/// - `0x050–0x09F` — Built-in fontset: 16 glyphs × 5 bytes, pre-loaded at init.
/// - `0x0A0–0x1FF` — Unused.
/// - `0x200–0xFFF` — Program ROM; all user ROMs are loaded here at startup.
struct Memory([u8; 4096]);

impl Memory {
    fn new() -> Memory {
        let mut memory = [0; 4096];
        let fontset = [
            0xF0, 0x90, 0x90, 0x90, 0xF0, // 0
            0x20, 0x60, 0x20, 0x20, 0x70, // 1
            0xF0, 0x10, 0xF0, 0x80, 0xF0, // 2
            0xF0, 0x10, 0xF0, 0x10, 0xF0, // 3
            0x90, 0x90, 0xF0, 0x10, 0x10, // 4
            0xF0, 0x80, 0xF0, 0x10, 0xF0, // 5
            0xF0, 0x80, 0xF0, 0x90, 0xF0, // 6
            0xF0, 0x10, 0x20, 0x40, 0x40, // 7
            0xF0, 0x90, 0xF0, 0x90, 0xF0, // 8
            0xF0, 0x90, 0xF0, 0x10, 0xF0, // 9
            0xF0, 0x90, 0xF0, 0x90, 0x90, // A
            0xE0, 0x90, 0xE0, 0x90, 0xE0, // B
            0xF0, 0x80, 0x80, 0x80, 0xF0, // C
            0xE0, 0x90, 0x90, 0x90, 0xE0, // D
            0xF0, 0x80, 0xF0, 0x80, 0xF0, // E
            0xF0, 0x80, 0xF0, 0x80, 0x80, // F
        ];
        memory[0x050..(0x050 + 80)].copy_from_slice(&fontset);
        Self(memory)
    }

    fn load(&mut self, file: &str) {
        let program = std::fs::read(file).expect("Failed to read file");
        self.0[0x200..(0x200 + program.len())].copy_from_slice(&program);
    }
}

/// 16-key hex keypad with shared ownership between the VM and the input thread.
///
/// The array is indexed by CHIP-8 key number (`0x0–0xF`). Each entry is `true`
/// while the corresponding key is physically held down. The input thread writes
/// on press/release events; the interpreter reads during `SkipIfKeyPressed`,
/// `SkipIfKeyNotPressed`, and `WaitForAnyKeyPress`.
struct Keypad(Arc<Mutex<[bool; 16]>>);

impl Keypad {
    fn new() -> Self {
        Self(Arc::new(Mutex::new([false; 16])))
    }
}

impl Clone for Keypad {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcode::Opcode;

    fn make_vm() -> VM {
        VM::new()
    }

    #[test]
    fn set_reg() {
        let mut vm = make_vm();
        vm.execute(Opcode::SetReg((3, 42)));
        assert_eq!(vm.v[3], 42);
    }

    #[test]
    fn add_to_reg_wraps() {
        let mut vm = make_vm();
        vm.v[0] = 250;
        vm.execute(Opcode::AddToReg((0, 10)));
        assert_eq!(vm.v[0], 4); // 260 % 256
    }

    #[test]
    fn jump() {
        let mut vm = make_vm();
        vm.execute(Opcode::Jump(0x300));
        assert_eq!(vm.pc, 0x300);
    }

    #[test]
    fn subroutine_call_and_return() {
        let mut vm = make_vm();
        let original_pc = vm.pc;
        vm.execute(Opcode::SubroutineCall(0x400));
        assert_eq!(vm.pc, 0x400);
        vm.execute(Opcode::SubroutineReturn);
        assert_eq!(vm.pc, original_pc);
    }

    #[test]
    fn add_registers_no_overflow() {
        let mut vm = make_vm();
        vm.v[0] = 10;
        vm.v[1] = 20;
        vm.execute(Opcode::AddRegisters { main: 0, other: 1 });
        assert_eq!(vm.v[0], 30);
        assert_eq!(vm.v[0xF], 0);
    }

    #[test]
    fn add_registers_overflow_sets_carry() {
        let mut vm = make_vm();
        vm.v[0] = 200;
        vm.v[1] = 100;
        vm.execute(Opcode::AddRegisters { main: 0, other: 1 });
        assert_eq!(vm.v[0], 44); // 300 & 0xFF
        assert_eq!(vm.v[0xF], 1);
    }

    #[test]
    fn add_registers_result_255_not_zero() {
        // Regression: previous code used `% 0xFF` which mapped 255 → 0.
        let mut vm = make_vm();
        vm.v[0] = 200;
        vm.v[1] = 55;
        vm.execute(Opcode::AddRegisters { main: 0, other: 1 });
        assert_eq!(vm.v[0], 255);
        assert_eq!(vm.v[0xF], 0);
    }

    #[test]
    fn subtract_no_borrow() {
        let mut vm = make_vm();
        vm.v[0] = 50;
        vm.v[1] = 30;
        vm.execute(Opcode::SubtractRegisters { main: 0, other: 1 });
        assert_eq!(vm.v[0], 20);
        assert_eq!(vm.v[0xF], 1); // no borrow
    }

    #[test]
    fn subtract_with_borrow() {
        let mut vm = make_vm();
        vm.v[0] = 10;
        vm.v[1] = 30;
        vm.execute(Opcode::SubtractRegisters { main: 0, other: 1 });
        assert_eq!(vm.v[0], 236u8); // wrapping: 10u8.wrapping_sub(30)
        assert_eq!(vm.v[0xF], 0); // borrow occurred
    }

    #[test]
    fn reverse_subtract_no_borrow() {
        let mut vm = make_vm();
        vm.v[0] = 10; // Vx
        vm.v[1] = 50; // Vy
        vm.execute(Opcode::ReverseSubtractRegisters { main: 0, other: 1 });
        assert_eq!(vm.v[0], 40); // Vy - Vx
        assert_eq!(vm.v[0xF], 1);
    }

    #[test]
    fn reverse_subtract_with_borrow() {
        let mut vm = make_vm();
        vm.v[0] = 50; // Vx
        vm.v[1] = 10; // Vy
        vm.execute(Opcode::ReverseSubtractRegisters { main: 0, other: 1 });
        assert_eq!(vm.v[0xF], 0); // borrow: Vy < Vx
    }

    #[test]
    fn bcd_decode() {
        let mut vm = make_vm();
        vm.v[0] = 123;
        vm.index = 0x300;
        vm.execute(Opcode::BinaryCodedDecimal(0));
        assert_eq!(vm.memory.0[0x300], 1);
        assert_eq!(vm.memory.0[0x301], 2);
        assert_eq!(vm.memory.0[0x302], 3);
    }

    #[test]
    fn font_character_index() {
        let mut vm = make_vm();
        vm.v[0] = 0xA;
        vm.execute(Opcode::FontCharacter(0));
        assert_eq!(vm.index, 0x050 + 0xA * 5);
    }
}
