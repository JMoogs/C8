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

use clap::Parser;

/// A CHIP-8 interpreter
///
/// The CHIP-8 interpreter is a simple virtual machine that was used to run games on older systems.
/// There are two versions of the interpreter, the original CHIP-8 interpreter, and the Super
/// CHIP-8 interpreter. The Super CHIP-8 interpreter has slightly different instructions to the
/// original CHIP-8 interpreter
/// Seperate binaries can be built for each version by using the `new_instructions` feature.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The speed of the interpreter in instructions per second
    #[arg(short, long, default_value_t = 700)]
    speed: u32,

    /// Enable speed controls
    ///
    /// The left and right arrow keys will decrease and increase the speed by 100.
    ///
    /// The up and down arrow keys will decrease and increase the speed by 1000.
    #[arg(short, long)]
    enable_speed_controls: bool,

    /// Disable sound
    #[arg(short, long)]
    disable_sound: bool,

    /// The program file
    program: String,
}

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
    crossterm::terminal::enable_raw_mode().unwrap();

    vm.start_read_inputs(enable_speed_controls, Arc::clone(&speed));
    loop {
        vm.run_step();
        std::thread::sleep(std::time::Duration::from_micros(
            1_000_000 / speed.load(Ordering::Relaxed),
        ));
    }
}

struct VM {
    /// Memory
    /// The interpreter is stored from 000 to 1FF
    memory: Memory,
    display: Display,
    pc: u16,
    index: u16,
    stack: Vec<u16>,
    delay_timer: Arc<AtomicU8>,
    sound_timer: Arc<AtomicU8>,
    // Variable registers
    v: [u8; 16],
    // Keypad
    keypad: Keypad,
}

impl VM {
    fn new() -> Self {
        Self {
            memory: Memory::new(),
            display: Display::new(),
            // We start from 0x200 because the interpreter is stored from 000 to 1FF in a normal
            // CHIP-8 interpreter
            pc: 0x200,
            index: 0,
            stack: Vec::new(),
            delay_timer: Arc::new(0.into()),
            sound_timer: Arc::new(0.into()),
            v: [0; 16],
            keypad: Keypad::new(),
        }
    }

    fn load_program(&mut self, file: &str) {
        self.memory.load(file);
    }

    fn start_timers(&self, no_sound: bool) {
        let delay_timer = self.delay_timer.clone();
        let sound_timer = self.sound_timer.clone();
        std::thread::spawn(move || loop {
            thread::sleep(Duration::from_micros(1_000_000 / 60));
            if delay_timer.load(Ordering::SeqCst) > 0 {
                if !no_sound {
                    print!("\x07");
                }
                std::io::stdout().flush().unwrap();
            }
            let _ = delay_timer.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                if x > 0 {
                    Some(x - 1)
                } else {
                    None
                }
            });
            let _ = sound_timer.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                if x > 0 {
                    Some(x - 1)
                } else {
                    None
                }
            });
        });
    }

    fn start_read_inputs(&self, enable_speed_controls: bool, speed: Arc<AtomicU64>) {
        let keypad = self.keypad.clone();
        std::thread::spawn(move || loop {
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
                                let s = speed.load(std::sync::atomic::Ordering::SeqCst);
                                let mut s = s.saturating_sub(100);
                                if s == 0 {
                                    s = 1
                                }
                                speed.store(s, std::sync::atomic::Ordering::SeqCst);
                            }
                            continue;
                        }
                        KeyCode::Right => {
                            if enable_speed_controls {
                                let s = speed.load(std::sync::atomic::Ordering::SeqCst);
                                let s = s.saturating_add(100);
                                speed.store(s, std::sync::atomic::Ordering::SeqCst);
                            }
                            continue;
                        }
                        KeyCode::Up => {
                            if enable_speed_controls {
                                let s = speed.load(std::sync::atomic::Ordering::SeqCst);
                                let s = s.saturating_add(1000);
                                speed.store(s, std::sync::atomic::Ordering::SeqCst);
                            }
                            continue;
                        }
                        KeyCode::Down => {
                            if enable_speed_controls {
                                let s = speed.load(std::sync::atomic::Ordering::SeqCst);
                                let mut s = s.saturating_sub(1000);
                                if s == 0 {
                                    s = 1
                                }
                                speed.store(s, std::sync::atomic::Ordering::SeqCst);
                            }
                            continue;
                        }
                        KeyCode::Esc => {
                            crossterm::terminal::disable_raw_mode().unwrap();
                            let mut stdout = std::io::stdout();
                            stdout.execute(crossterm::cursor::Show).unwrap();
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
                _ => (),
            }
        });
    }

    fn run_step(&mut self) {
        let opcode = decode(self.fetch());
        self.execute(opcode);
        self.display.display();
    }
    // Fetch the next opcode from memory, incrementing the program counter
    fn fetch(&mut self) -> u16 {
        let pc = self.pc as usize;
        let opcode = (self.memory.0[pc] as u16) << 8 | self.memory.0[pc + 1] as u16;
        self.pc += 2;
        opcode
    }

    fn execute(&mut self, opcode: Opcode) {
        match opcode {
            Opcode::ClearScreen => self.display.clear_screen(),
            Opcode::Jump(addr) => self.pc = addr,
            Opcode::SetReg((reg, val)) => self.v[reg as usize] = val,
            Opcode::AddToReg((reg, val)) => {
                let reg_val = self.v[reg as usize];
                self.v[reg as usize] = reg_val.wrapping_add(val);
            }
            Opcode::SetIndex(val) => self.index = val,
            Opcode::Display(reg_x, reg_y, n) => {
                self.v[0xF] = 0;
                let mut y = self.v[reg_y as usize] % 32;
                for i in 0..n {
                    // Stop if we reach the end of the screen
                    if y >= 32 {
                        break;
                    }
                    let mut x = self.v[reg_x as usize] % 64;
                    let byte = self.memory.0[self.index as usize + i as usize];
                    // We'll draw a maximum of 8 pixels
                    for j in 0..8 {
                        // Stop if we reach the end of the screen
                        if x >= 64 {
                            break;
                        }

                        let display_pixel = self.display.get_pixel_mut(x, y);
                        let sprite_pixel = (byte >> (7 - j)) & 1 == 1;

                        // If the pixel is already set, and the sprite pixel is set, we need to
                        // toggle the pixel off and set the flag
                        if *display_pixel && sprite_pixel {
                            self.v[0xF] = 1;
                            *display_pixel = false;
                        }
                        // If the pixel is off, and the sprite pixel is set, we need to toggle the
                        // pixel on
                        else if !*display_pixel && sprite_pixel {
                            *display_pixel = true;
                        }

                        // Increment x
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
                if res > 0xFF {
                    self.v[0xF] = 1;
                } else {
                    self.v[0xF] = 0;
                }

                self.v[main as usize] = (res % 0xFF) as u8;
            }
            Opcode::SubtractRegisters { main, other } => {
                self.v[main as usize] = self.v[main as usize].wrapping_sub(self.v[other as usize]);
                if self.v[main as usize] > self.v[other as usize] {
                    self.v[0xF] = 1;
                } else {
                    self.v[0xF] = 0;
                }
            }
            Opcode::ReverseSubtractRegisters { main, other } => {
                self.v[main as usize] = self.v[other as usize].wrapping_sub(self.v[main as usize]);
                if self.v[other as usize] > self.v[main as usize] {
                    self.v[0xF] = 0;
                } else {
                    self.v[0xF] = 1;
                }
            }
            #[cfg(not(feature = "new_instructions"))]
            Opcode::ShiftLeft { main, other } => {
                let shifted_out_bit = self.v[other as usize] >> 7;
                self.v[main as usize] = self.v[other as usize] << 1;
                self.v[0xF] = shifted_out_bit;
            }
            #[cfg(feature = "new_instructions")]
            Opcode::ShiftLeft(main) => {
                self.v[main as usize] = self.v[main as usize] << 1;
            }
            #[cfg(not(feature = "new_instructions"))]
            Opcode::ShiftRight { main, other } => {
                let shifted_out_bit = self.v[other as usize] & 1;
                self.v[main as usize] = self.v[other as usize] >> 1;
                self.v[0xF] = shifted_out_bit;
            }
            #[cfg(feature = "new_instructions")]
            Opcode::ShiftRight(main) => {
                self.v[main as usize] = self.v[main as usize] >> 1;
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
                let random = rand::random::<u8>();
                self.v[reg as usize] = random & val;
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
            Opcode::SetDelayTimer(reg) => self
                .delay_timer
                .store(self.v[reg as usize], Ordering::SeqCst),
            Opcode::SetSoundTimer(reg) => self
                .sound_timer
                .store(self.v[reg as usize], Ordering::SeqCst),
            Opcode::AddToIndex(reg) => {
                self.index += self.v[reg as usize] as u16;
            }
            Opcode::WaitforAnyKeyPress(reg) => {
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
                // Characters are stored from 0x050 to 0x050 + 80
                let character = self.v[reg as usize] & 0xF;
                self.index = 0x050 + (character as u16 * 5)
            }
            Opcode::BinaryCodedDecimal(reg) => {
                let num = self.v[reg as usize];
                self.memory.0[self.index as usize] = num / 100;
                self.memory.0[self.index as usize + 1] = (num / 10) % 10;
                self.memory.0[self.index as usize + 2] = num % 10;
            }
            Opcode::StoreMemory(max_reg) => {
                #[cfg(not(feature = "new_instructions"))]
                {
                    for reg in 0..=max_reg {
                        self.memory.0[self.index as usize] = self.v[reg as usize];
                        self.index += 1;
                    }
                }
                #[cfg(feature = "new_instructions")]
                {
                    for reg in 0..=max_reg {
                        self.memory.0[self.index as usize + reg as usize] = self.v[reg as usize];
                    }
                }
            }
            Opcode::LoadMemory(max_reg) => {
                #[cfg(not(feature = "new_instructions"))]
                {
                    for reg in 0..=max_reg {
                        self.v[reg as usize] = self.memory.0[self.index as usize];
                        self.index += 1;
                    }
                }
                #[cfg(feature = "new_instructions")]
                {
                    for reg in 0..=max_reg {
                        self.v[reg as usize] = self.memory.0[self.index as usize + reg as usize];
                    }
                }
            }
            Opcode::Unknown(x) => panic!("Encountered unknown opcode: {x:#04X}"),
        }
    }
}

struct Memory([u8; 4096]);

impl Memory {
    fn new() -> Memory {
        let mut memory = [0; 4096];
        // Load the fontset into memory
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

struct Display {
    display: [bool; 64 * 32],
    changed: bool,
}

impl Display {
    fn new() -> Self {
        Self {
            display: [false; 64 * 32],
            changed: true,
        }
    }

    fn clear_screen(&mut self) {
        self.display = [false; 64 * 32];
        self.changed = true;
    }

    fn display(&mut self) {
        // Clear the screen
        if self.changed {
            print!("\x1B[2J\x1B[H");
            for x in 0..64 {
                for y in 0..32 {
                    let pixel = if self.display[x + y * 64] { "█" } else { " " };
                    print!("\x1B[{};{}H{}", y + 1, x + 1, pixel);
                }
            }
            self.changed = false
        }
        Self::hide_cursor();
    }

    fn hide_cursor() {
        print!("\x1B[?25l");
    }

    fn get_pixel_mut<T: Into<usize>>(&mut self, x: T, y: T) -> &mut bool {
        // If borrowing the display mutably, we need to set the changed flag
        self.changed = true;
        &mut self.display[x.into() + y.into() * 64]
    }
}

type Register = u8;

enum Opcode {
    ClearScreen,
    /// A jump instruction with the address to jump to
    Jump(u16),
    SubroutineCall(u16),
    SkipEqVal((Register, u8)),
    SkipNeqVal((Register, u8)),
    SkipEqReg((Register, Register)),
    SkipNeqReg((Register, Register)),
    SetReg((Register, u8)),
    AddToReg((Register, u8)),
    SetIndex(u16),
    CopyReg {
        /// The register to copy from
        src: Register,
        /// The register to copy to
        dest: Register,
    },
    BinaryOr {
        /// The register that is changed
        main: Register,
        /// The other register
        other: Register,
    },
    BinaryAnd {
        /// The register that is changed
        main: Register,
        /// The other register
        other: Register,
    },
    LogicalXor {
        /// The register that is changed
        main: Register,
        /// The other register
        other: Register,
    },
    AddRegisters {
        /// The register that is changed
        main: Register,
        /// The other register
        other: Register,
    },
    /// Sets main to main - other
    SubtractRegisters {
        /// The register that is changed
        main: Register,
        /// The other register
        other: Register,
    },
    /// Sets main to other - main
    ReverseSubtractRegisters {
        /// The register that is changed
        main: Register,
        /// The other register
        other: Register,
    },
    #[cfg(not(feature = "new_instructions"))]
    ShiftLeft {
        /// The register that is changed
        main: Register,
        /// The other register
        other: Register,
    },
    #[cfg(feature = "new_instructions")]
    ShiftLeft(Register),
    #[cfg(not(feature = "new_instructions"))]
    ShiftRight {
        /// The register that is changed
        main: Register,
        /// The other register
        other: Register,
    },
    #[cfg(feature = "new_instructions")]
    ShiftRight(Register),
    #[cfg(not(feature = "new_instructions"))]
    JumpWithOffset(u16),
    #[cfg(feature = "new_instructions")]
    JumpWithOffset((Register, u16)),
    Random(Register, u8),
    Display(Register, Register, u8),
    SubroutineReturn,
    SkipIfKeyPressed(Register),
    SkipIfKeyNotPressed(Register),
    SetToDelayTimer(Register),
    SetDelayTimer(Register),
    SetSoundTimer(Register),
    AddToIndex(Register),
    WaitforAnyKeyPress(Register),
    FontCharacter(Register),
    BinaryCodedDecimal(Register),
    StoreMemory(Register),
    LoadMemory(Register),
    Unknown(u16),
}

/// Decode an opcode
fn decode(opcode: u16) -> Opcode {
    let nibble1 = (opcode & 0xF000) >> 12;
    let nibble2 = (opcode & 0x0F00) >> 8;
    let nibble3 = (opcode & 0x00F0) >> 4;
    let nibble4 = opcode & 0x000F;

    match nibble1 {
        0x0 => match opcode {
            0x00E0 => Opcode::ClearScreen,
            0x00EE => Opcode::SubroutineReturn,
            x => Opcode::Unknown(x),
        },
        0x1 => {
            let addr = opcode & 0x0FFF;
            Opcode::Jump(addr)
        }
        0x2 => {
            let addr = opcode & 0x0FFF;
            Opcode::SubroutineCall(addr)
        }
        0x3 => {
            let reg = nibble2;
            let value = opcode & 0x00FF;
            Opcode::SkipEqVal((reg as u8, value as u8))
        }
        0x4 => {
            let reg = nibble2;
            let value = opcode & 0x00FF;
            Opcode::SkipNeqVal((reg as u8, value as u8))
        }
        0x5 => match nibble4 {
            0x0 => {
                let reg1 = nibble2;
                let reg2 = nibble3;
                Opcode::SkipEqReg((reg1 as u8, reg2 as u8))
            }
            _ => Opcode::Unknown(opcode),
        },
        0x6 => {
            let reg = nibble2;
            let value = opcode & 0x00FF;
            Opcode::SetReg((reg as u8, value as u8))
        }
        0x7 => {
            let reg = nibble2;
            let value = opcode & 0x00FF;
            Opcode::AddToReg((reg as u8, value as u8))
        }
        0x8 => match nibble4 {
            0x0 => Opcode::CopyReg {
                src: nibble3 as u8,
                dest: nibble2 as u8,
            },
            0x1 => Opcode::BinaryOr {
                main: nibble2 as u8,
                other: nibble3 as u8,
            },
            0x2 => Opcode::BinaryAnd {
                main: nibble2 as u8,
                other: nibble3 as u8,
            },
            0x3 => Opcode::LogicalXor {
                main: nibble2 as u8,
                other: nibble3 as u8,
            },
            0x4 => Opcode::AddRegisters {
                main: nibble2 as u8,
                other: nibble3 as u8,
            },
            0x5 => Opcode::SubtractRegisters {
                main: nibble2 as u8,
                other: nibble3 as u8,
            },
            0x6 => {
                #[cfg(not(feature = "new_instructions"))]
                let op = Opcode::ShiftRight {
                    main: nibble2 as u8,
                    other: nibble3 as u8,
                };
                #[cfg(feature = "new_instructions")]
                let op = Opcode::ShiftRight(nibble2 as u8);

                op
            }
            0x7 => Opcode::ReverseSubtractRegisters {
                main: nibble2 as u8,
                other: nibble3 as u8,
            },
            0xE => {
                #[cfg(not(feature = "new_instructions"))]
                let op = Opcode::ShiftLeft {
                    main: nibble2 as u8,
                    other: nibble3 as u8,
                };
                #[cfg(feature = "new_instructions")]
                let op = Opcode::ShiftLeft(nibble2 as u8);

                op
            }
            _ => Opcode::Unknown(opcode),
        },
        0x9 => match nibble4 {
            0x0 => {
                let reg1 = nibble2;
                let reg2 = nibble3;
                Opcode::SkipNeqReg((reg1 as u8, reg2 as u8))
            }
            _ => Opcode::Unknown(opcode),
        },
        0xA => Opcode::SetIndex(opcode & 0x0FFF),
        0xB => {
            #[cfg(not(feature = "new_instructions"))]
            let op = Opcode::JumpWithOffset(opcode & 0x0FFF);
            #[cfg(feature = "new_instructions")]
            let op = Opcode::JumpWithOffset((nibble2 as u8, opcode & 0x0FFF));

            op
        }
        0xC => Opcode::Random(nibble2 as u8, (opcode & 0x00FF) as u8),
        0xD => Opcode::Display(nibble2 as u8, nibble3 as u8, nibble4 as u8),
        0xE => match opcode & 0x00FF {
            0x9E => Opcode::SkipIfKeyPressed(nibble2 as u8),
            0xA1 => Opcode::SkipIfKeyNotPressed(nibble2 as u8),
            _ => Opcode::Unknown(opcode),
        },
        0xF => match opcode & 0x00FF {
            0x07 => Opcode::SetToDelayTimer(nibble2 as u8),
            0x15 => Opcode::SetDelayTimer(nibble2 as u8),
            0x18 => Opcode::SetSoundTimer(nibble2 as u8),
            0x1E => Opcode::AddToIndex(nibble2 as u8),
            0x0A => Opcode::WaitforAnyKeyPress(nibble2 as u8),
            0x29 => Opcode::FontCharacter(nibble2 as u8),
            0x33 => Opcode::BinaryCodedDecimal(nibble2 as u8),
            0x55 => Opcode::StoreMemory(nibble2 as u8),
            0x65 => Opcode::LoadMemory(nibble2 as u8),
            _ => Opcode::Unknown(opcode),
        },
        _ => Opcode::Unknown(opcode),
    }
}
