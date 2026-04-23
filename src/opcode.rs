//! CHIP-8 instruction set representation and decoding.
//!
//! Each [`Opcode`] variant maps one-to-one to a CHIP-8 instruction. The
//! [`decode`] function parses a raw 16-bit big-endian word fetched from
//! memory into the appropriate variant by inspecting its nibbles.
//!
//! Several instructions differ between the original CHIP-8 and the
//! CHIP-48/SCHIP variant. These are gated behind the `new_instructions`
//! feature flag, which changes the shape of `ShiftLeft`, `ShiftRight`, and
//! `JumpWithOffset`, and the index-register side-effects of `StoreMemory`
//! and `LoadMemory`.

/// A register index in the range 0x0–0xF. Register 0xF (`VF`) doubles as the
/// flag register and is overwritten by arithmetic, shift, and sprite
/// instructions — programs generally avoid using it as general storage.
pub type Register = u8;

/// A decoded CHIP-8 instruction.
///
/// Variants are named after their logical operation rather than their raw
/// opcode pattern. The opcode pattern is noted in each variant's doc comment
/// using the conventional notation:
/// - `N` / `NNN` — 4-bit / 12-bit immediate address
/// - `X` / `Y` — 4-bit register index
/// - `NN` — 8-bit immediate value
pub enum Opcode {
    /// `00E0` — Clear the display; set all pixels to off.
    ClearScreen,
    /// `1NNN` — Unconditional jump; set PC to `NNN`.
    Jump(u16),
    /// `2NNN` — Call subroutine at `NNN`; push current PC onto the stack.
    SubroutineCall(u16),
    /// `00EE` — Return from subroutine; pop the stack into PC.
    SubroutineReturn,
    /// `3XNN` — Skip the next instruction if `VX == NN`.
    SkipEqVal((Register, u8)),
    /// `4XNN` — Skip the next instruction if `VX != NN`.
    SkipNeqVal((Register, u8)),
    /// `5XY0` — Skip the next instruction if `VX == VY`.
    SkipEqReg((Register, Register)),
    /// `9XY0` — Skip the next instruction if `VX != VY`.
    SkipNeqReg((Register, Register)),
    /// `6XNN` — Set `VX = NN`.
    SetReg((Register, u8)),
    /// `7XNN` — Set `VX = VX + NN`. No carry; `VF` is not affected.
    AddToReg((Register, u8)),
    /// `ANNN` — Set the index register `I` to `NNN`.
    SetIndex(u16),
    /// `8XY0` — Set `VX = VY`.
    CopyReg {
        src: Register,
        dest: Register,
    },
    /// `8XY1` — Set `VX = VX | VY` (bitwise OR).
    BinaryOr {
        main: Register,
        other: Register,
    },
    /// `8XY2` — Set `VX = VX & VY` (bitwise AND).
    BinaryAnd {
        main: Register,
        other: Register,
    },
    /// `8XY3` — Set `VX = VX ^ VY` (bitwise XOR).
    LogicalXor {
        main: Register,
        other: Register,
    },
    /// `8XY4` — Set `VX = VX + VY`; `VF = 1` on carry (result > 255), else `0`.
    ///
    /// VF is written *after* the result, so if `X == 0xF` the carry takes
    /// precedence over the truncated sum.
    AddRegisters {
        main: Register,
        other: Register,
    },
    /// `8XY5` — Set `VX = VX - VY`; `VF = 1` if no borrow (`VX >= VY`), else `0`.
    SubtractRegisters {
        main: Register,
        other: Register,
    },
    /// `8XY7` — Set `VX = VY - VX`; `VF = 1` if no borrow (`VY >= VX`), else `0`.
    ReverseSubtractRegisters {
        main: Register,
        other: Register,
    },
    /// `8XYE` — Shift left by 1; `VF` = the shifted-out MSB.
    ///
    /// **Original:** `VX = VY << 1` (source register is `VY`).
    /// **`new_instructions`:** `VX = VX << 1` (in-place; `VY` is ignored).
    #[cfg(not(feature = "new_instructions"))]
    ShiftLeft {
        main: Register,
        other: Register,
    },
    #[cfg(feature = "new_instructions")]
    ShiftLeft(Register),
    /// `8XY6` — Shift right by 1; `VF` = the shifted-out LSB.
    ///
    /// **Original:** `VX = VY >> 1` (source register is `VY`).
    /// **`new_instructions`:** `VX = VX >> 1` (in-place; `VY` is ignored).
    #[cfg(not(feature = "new_instructions"))]
    ShiftRight {
        main: Register,
        other: Register,
    },
    #[cfg(feature = "new_instructions")]
    ShiftRight(Register),
    /// `BNNN` — Jump to address with a register offset.
    ///
    /// **Original:** `PC = V0 + NNN`.
    /// **`new_instructions`:** `PC = VX + NNN`, where `X` is the high nibble of `NNN`.
    #[cfg(not(feature = "new_instructions"))]
    JumpWithOffset(u16),
    #[cfg(feature = "new_instructions")]
    JumpWithOffset((Register, u16)),
    /// `CXNN` — Set `VX = random_byte & NN`. Uses the system RNG.
    Random(Register, u8),
    /// `DXYN` — Draw an N-row sprite from `memory[I]` at pixel `(VX, VY)`.
    ///
    /// Each byte of the sprite is a row of 8 pixels, MSB on the left.
    /// Pixels are XOR'd onto the display; `VF = 1` if any set pixel is
    /// turned off (collision), else `0`. Drawing wraps at the screen boundary.
    Display(Register, Register, u8),
    /// `EX9E` — Skip the next instruction if the key numbered `VX` is pressed.
    SkipIfKeyPressed(Register),
    /// `EXA1` — Skip the next instruction if the key numbered `VX` is not pressed.
    SkipIfKeyNotPressed(Register),
    /// `FX07` — Set `VX` to the current value of the delay timer.
    SetToDelayTimer(Register),
    /// `FX15` — Set the delay timer to `VX`.
    SetDelayTimer(Register),
    /// `FX18` — Set the sound timer to `VX`. The terminal bell fires each
    /// 60 Hz tick while this timer is non-zero.
    SetSoundTimer(Register),
    /// `FX1E` — Set `I = I + VX`.
    AddToIndex(Register),
    /// `FX0A` — Block until any key is pressed; store the key number in `VX`.
    ///
    /// Implemented by re-executing the instruction (decrementing PC by 2) on
    /// every cycle until the input thread reports a key down.
    WaitForAnyKeyPress(Register),
    /// `FX29` — Set `I` to the address of the built-in 4×5 glyph for `VX & 0xF`.
    ///
    /// The fontset covers hex digits 0–F and is stored at `0x050–0x09F`.
    FontCharacter(Register),
    /// `FX33` — Store the binary-coded decimal representation of `VX` at
    /// `I`, `I+1`, `I+2` (hundreds, tens, units digits respectively).
    BinaryCodedDecimal(Register),
    /// `FX55` — Store registers `V0–VX` in memory starting at `I`.
    ///
    /// **Original:** `I` is incremented after each write (so `I = I + X + 1` after).
    /// **`new_instructions`:** `I` is unchanged; registers are written at `I + reg`.
    StoreMemory(Register),
    /// `FX65` — Load registers `V0–VX` from memory starting at `I`.
    ///
    /// **Original:** `I` is incremented after each read (so `I = I + X + 1` after).
    /// **`new_instructions`:** `I` is unchanged; registers are read from `I + reg`.
    LoadMemory(Register),
    /// An opcode that does not match any known instruction pattern.
    Unknown(u16),
}

/// Decode a raw 16-bit big-endian opcode word into an [`Opcode`] variant.
///
/// CHIP-8 opcodes are two bytes fetched from consecutive memory addresses and
/// combined big-endian. The top nibble selects the instruction family;
/// subsequent nibbles carry register indices, immediates, or addresses.
///
/// Nibble layout for a word `0xABCD`:
/// - `nibble1 = 0xA` — instruction family
/// - `nibble2 = 0xB` — `VX` register index
/// - `nibble3 = 0xC` — `VY` register index (or part of an immediate)
/// - `nibble4 = 0xD` — subtype selector or sprite height `N`
pub fn decode(opcode: u16) -> Opcode {
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
        0x1 => Opcode::Jump(opcode & 0x0FFF),
        0x2 => Opcode::SubroutineCall(opcode & 0x0FFF),
        0x3 => Opcode::SkipEqVal((nibble2 as u8, (opcode & 0x00FF) as u8)),
        0x4 => Opcode::SkipNeqVal((nibble2 as u8, (opcode & 0x00FF) as u8)),
        0x5 => match nibble4 {
            0x0 => Opcode::SkipEqReg((nibble2 as u8, nibble3 as u8)),
            _ => Opcode::Unknown(opcode),
        },
        0x6 => Opcode::SetReg((nibble2 as u8, (opcode & 0x00FF) as u8)),
        0x7 => Opcode::AddToReg((nibble2 as u8, (opcode & 0x00FF) as u8)),
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
            0x0 => Opcode::SkipNeqReg((nibble2 as u8, nibble3 as u8)),
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
            0x0A => Opcode::WaitForAnyKeyPress(nibble2 as u8),
            0x29 => Opcode::FontCharacter(nibble2 as u8),
            0x33 => Opcode::BinaryCodedDecimal(nibble2 as u8),
            0x55 => Opcode::StoreMemory(nibble2 as u8),
            0x65 => Opcode::LoadMemory(nibble2 as u8),
            _ => Opcode::Unknown(opcode),
        },
        _ => Opcode::Unknown(opcode),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_clear_screen() {
        assert!(matches!(decode(0x00E0), Opcode::ClearScreen));
    }

    #[test]
    fn decode_subroutine_return() {
        assert!(matches!(decode(0x00EE), Opcode::SubroutineReturn));
    }

    #[test]
    fn decode_jump() {
        let Opcode::Jump(addr) = decode(0x1ABC) else { panic!() };
        assert_eq!(addr, 0xABC);
    }

    #[test]
    fn decode_subroutine_call() {
        let Opcode::SubroutineCall(addr) = decode(0x2DEF) else { panic!() };
        assert_eq!(addr, 0xDEF);
    }

    #[test]
    fn decode_skip_eq_val() {
        let Opcode::SkipEqVal((reg, val)) = decode(0x3A42) else { panic!() };
        assert_eq!(reg, 0xA);
        assert_eq!(val, 0x42);
    }

    #[test]
    fn decode_skip_neq_val() {
        let Opcode::SkipNeqVal((reg, val)) = decode(0x4B99) else { panic!() };
        assert_eq!(reg, 0xB);
        assert_eq!(val, 0x99);
    }

    #[test]
    fn decode_set_reg() {
        let Opcode::SetReg((reg, val)) = decode(0x6F77) else { panic!() };
        assert_eq!(reg, 0xF);
        assert_eq!(val, 0x77);
    }

    #[test]
    fn decode_set_index() {
        let Opcode::SetIndex(addr) = decode(0xABCD) else { panic!() };
        assert_eq!(addr, 0xBCD);
    }

    #[test]
    fn decode_random() {
        let Opcode::Random(reg, mask) = decode(0xC5FF) else { panic!() };
        assert_eq!(reg, 0x5);
        assert_eq!(mask, 0xFF);
    }

    #[test]
    fn decode_display() {
        let Opcode::Display(rx, ry, n) = decode(0xD123) else { panic!() };
        assert_eq!(rx, 0x1);
        assert_eq!(ry, 0x2);
        assert_eq!(n, 0x3);
    }

    #[test]
    fn decode_skip_if_key_pressed() {
        let Opcode::SkipIfKeyPressed(reg) = decode(0xE29E) else { panic!() };
        assert_eq!(reg, 0x2);
    }

    #[test]
    fn decode_bcd() {
        let Opcode::BinaryCodedDecimal(reg) = decode(0xF333) else { panic!() };
        assert_eq!(reg, 0x3);
    }

    #[test]
    fn decode_unknown_returns_unknown() {
        assert!(matches!(decode(0xFFFF), Opcode::Unknown(_)));
    }
}
