//! EIP-8141 frame transaction instructions.
//!
//! Frame transactions decompose a transaction into frames that validate it,
//! approve gas payment and execute user operations. These instructions let a
//! frame inspect the transaction it belongs to, and let it approve execution
//! and payment.
//!
//! Outside a frame transaction there is no context to report on, so every
//! instruction here halts exceptionally, which is what the spec requires.

use crate::{
    gas,
    instruction_result::InstructionResult,
    instructions::system::copy_cost_and_memory_resize,
    interpreter_action::InterpreterAction,
    interpreter_types::{InterpreterTypes as ITy, LoopControl, MemoryTr, StackTr},
    Host, InstructionContext as Ictx, InstructionExecResult as Result,
};
use context_interface::host::FrameTxContext;
use primitives::{Bytes, U256};

/// `SIGPARAM` parameter selecting the memory-copy form.
const SIGPARAM_COPY: u64 = 0x04;

/// Operands the `SIGPARAM` copy form consumes: index, param, memOffset,
/// dataOffset, length.
const SIGPARAM_COPY_STACK: usize = 5;

/// Signature scheme id for `ARBITRARY` entries.
const SCHEME_ARBITRARY: u8 = 0x00;

/// Implements the APPROVE instruction (0xaa).
///
/// Exits the current frame successfully like RETURN, and updates the
/// transaction-scoped approval context. The memory region `[offset, offset+len)`
/// becomes the frame's return data, and only memory expansion is charged.
pub fn approve<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    if context.host.frame_context().is_none() {
        return Err(InstructionResult::InvalidFEOpcode);
    }
    popn!([offset, len, scope], context.interpreter);
    let scope = u64::try_from(scope).unwrap_or(u64::MAX);
    let len = as_usize_or_fail!(context.interpreter, len);

    let mut output = Bytes::default();
    if len != 0 {
        let offset = as_usize_or_fail!(context.interpreter, offset);
        context
            .interpreter
            .resize_memory(context.host.gas_params(), offset, len)?;
        output = context
            .interpreter
            .memory
            .slice_len(offset, len)
            .to_vec()
            .into();
    }
    // A host that does not model frame transactions rejects, so APPROVE can
    // never silently succeed where there is nothing to approve.
    if !context.host.frame_approve(scope) {
        return Err(InstructionResult::Revert);
    }
    context
        .interpreter
        .bytecode
        .set_action(InterpreterAction::new_return(
            InstructionResult::Return,
            output,
            context.interpreter.gas,
        ));
    Err(InstructionResult::Return)
}

/// Implements the TXPARAM instruction (0xb0).
///
/// Reads a transaction-scoped parameter. An undefined parameter halts.
pub fn txparam<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    gas!(context.interpreter, gas::BASE);
    let Some(frame) = context.host.frame_context().cloned() else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    popn_top!([], param, context.interpreter);
    // An out-of-range parameter is undefined, not a truncated small one.
    let Ok(selector) = u64::try_from(*param) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    *param = match selector {
        0x00 => U256::from(0x06u8), // frame transaction type
        0x01 => U256::from(frame.nonce),
        0x02 => frame.sender.into_word().into(),
        0x03 => frame.max_priority_fee_per_gas,
        0x04 => frame.max_fee_per_gas,
        0x05 => frame.max_fee_per_blob_gas,
        0x06 => frame.max_cost,
        0x07 => U256::from(frame.blob_count),
        0x08 => frame.sig_hash.into(),
        0x09 => U256::from(frame.frames.len() as u64),
        0x0A => U256::from(frame.frame_index),
        0x0B => U256::from(frame.signatures.len() as u64),
        _ => return Err(InstructionResult::InvalidFEOpcode),
    };
    Ok(())
}

/// Implements the FRAMEDATALOAD instruction (0xb1).
///
/// Loads a 32-byte word from the chosen frame's calldata, zero-extended past the
/// end of the data.
pub fn framedataload<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    gas!(context.interpreter, gas::VERYLOW);
    let Some(frame) = context.host.frame_context().cloned() else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    popn_top!([offset], frame_index, context.interpreter);
    let Some(data) = frame_data(&frame, frame_index) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    let mut word = [0u8; 32];
    if let Ok(off) = usize::try_from(offset) {
        if off < data.len() {
            let n = 32.min(data.len() - off);
            word[..n].copy_from_slice(&data[off..off + n]);
        }
    }
    *frame_index = U256::from_be_bytes(word);
    Ok(())
}

/// Implements the FRAMEDATACOPY instruction (0xb2).
///
/// Copies from the chosen frame's calldata into memory, zero-extending beyond
/// the end of the data. Priced exactly as CALLDATACOPY.
pub fn framedatacopy<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    let Some(frame) = context.host.frame_context().cloned() else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    popn!(
        [memory_offset, data_offset, len, frame_index],
        context.interpreter
    );
    let Some(data) = frame_data(&frame, &frame_index) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    let len = as_usize_or_fail!(context.interpreter, len);
    let Some(memory_offset) = copy_cost_and_memory_resize(
        context.interpreter,
        context.host.gas_params(),
        memory_offset,
        len,
    )?
    else {
        return Ok(());
    };
    let data_offset = as_usize_saturated!(data_offset);
    context
        .interpreter
        .memory
        .set_data(memory_offset, data_offset, len, &data);
    Ok(())
}

/// Implements the FRAMEPARAM instruction (0xb3).
///
/// Reads a frame-scoped parameter. The status of the current or a later frame
/// does not exist yet and halts.
pub fn frameparam<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    gas!(context.interpreter, gas::BASE);
    let Some(frame) = context.host.frame_context().cloned() else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    // frameIndex is on top, param second; the result replaces param.
    popn_top!([frame_index], param_slot, context.interpreter);
    let Ok(index) = u64::try_from(frame_index) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    let Some(info) = frame.frames.get(index as usize) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    let Ok(selector) = u64::try_from(*param_slot) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    *param_slot = match selector {
        0x00 => info.resolved_target.into_word().into(),
        0x01 => U256::from(info.gas_limit),
        0x02 => U256::from(info.mode),
        0x03 => U256::from(info.flags),
        0x04 => U256::from(info.data.len() as u64),
        0x05 => {
            // The status of the current or a later frame does not exist yet.
            if index >= frame.frame_index {
                return Err(InstructionResult::InvalidFEOpcode);
            }
            U256::from(info.status)
        }
        0x06 => U256::from(info.flags & 0x03),
        0x07 => U256::from((info.flags >> 2) & 0x01),
        0x08 => info.value,
        _ => return Err(InstructionResult::InvalidFEOpcode),
    };
    Ok(())
}

/// Implements the SIGPARAM instruction (0xb4).
///
/// Reads signature-scoped metadata, or copies the raw bytes of an `ARBITRARY`
/// entry into memory.
///
/// The copy form (`param == 0x04`) consumes five operands rather than two. The
/// opcode table declares a fixed arity and cannot express that, so the deeper
/// requirement is checked here -- without this, the copy form would read below
/// the stack.
pub fn sigparam<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    let Some(frame) = context.host.frame_context().cloned() else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    // Inspect the param before consuming anything: its value decides how many
    // operands this instruction takes.
    let stack_len = context.interpreter.stack.len();
    if stack_len < 2 {
        return Err(InstructionResult::StackUnderflow);
    }
    let param_val = context.interpreter.stack.data()[stack_len - 2];
    let is_copy = u64::try_from(param_val)
        .map(|p| p == SIGPARAM_COPY)
        .unwrap_or(false);
    if is_copy && stack_len < SIGPARAM_COPY_STACK {
        return Err(InstructionResult::StackUnderflow);
    }

    if !is_copy {
        gas!(context.interpreter, gas::BASE);
        popn_top!([sig_index], param_slot, context.interpreter);
        let Ok(index) = u64::try_from(sig_index) else {
            return Err(InstructionResult::InvalidFEOpcode);
        };
        let Some(sig) = frame.signatures.get(index as usize) else {
            return Err(InstructionResult::InvalidFEOpcode);
        };
        let Ok(selector) = u64::try_from(*param_slot) else {
            return Err(InstructionResult::InvalidFEOpcode);
        };
        *param_slot = match selector {
            0x00 => match sig.resolved_signer {
                // ARBITRARY entries have no protocol-assigned signer.
                None => return Err(InstructionResult::InvalidFEOpcode),
                Some(addr) => addr.into_word().into(),
            },
            0x01 => U256::from(sig.scheme),
            0x02 => U256::from_be_bytes(sig.msg.0),
            0x03 => U256::from(sig.signature.len() as u64),
            _ => return Err(InstructionResult::InvalidFEOpcode),
        };
        return Ok(());
    }

    // Copy form. Only ARBITRARY signature bytes are introspectable; the others
    // are withheld so they remain aggregatable under future schemes.
    popn!(
        [sig_index, _param, memory_offset, data_offset, len],
        context.interpreter
    );
    let Ok(index) = u64::try_from(sig_index) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    let Some(sig) = frame.signatures.get(index as usize) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    if sig.scheme != SCHEME_ARBITRARY {
        return Err(InstructionResult::InvalidFEOpcode);
    }
    let len = as_usize_or_fail!(context.interpreter, len);
    let Some(memory_offset) = copy_cost_and_memory_resize(
        context.interpreter,
        context.host.gas_params(),
        memory_offset,
        len,
    )?
    else {
        return Ok(());
    };
    let data_offset = as_usize_saturated!(data_offset);
    context
        .interpreter
        .memory
        .set_data(memory_offset, data_offset, len, &sig.signature);
    Ok(())
}

/// Resolves a frame index operand to that frame's calldata.
fn frame_data(frame: &FrameTxContext, index: &U256) -> Option<Bytes> {
    let index = u64::try_from(*index).ok()?;
    frame.frames.get(index as usize).map(|f| f.data.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{host::DummyHost, InstructionContext as Ictx, Interpreter};
    use context_interface::host::{FrameInfo, FrameSigInfo};
    use primitives::{address, hardfork::SpecId, Address, B256};

    fn ctx() -> FrameTxContext {
        FrameTxContext {
            sender: address!("1111111111111111111111111111111111111111"),
            nonce: 7,
            sig_hash: B256::repeat_byte(0xAB),
            max_cost: U256::from(1234u64),
            frame_index: 1,
            frames: vec![
                FrameInfo {
                    resolved_target: address!("2222222222222222222222222222222222222222"),
                    gas_limit: 50_000,
                    mode: 1,
                    flags: 0x3,
                    status: 1,
                    data: Bytes::from_static(&[0xde, 0xad]),
                    ..Default::default()
                },
                FrameInfo {
                    resolved_target: address!("3333333333333333333333333333333333333333"),
                    gas_limit: 100_000,
                    mode: 2,
                    ..Default::default()
                },
            ],
            signatures: vec![FrameSigInfo {
                resolved_signer: Some(address!("4444444444444444444444444444444444444444")),
                scheme: 1,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// Outside a frame transaction the instructions must halt: there is no
    /// context to report on, and silently returning zero would be worse.
    #[test]
    fn halts_without_frame_context() {
        let mut interpreter = Interpreter::default();
        let mut host = DummyHost::new(SpecId::default());
        let _ = interpreter.stack.push(U256::ZERO);
        let res = txparam(Ictx {
            interpreter: &mut interpreter,
            host: &mut host,
        });
        assert_eq!(res, Err(InstructionResult::InvalidFEOpcode));
    }

    #[test]
    fn txparam_reads_transaction_scope() {
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(ctx(), 0x3);
        for (param, want) in [
            (0x00u64, U256::from(0x06u8)),
            (0x01, U256::from(7u64)),
            (0x07, U256::ZERO),
            (0x09, U256::from(2u64)), // frame count
            (0x0A, U256::from(1u64)), // current frame index
            (0x0B, U256::from(1u64)), // signature count
        ] {
            let mut interpreter = Interpreter::default();
            let _ = interpreter.stack.push(U256::from(param));
            txparam(Ictx {
                interpreter: &mut interpreter,
                host: &mut host,
            })
            .unwrap();
            assert_eq!(interpreter.stack.data()[0], want, "TXPARAM({param:#x})");
        }
        // An undefined parameter halts rather than returning zero.
        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::from(0xFFu64));
        assert_eq!(
            txparam(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::InvalidFEOpcode)
        );
    }

    /// A frame index beyond 64 bits must halt, not truncate to a valid frame.
    #[test]
    fn frameparam_rejects_out_of_range_index() {
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(ctx(), 0x3);
        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::ZERO); // param
        let _ = interpreter.stack.push(U256::from(1u64) << 64); // frameIndex, out of range
        assert_eq!(
            frameparam(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::InvalidFEOpcode)
        );
    }

    /// The status of the current or a later frame does not exist yet.
    #[test]
    fn frameparam_status_of_current_frame_halts() {
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(ctx(), 0x3);
        // Frame 0 has already run (frame_index is 1), so its status is readable.
        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::from(0x05u64));
        let _ = interpreter.stack.push(U256::ZERO);
        frameparam(Ictx {
            interpreter: &mut interpreter,
            host: &mut host,
        })
        .unwrap();
        assert_eq!(interpreter.stack.data()[0], U256::from(1u64));
        // Frame 1 is the current frame: halt.
        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::from(0x05u64));
        let _ = interpreter.stack.push(U256::from(1u64));
        assert_eq!(
            frameparam(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::InvalidFEOpcode)
        );
    }

    /// The SIGPARAM copy form takes five operands, which the opcode table cannot
    /// declare. Supplying only the two the metadata form needs must underflow
    /// cleanly rather than read below the stack.
    #[test]
    fn sigparam_copy_form_checks_its_own_depth() {
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(ctx(), 0x3);
        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::from(SIGPARAM_COPY)); // param = copy form
        let _ = interpreter.stack.push(U256::ZERO); // signatureIndex
        assert_eq!(
            sigparam(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::StackUnderflow)
        );
    }

    #[test]
    fn sigparam_reads_metadata() {
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(ctx(), 0x3);
        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::ZERO); // param 0x00 = resolved_signer
        let _ = interpreter.stack.push(U256::ZERO); // signatureIndex
        sigparam(Ictx {
            interpreter: &mut interpreter,
            host: &mut host,
        })
        .unwrap();
        let want: U256 = address!("4444444444444444444444444444444444444444")
            .into_word()
            .into();
        assert_eq!(interpreter.stack.data()[0], want);
    }

    /// APPROVE must reject a scope the host does not permit, and terminate on one
    /// it does.
    #[test]
    fn approve_respects_permitted_scope() {
        // Host permits payment only (0x1); asking for 0x3 must revert.
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(ctx(), 0x1);
        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::from(3u64)); // scope
        let _ = interpreter.stack.push(U256::ZERO); // length
        let _ = interpreter.stack.push(U256::ZERO); // offset
        assert_eq!(
            approve(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::Revert)
        );

        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::from(1u64));
        let _ = interpreter.stack.push(U256::ZERO);
        let _ = interpreter.stack.push(U256::ZERO);
        assert_eq!(
            approve(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::Return)
        );
    }
}
