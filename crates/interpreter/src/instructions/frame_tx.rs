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
    instructions::{system::copy_cost_and_memory_resize, utility::IntoAddress},
    interpreter_action::InterpreterAction,
    interpreter_types::{InputsTr, InterpreterTypes as ITy, LoopControl, MemoryTr, StackTr},
    Host, InstructionContext as Ictx, InstructionExecResult as Result,
};
use context_interface::{
    host::{FrameInfo, FrameTxContext, FrameTxTrace},
    transaction::TransactionType,
    Transaction,
};
use primitives::{Address, Bytes, TxKind, B256, U256};
use std::sync::Arc;

/// Signature scheme id for `ARBITRARY` entries.
const SCHEME_ARBITRARY: u8 = 0x00;

/// EIP-7906 frame mode in which transaction outcome introspection is valid.
const FRAME_MODE_POST_TX: u8 = 0x03;

/// EIP-8141 frame mode in which validation executes without state changes.
const FRAME_MODE_VERIFY: u8 = 0x01;

/// Classification latched by the handler for one fully matched synthetic call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameTxCallMatch {
    /// Declared outer sender whose warmth was established at lifecycle start.
    pub sender: Address,
    /// Whether the matched top-level frame executes statically.
    pub is_static: bool,
}

/// Provisional flat gas for TXTRACE and transaction-local TXDIFF queries.
///
/// EIP-7906 leaves this value TBD. The combined Hegota experiment assigns 100
/// until the proposal finalizes its gas schedule.
pub(crate) const PROVISIONAL_TXTRACE_GAS: u16 = 100;

/// A process-local EIP-8141 frame transaction context.
///
/// Frame transactions are a draft, and wiring their context through `TxEnv` or
/// the `Transaction` trait would change types the whole ecosystem constructs --
/// downstream crates build `TxEnv` with struct literals, and adding a trait
/// method forces lifetime bounds on every implementor. Both break crates that
/// have nothing to do with EIP-8141.
///
/// So tooling installs the context here instead. It is scoped to the current
/// thread, which matches how test runners execute: one transaction at a time per
/// thread. A host that models frame transactions natively should override
/// [`Host::frame_context`] instead and ignore this entirely; the instructions
/// prefer the host and fall back to this slot.
#[cfg(feature = "std")]
mod slot {
    use super::FrameTxContext;
    use std::{cell::RefCell, marker::PhantomData, rc::Rc, sync::Arc};

    thread_local! {
        static FRAME_TX: RefCell<Option<Arc<FrameTxContext>>> = const { RefCell::new(None) };
    }

    /// Installs a frame transaction context for the current thread.
    pub fn set(context: Option<FrameTxContext>) {
        FRAME_TX.with(|slot| {
            *slot.borrow_mut() = context.map(FrameTxContext::into_shared);
        });
    }

    /// Returns the current thread's frame transaction context, if any.
    pub fn get() -> Option<Arc<FrameTxContext>> {
        FRAME_TX.with(|slot| slot.borrow().clone())
    }

    /// Guard returned by [`install`]. Restores the prior thread-local context
    /// when dropped, including during unwinding.
    #[derive(Debug)]
    #[must_use = "dropping the guard immediately restores the previous frame context"]
    pub struct FrameTxContextGuard {
        previous: Option<Arc<FrameTxContext>>,
        not_send: PhantomData<Rc<()>>,
    }

    impl Drop for FrameTxContextGuard {
        fn drop(&mut self) {
            FRAME_TX.with(|slot| {
                *slot.borrow_mut() = self.previous.take();
            });
        }
    }

    /// Installs a frame transaction context until the returned guard is
    /// dropped. Installations may be nested and restore in LIFO order.
    pub fn install(context: FrameTxContext) -> FrameTxContextGuard {
        let context = Some(context.into_shared());
        let previous = FRAME_TX.with(|slot| slot.replace(context));
        FrameTxContextGuard {
            previous,
            not_send: PhantomData,
        }
    }
}

#[cfg(feature = "std")]
pub use slot::{
    get as frame_tx_context, install as install_frame_tx_context, set as set_frame_tx_context,
    FrameTxContextGuard,
};

/// Returns whether a call to `target` must execute statically for the active
/// frame transaction context.
///
/// Only a call to the current frame's resolved target is affected, and only in
/// VERIFY or POST_TX mode. Native host contexts work in both `std` and
/// `no_std`; the thread-local tooling context is only a `std` fallback.
pub fn frame_tx_call_requires_static<H: Host + ?Sized>(host: &H, target: Address) -> bool {
    active_context(host)
        .and_then(|context| {
            current_frame(&context).map(|frame| {
                frame.resolved_target == target
                    && matches!(frame.mode, FRAME_MODE_VERIFY | FRAME_MODE_POST_TX)
            })
        })
        .unwrap_or(false)
}

/// Returns whether a synthetic call exactly matches the valid current frame.
///
/// This is used by the mainnet handler to distinguish frame execution from an
/// unrelated transaction while a tooling context happens to be installed.
pub fn frame_tx_call_matches_current_frame<H: Host + ?Sized, TX: Transaction + ?Sized>(
    host: &H,
    tx: &TX,
) -> bool {
    frame_tx_call_match(host, tx).is_some()
}

/// Fully binds a synthetic transaction to the current frame and transaction
/// context, returning the top-level execution mode to latch on success.
pub fn frame_tx_call_match<H: Host + ?Sized, TX: Transaction + ?Sized>(
    host: &H,
    tx: &TX,
) -> Option<FrameTxCallMatch> {
    let TxKind::Call(target) = tx.kind() else {
        return None;
    };
    let context = active_context(host)?;
    let frame = current_frame(&context)?;
    let expected_type = if context.blob_count == 0 {
        TransactionType::Eip1559
    } else {
        TransactionType::Eip4844
    };
    let has_access_list = tx
        .access_list()
        .is_some_and(|mut items| items.next().is_some());
    let blob_count = u64::try_from(tx.blob_versioned_hashes().len()).ok()?;
    if frame.mode > FRAME_MODE_POST_TX
        || frame.resolved_target != target
        || frame.expected_caller != tx.caller()
        || frame.data != *tx.input()
        || frame.value != tx.value()
        || frame.gas_limit != tx.gas_limit()
        || TransactionType::from(tx.tx_type()) != expected_type
        || has_access_list
        || tx.authorization_list_len() != 0
        || U256::from(tx.max_fee_per_gas()) != context.max_fee_per_gas
        || tx.max_priority_fee_per_gas().map(U256::from) != Some(context.max_priority_fee_per_gas)
        || U256::from(tx.max_fee_per_blob_gas()) != context.max_fee_per_blob_gas
        || blob_count != context.blob_count
    {
        return None;
    }

    Some(FrameTxCallMatch {
        sender: context.sender,
        is_static: matches!(frame.mode, FRAME_MODE_VERIFY | FRAME_MODE_POST_TX),
    })
}

/// Resolves the active frame transaction context: the host first, then the
/// thread-local slot that tooling installs.
fn active_context<H: Host + ?Sized>(host: &H) -> Option<Arc<FrameTxContext>> {
    active_context_with_source(host).map(|(context, _)| context)
}

/// Returns the active context and whether it came from the host itself.
fn active_context_with_source<H: Host + ?Sized>(host: &H) -> Option<(Arc<FrameTxContext>, bool)> {
    if let Some(ctx) = host.frame_context() {
        return Some((ctx, true));
    }
    #[cfg(feature = "std")]
    {
        frame_tx_context().map(|context| (context, false))
    }
    #[cfg(not(feature = "std"))]
    {
        None
    }
}

/// Implements the APPROVE instruction (0xaa).
///
/// Exits the current frame successfully like RETURN, and updates the
/// transaction-scoped approval context. The memory region `[offset, offset+len)`
/// becomes the frame's return data, and only memory expansion is charged.
pub fn approve<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    let Some((frame, host_models_frames)) = active_context_with_source(context.host) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    let Some(current_frame) = current_frame(&frame) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    if current_frame.mode >= FRAME_MODE_POST_TX {
        return Err(InstructionResult::InvalidFEOpcode);
    }
    if context.interpreter.input.target_address() != current_frame.resolved_target {
        return Err(InstructionResult::Revert);
    }
    let allowed_scopes = u64::from(current_frame.flags & 0x03) & frame.approvable_scopes & 0x03;
    popn!([offset, len, scope], context.interpreter);
    let Ok(scope) = u64::try_from(scope) else {
        return Err(InstructionResult::Revert);
    };
    if scope == 0 || scope & !allowed_scopes != 0 {
        return Err(InstructionResult::Revert);
    }
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
    // A native host may impose additional approval policy after the frame and
    // transaction context masks have both been enforced above.
    let approved = if host_models_frames {
        context.host.frame_approve(scope)
    } else {
        true
    };
    if !approved {
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
    let Some(frame) = active_context(context.host) else {
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
        0x0C => U256::from(frame.legacy_nonce),
        0x0D => U256::from(frame.nonce_keys.len() as u64),
        0x0E => U256::from_be_bytes(frame.nonce_keys_hash.0),
        0x0F => U256::from(frame.recent_root_references.len() as u64),
        0x10 => frame
            .nonce_keys
            .first()
            .copied()
            .ok_or(InstructionResult::InvalidFEOpcode)?,
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
    let Some(frame) = active_context(context.host) else {
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
    let Some(frame) = active_context(context.host) else {
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
        .set_data(memory_offset, data_offset, len, data);
    Ok(())
}

/// Implements the FRAMEPARAM instruction (0xb3).
///
/// Reads a frame-scoped parameter. The status of the current or a later frame
/// does not exist yet and halts.
pub fn frameparam<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    gas!(context.interpreter, gas::BASE);
    let Some(frame) = active_context(context.host) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    // frameIndex is on top, param second; the result replaces param.
    popn_top!([frame_index], param_slot, context.interpreter);
    let Ok(index) = usize::try_from(frame_index) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    let Some(info) = frame.frames.get(index) else {
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
            if frame_index >= U256::from(frame.frame_index) {
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
/// Reads signature-scoped metadata. An undefined parameter halts.
pub fn sigparam<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    let Some(frame) = active_context(context.host) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    gas!(context.interpreter, gas::BASE);
    popn_top!([sig_index], param_slot, context.interpreter);
    let Ok(index) = usize::try_from(sig_index) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    let Some(sig) = frame.signatures.get(index) else {
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
    Ok(())
}

/// Implements the SIGDATACOPY instruction (0xb5).
///
/// Copies raw bytes from an `ARBITRARY` signature into memory, zero-extending
/// beyond the end of the data. Priced exactly as CALLDATACOPY.
pub fn sigdatacopy<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    let Some(frame) = active_context(context.host) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    popn!(
        [memory_offset, data_offset, len, sig_index],
        context.interpreter
    );
    let Ok(index) = usize::try_from(sig_index) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    let Some(sig) = frame.signatures.get(index) else {
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

/// Implements the RECENTROOTREFLOAD instruction (0xb6).
///
/// Loads one field from a verified recent-root reference. The field operand is
/// on top of the stack and the reference index is beneath it.
pub fn recentrootrefload<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    let Some(frame) = active_context(context.host) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    popn_top!([field], index_slot, context.interpreter);
    let index = operand_index(*index_slot)?;
    let Some(reference) = frame.recent_root_references.get(index) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    let Ok(field) = u64::try_from(field) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    *index_slot = match field {
        0x00 => hash_word(reference.source_id),
        0x01 => U256::from(reference.slot),
        0x02 => hash_word(reference.root),
        _ => return Err(InstructionResult::InvalidFEOpcode),
    };
    Ok(())
}

/// Implements the provisional TXTRACE instruction (0xb7).
///
/// Reads the precomputed EIP-7906 transaction trace. This opcode is valid only
/// in the current transaction's POST_TX frame.
pub fn txtrace<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    let frame = active_post_tx_context(context.host)?;
    popn_top!([in2], param_slot, context.interpreter);
    let Ok(param) = u64::try_from(*param_slot) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    let trace = &frame.trace;
    *param_slot = match param {
        0x00 => {
            require_zero(in2)?;
            U256::from(trace.balance_diffs.len())
        }
        0x01 => {
            require_zero(in2)?;
            U256::from(trace.storage_diffs.len())
        }
        0x02 => {
            require_zero(in2)?;
            U256::from(trace.deployed_contracts.len())
        }
        0x03..=0x05 => {
            let Some(diff) = trace.balance_diffs.get(operand_index(in2)?) else {
                return Err(InstructionResult::InvalidFEOpcode);
            };
            match param {
                0x03 => diff.address.into_word().into(),
                0x04 => diff.before,
                0x05 => diff.after,
                _ => unreachable!(),
            }
        }
        0x06..=0x09 => {
            let Some(diff) = trace.storage_diffs.get(operand_index(in2)?) else {
                return Err(InstructionResult::InvalidFEOpcode);
            };
            match param {
                0x06 => diff.address.into_word().into(),
                0x07 => diff.key,
                0x08 => diff.before,
                0x09 => diff.after,
                _ => unreachable!(),
            }
        }
        0x0A..=0x0B => {
            let Some(deployment) = trace.deployed_contracts.get(operand_index(in2)?) else {
                return Err(InstructionResult::InvalidFEOpcode);
            };
            if param == 0x0A {
                deployment.address.into_word().into()
            } else {
                hash_word(deployment.code_hash)
            }
        }
        0x0C => {
            require_zero(in2)?;
            U256::from(
                frame
                    .event_snapshot()
                    .ok_or(InstructionResult::InvalidFEOpcode)?
                    .len(),
            )
        }
        0x0D..=0x13 => {
            let Some(event) = frame
                .event_snapshot()
                .ok_or(InstructionResult::InvalidFEOpcode)?
                .get(operand_index(in2)?)
            else {
                return Err(InstructionResult::InvalidFEOpcode);
            };
            match param {
                0x0D => event.address.into_word().into(),
                0x0E => U256::from(event.topics.len()),
                0x0F..=0x12 => {
                    let topic_index = (param - 0x0F) as usize;
                    let Some(topic) = event.topics.get(topic_index) else {
                        return Err(InstructionResult::InvalidFEOpcode);
                    };
                    hash_word(*topic)
                }
                0x13 => U256::from(event.data.len()),
                _ => unreachable!(),
            }
        }
        0x14 => {
            require_zero(in2)?;
            trace.gas_pre_charge
        }
        0x15 => {
            require_zero(in2)?;
            trace.gas_payer.into_word().into()
        }
        _ => return Err(InstructionResult::InvalidFEOpcode),
    };
    Ok(())
}

/// Implements the provisional TXDIFF instruction (0xb8).
///
/// Directly queries one address or storage key from the transaction-local diff.
/// Unrecorded balance and code-hash queries fall back to live host state with
/// EIP-2929 warm/cold charging.
pub fn txdiff<IT: ITy, H: Host + ?Sized>(mut context: Ictx<'_, H, IT>) -> Result {
    let frame = active_post_tx_context(context.host)?;
    popn!([param, address_word, in3], context.interpreter);
    let Ok(param) = u64::try_from(param) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    let address = address_word.into_address();
    let trace = &frame.trace;

    let output = match param {
        0x00..=0x01 => {
            let key = in3;
            let live = txdiff_live_storage(&mut context, address, key)?;
            if let Some(index) = storage_diff_index(trace, address, key) {
                let diff = &trace.storage_diffs[index];
                if param == 0x00 {
                    diff.before
                } else {
                    diff.after
                }
            } else {
                live
            }
        }
        0x02..=0x03 => {
            require_zero(in3)?;
            let live = txdiff_live_account(&mut context, address)?;
            if let Some(index) = balance_diff_index(trace, address) {
                let diff = &trace.balance_diffs[index];
                if param == 0x02 {
                    diff.before
                } else {
                    diff.after
                }
            } else {
                live.0
            }
        }
        0x04..=0x05 => {
            require_zero(in3)?;
            let live = txdiff_live_account(&mut context, address)?;
            if let Some(index) = account_diff_index(trace, address) {
                let diff = &trace.account_diffs[index];
                hash_word(if param == 0x04 {
                    diff.code_hash_before
                } else {
                    diff.code_hash_after
                })
            } else {
                hash_word(live.1)
            }
        }
        0x06 => {
            require_zero(in3)?;
            U256::from(storage_diff_range(trace, address).len())
        }
        0x07 => {
            let local_index = operand_index(in3)?;
            let range = storage_diff_range(trace, address);
            let Some(global_index) = range.start.checked_add(local_index) else {
                return Err(InstructionResult::InvalidFEOpcode);
            };
            if global_index >= range.end {
                return Err(InstructionResult::InvalidFEOpcode);
            }
            U256::from(global_index)
        }
        0x08 => {
            require_zero(in3)?;
            U256::from(
                frame
                    .event_count_for_address(address)
                    .ok_or(InstructionResult::InvalidFEOpcode)?,
            )
        }
        0x09 => {
            let local_index = operand_index(in3)?;
            let global_index = frame
                .event_global_index_for_address(address, local_index)
                .ok_or(InstructionResult::InvalidFEOpcode)?;
            U256::from(global_index)
        }
        0x0A => {
            require_zero(in3)?;
            let account =
                account_diff_index(trace, address).map(|index| &trace.account_diffs[index]);
            let mut flags = 0u8;
            if account.is_some_and(|diff| diff.nonce_changed) {
                flags |= 0b0001;
            }
            if balance_diff_index(trace, address)
                .map(|index| &trace.balance_diffs[index])
                .is_some_and(|diff| diff.before != diff.after)
            {
                flags |= 0b0010;
            }
            if trace.storage_diffs[storage_diff_range(trace, address)]
                .iter()
                .any(|diff| diff.before != diff.after)
            {
                flags |= 0b0100;
            }
            if account.is_some_and(|diff| diff.code_hash_before != diff.code_hash_after) {
                flags |= 0b1000;
            }
            U256::from(flags)
        }
        _ => return Err(InstructionResult::InvalidFEOpcode),
    };
    push!(context.interpreter, output);
    Ok(())
}

/// Implements the EVENTDATACOPY instruction (0xb9).
///
/// Copies non-indexed event data with strict source bounds. The event index is
/// on top of the stack, followed by memory offset, data offset, and length.
pub fn eventdatacopy<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    let frame = active_post_tx_context(context.host)?;
    popn!(
        [event_index, memory_offset, data_offset, len],
        context.interpreter
    );
    let Some(event) = frame
        .event_snapshot()
        .ok_or(InstructionResult::InvalidFEOpcode)?
        .get(operand_index(event_index)?)
    else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    let len = as_usize_or_fail!(context.interpreter, len);
    let data_offset = as_usize_saturated!(data_offset);
    if data_offset.saturating_add(len) > event.data.len() {
        return Err(InstructionResult::OutOfOffset);
    }
    let Some(memory_offset) = copy_cost_and_memory_resize(
        context.interpreter,
        context.host.gas_params(),
        memory_offset,
        len,
    )?
    else {
        return Ok(());
    };
    context
        .interpreter
        .memory
        .set_data(memory_offset, data_offset, len, &event.data);
    Ok(())
}

fn active_post_tx_context<H: Host + ?Sized>(host: &H) -> Result<Arc<FrameTxContext>> {
    let Some(frame) = active_context(host) else {
        return Err(InstructionResult::InvalidFEOpcode);
    };
    if current_frame_mode(&frame) != Some(FRAME_MODE_POST_TX) {
        return Err(InstructionResult::InvalidFEOpcode);
    }
    Ok(frame)
}

fn current_frame_mode(frame: &FrameTxContext) -> Option<u8> {
    current_frame(frame).map(|info| info.mode)
}

fn current_frame(frame: &FrameTxContext) -> Option<&FrameInfo> {
    frame.frames.get(usize::try_from(frame.frame_index).ok()?)
}

fn txdiff_live_account<IT: ITy, H: Host + ?Sized>(
    context: &mut Ictx<'_, H, IT>,
    address: Address,
) -> Result<(U256, B256)> {
    let cold_cost = context.host.gas_params().cold_account_additional_cost();
    let skip_cold = context.interpreter.gas.remaining() < cold_cost;
    let account = context
        .host
        .load_account_info_skip_cold_load(address, false, skip_cold)?;
    if account.is_cold {
        gas!(context.interpreter, cold_cost);
    }
    Ok((account.balance, account.code_hash))
}

fn txdiff_live_storage<IT: ITy, H: Host + ?Sized>(
    context: &mut Ictx<'_, H, IT>,
    address: Address,
    key: U256,
) -> Result<U256> {
    let cold_cost = context.host.gas_params().cold_storage_additional_cost();
    let skip_cold = context.interpreter.gas.remaining() < cold_cost;
    let storage = context.host.sload_skip_cold_load(address, key, skip_cold)?;
    if storage.is_cold {
        gas!(context.interpreter, cold_cost);
    }
    Ok(storage.data)
}

fn balance_diff_index(trace: &FrameTxTrace, address: Address) -> Option<usize> {
    trace
        .balance_diffs
        .binary_search_by_key(&address, |diff| diff.address)
        .ok()
}

fn account_diff_index(trace: &FrameTxTrace, address: Address) -> Option<usize> {
    trace
        .account_diffs
        .binary_search_by_key(&address, |diff| diff.address)
        .ok()
}

fn storage_diff_index(trace: &FrameTxTrace, address: Address, key: U256) -> Option<usize> {
    trace
        .storage_diffs
        .binary_search_by(|diff| diff.address.cmp(&address).then_with(|| diff.key.cmp(&key)))
        .ok()
}

fn storage_diff_range(trace: &FrameTxTrace, address: Address) -> core::ops::Range<usize> {
    let start = trace
        .storage_diffs
        .partition_point(|diff| diff.address < address);
    let end = trace
        .storage_diffs
        .partition_point(|diff| diff.address <= address);
    start..end
}

fn operand_index(value: U256) -> Result<usize> {
    usize::try_from(value).map_err(|_| InstructionResult::InvalidFEOpcode)
}

fn require_zero(value: U256) -> Result {
    if value.is_zero() {
        Ok(())
    } else {
        Err(InstructionResult::InvalidFEOpcode)
    }
}

const fn hash_word(hash: B256) -> U256 {
    U256::from_be_bytes(hash.0)
}

/// Resolves a frame index operand to that frame's calldata.
fn frame_data<'a>(frame: &'a FrameTxContext, index: &U256) -> Option<&'a Bytes> {
    let index = usize::try_from(*index).ok()?;
    frame.frames.get(index).map(|f| &f.data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gas_table, host::DummyHost, instruction_table, interpreter::EthInterpreter,
        InstructionContext as Ictx, Interpreter,
    };
    use bytecode::{
        opcode::{EVENTDATACOPY, SIGDATACOPY, TXDIFF, TXTRACE},
        Bytecode,
    };
    use context_interface::host::{
        FrameInfo, FrameSigInfo, FrameTxAccountDiff, FrameTxBalanceDiff, FrameTxDeployedContract,
        FrameTxEvent, FrameTxRecentRootReference, FrameTxStorageDiff, FrameTxTrace,
    };
    use primitives::{address, hardfork::SpecId, B256, KECCAK_EMPTY};
    use state::AccountInfo;

    fn ctx() -> FrameTxContext {
        FrameTxContext {
            sender: address!("1111111111111111111111111111111111111111"),
            nonce: 7,
            legacy_nonce: 9,
            nonce_keys: vec![U256::from(3u64), U256::from(7u64)],
            nonce_keys_hash: B256::repeat_byte(0xE1),
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
                msg: B256::repeat_byte(0xCD),
                signature: Bytes::from_static(&[0xAA, 0xBB]),
            }],
            recent_root_references: vec![FrameTxRecentRootReference {
                source_id: B256::repeat_byte(0xF1),
                slot: 123,
                root: B256::repeat_byte(0xF2),
            }],
            ..Default::default()
        }
    }

    fn ctx_with_arbitrary_signature() -> FrameTxContext {
        let mut context = ctx();
        context.signatures.push(FrameSigInfo {
            scheme: SCHEME_ARBITRARY,
            signature: Bytes::from_static(&[0x11, 0x22, 0x33]),
            ..Default::default()
        });
        context
    }

    fn post_tx_ctx() -> FrameTxContext {
        let address_a = address!("1000000000000000000000000000000000000000");
        let address_b = address!("2000000000000000000000000000000000000000");
        let deployed_hash = B256::repeat_byte(0xD1);
        let mut context = ctx();
        context.frames[context.frame_index as usize].mode = FRAME_MODE_POST_TX;
        context.trace = FrameTxTrace {
            balance_diffs: vec![
                FrameTxBalanceDiff {
                    address: address_a,
                    before: U256::from(10u64),
                    after: U256::from(7u64),
                },
                FrameTxBalanceDiff {
                    address: address_b,
                    before: U256::from(20u64),
                    after: U256::from(25u64),
                },
            ],
            storage_diffs: vec![
                FrameTxStorageDiff {
                    address: address_a,
                    key: U256::from(1u64),
                    before: U256::ZERO,
                    after: U256::from(11u64),
                },
                FrameTxStorageDiff {
                    address: address_a,
                    key: U256::from(2u64),
                    before: U256::from(3u64),
                    after: U256::from(4u64),
                },
                FrameTxStorageDiff {
                    address: address_b,
                    key: U256::from(9u64),
                    before: U256::from(5u64),
                    after: U256::from(6u64),
                },
            ],
            deployed_contracts: vec![FrameTxDeployedContract {
                address: address_b,
                code_hash: deployed_hash,
            }],
            account_diffs: vec![
                FrameTxAccountDiff {
                    address: address_a,
                    nonce_changed: true,
                    code_hash_before: KECCAK_EMPTY,
                    code_hash_after: KECCAK_EMPTY,
                },
                FrameTxAccountDiff {
                    address: address_b,
                    nonce_changed: false,
                    code_hash_before: KECCAK_EMPTY,
                    code_hash_after: deployed_hash,
                },
            ],
            // Events retain emission order rather than address order.
            events: vec![
                FrameTxEvent {
                    address: address_b,
                    topics: vec![B256::repeat_byte(0xA1)],
                    data: Bytes::from_static(&[0x01, 0x02, 0x03]),
                },
                FrameTxEvent {
                    address: address_a,
                    topics: vec![
                        B256::repeat_byte(0xB1),
                        B256::repeat_byte(0xB2),
                        B256::repeat_byte(0xB3),
                        B256::repeat_byte(0xB4),
                    ],
                    data: Bytes::from_static(&[0x04, 0x05, 0x06, 0x07]),
                },
            ],
            gas_pre_charge: U256::from(1_000u64),
            gas_payer: address_b,
        };
        context
    }

    fn execute_txdiff(
        host: &mut DummyHost,
        param: u64,
        address: Address,
        in3: U256,
    ) -> Interpreter {
        let bytecode = Bytecode::new_raw(Bytes::from_static(&[TXDIFF]));
        let mut interpreter = Interpreter::default().with_bytecode(bytecode);
        let address_word: U256 = address.into_word().into();
        let _ = interpreter.stack.push(in3);
        let _ = interpreter.stack.push(address_word);
        let _ = interpreter.stack.push(U256::from(param));
        let instructions = instruction_table::<EthInterpreter, DummyHost>();
        let gas = gas_table();
        interpreter.step(&instructions, &gas, host).unwrap();
        interpreter
    }

    fn execute_txtrace(host: &mut DummyHost, param: u64, in2: U256) -> Interpreter {
        let bytecode = Bytecode::new_raw(Bytes::from_static(&[TXTRACE]));
        let mut interpreter = Interpreter::default().with_bytecode(bytecode);
        let _ = interpreter.stack.push(U256::from(param));
        let _ = interpreter.stack.push(in2);
        let instructions = instruction_table::<EthInterpreter, DummyHost>();
        let gas = gas_table();
        interpreter.step(&instructions, &gas, host).unwrap();
        interpreter
    }

    fn approve_interpreter(target: Address, scope: U256) -> Interpreter {
        let mut interpreter = Interpreter::default();
        interpreter.runtime_flag.is_static = true;
        interpreter.input.target_address = target;
        let _ = interpreter.stack.push(scope);
        let _ = interpreter.stack.push(U256::ZERO);
        let _ = interpreter.stack.push(U256::ZERO);
        interpreter
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
            (0x0C, U256::from(9u64)),
            (0x0D, U256::from(2u64)),
            (0x0E, U256::from_be_bytes([0xE1; 32])),
            (0x0F, U256::from(1u64)),
            (0x10, U256::from(3u64)),
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
        // 0x11 and all later unassigned parameters are undefined.
        for param in [0x11u64, 0xFF] {
            let mut interpreter = Interpreter::default();
            let _ = interpreter.stack.push(U256::from(param));
            assert_eq!(
                txparam(Ictx {
                    interpreter: &mut interpreter,
                    host: &mut host
                }),
                Err(InstructionResult::InvalidFEOpcode),
                "TXPARAM({param:#x})"
            );
        }

        let mut empty_keys = ctx();
        empty_keys.nonce_keys.clear();
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(empty_keys, 0x3);
        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::from(0x10u64));
        assert_eq!(
            txparam(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::InvalidFEOpcode)
        );
    }

    /// A frame index beyond the native pointer width must halt, not truncate to
    /// a valid frame on 32-bit hosts.
    #[test]
    fn frame_operands_reject_indices_larger_than_usize() {
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(ctx(), 0x3);
        let oversized = U256::from(usize::MAX) + U256::from(1u64);

        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::ZERO); // param
        let _ = interpreter.stack.push(oversized); // frameIndex
        assert_eq!(
            frameparam(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::InvalidFEOpcode)
        );

        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(oversized); // frameIndex
        let _ = interpreter.stack.push(U256::ZERO); // offset
        assert_eq!(
            framedataload(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::InvalidFEOpcode)
        );

        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(oversized); // frameIndex
        let _ = interpreter.stack.push(U256::ZERO); // length
        let _ = interpreter.stack.push(U256::ZERO); // data offset
        let _ = interpreter.stack.push(U256::ZERO); // memory offset
        assert_eq!(
            framedatacopy(Ictx {
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

    #[test]
    fn sigparam_rejects_removed_copy_param() {
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(ctx(), 0x3);
        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::from(0x04u64)); // removed copy param
        let _ = interpreter.stack.push(U256::ZERO); // signatureIndex
        assert_eq!(
            sigparam(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::InvalidFEOpcode)
        );
    }

    #[test]
    fn sigparam_reads_metadata() {
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(ctx(), 0x3);
        let signer: U256 = address!("4444444444444444444444444444444444444444")
            .into_word()
            .into();
        for (param, want) in [
            (0x00u64, signer),
            (0x01, U256::from(1u64)),
            (0x02, U256::from_be_bytes([0xCD; 32])),
            (0x03, U256::from(2u64)),
        ] {
            let mut interpreter = Interpreter::default();
            let _ = interpreter.stack.push(U256::from(param));
            let _ = interpreter.stack.push(U256::ZERO); // signatureIndex
            sigparam(Ictx {
                interpreter: &mut interpreter,
                host: &mut host,
            })
            .unwrap();
            assert_eq!(interpreter.stack.data()[0], want, "SIGPARAM({param:#x})");
        }
    }

    #[test]
    fn sigdatacopy_copies_with_zero_fill_and_calldatacopy_gas() {
        let bytecode = Bytecode::new_raw(Bytes::from_static(&[SIGDATACOPY]));
        let mut interpreter = Interpreter::default().with_bytecode(bytecode);
        // signatureIndex is deepest; memOffset is on top.
        let _ = interpreter.stack.push(U256::from(1u64));
        let _ = interpreter.stack.push(U256::from(5u64));
        let _ = interpreter.stack.push(U256::from(1u64));
        let _ = interpreter.stack.push(U256::from(4u64));
        let mut host =
            DummyHost::new(SpecId::default()).with_frame_tx(ctx_with_arbitrary_signature(), 0x3);
        let instructions = instruction_table::<EthInterpreter, DummyHost>();
        let gas = gas_table();

        interpreter.step(&instructions, &gas, &mut host).unwrap();

        assert!(interpreter.stack.data().is_empty());
        assert_eq!(
            interpreter.memory.slice_len(4, 5).as_ref(),
            &[0x22, 0x33, 0x00, 0x00, 0x00]
        );
        // Fixed 3 + one copy word (3) + one memory word (3).
        assert_eq!(interpreter.gas.total_gas_spent(), 9);
    }

    #[test]
    fn sigdatacopy_rejects_invalid_signature_entries() {
        for (case, frame, sig_index) in [
            ("non-ARBITRARY", ctx(), U256::ZERO),
            (
                "out-of-bounds index",
                ctx_with_arbitrary_signature(),
                U256::from(2u64),
            ),
        ] {
            let mut host = DummyHost::new(SpecId::default()).with_frame_tx(frame, 0x3);
            let mut interpreter = Interpreter::default();
            // signatureIndex is deepest; the remaining operands are zero.
            let _ = interpreter.stack.push(sig_index);
            let _ = interpreter.stack.push(U256::ZERO); // length
            let _ = interpreter.stack.push(U256::ZERO); // dataOffset
            let _ = interpreter.stack.push(U256::ZERO); // memOffset
            assert_eq!(
                sigdatacopy(Ictx {
                    interpreter: &mut interpreter,
                    host: &mut host
                }),
                Err(InstructionResult::InvalidFEOpcode),
                "{case}"
            );
        }
    }

    #[test]
    fn recentrootrefload_reads_fields_and_checks_bounds() {
        for (field, want) in [
            (0u64, U256::from_be_bytes([0xF1; 32])),
            (1, U256::from(123u64)),
            (2, U256::from_be_bytes([0xF2; 32])),
        ] {
            let mut host = DummyHost::new(SpecId::default()).with_frame_tx(ctx(), 0x3);
            let mut interpreter = Interpreter::default();
            let _ = interpreter.stack.push(U256::ZERO); // reference index
            let _ = interpreter.stack.push(U256::from(field)); // field on top
            recentrootrefload(Ictx {
                interpreter: &mut interpreter,
                host: &mut host,
            })
            .unwrap();
            assert_eq!(interpreter.stack.data(), &[want]);
        }

        for (index, field) in [(1u64, 0u64), (0, 3)] {
            let mut host = DummyHost::new(SpecId::default()).with_frame_tx(ctx(), 0x3);
            let mut interpreter = Interpreter::default();
            let _ = interpreter.stack.push(U256::from(index));
            let _ = interpreter.stack.push(U256::from(field));
            assert_eq!(
                recentrootrefload(Ictx {
                    interpreter: &mut interpreter,
                    host: &mut host
                }),
                Err(InstructionResult::InvalidFEOpcode)
            );
        }
    }

    #[test]
    fn txtrace_reads_ordered_trace_and_events() {
        let address_a: U256 = address!("1000000000000000000000000000000000000000")
            .into_word()
            .into();
        let address_b: U256 = address!("2000000000000000000000000000000000000000")
            .into_word()
            .into();
        let cases = [
            (0x00u64, 0u64, U256::from(2u64)),
            (0x01, 0, U256::from(3u64)),
            (0x02, 0, U256::from(1u64)),
            (0x03, 0, address_a),
            (0x04, 0, U256::from(10u64)),
            (0x05, 1, U256::from(25u64)),
            (0x06, 2, address_b),
            (0x07, 1, U256::from(2u64)),
            (0x08, 1, U256::from(3u64)),
            (0x09, 2, U256::from(6u64)),
            (0x0A, 0, address_b),
            (0x0B, 0, U256::from_be_bytes([0xD1; 32])),
            (0x0C, 0, U256::from(2u64)),
            (0x0D, 0, address_b),
            (0x0E, 1, U256::from(4u64)),
            (0x0F, 1, U256::from_be_bytes([0xB1; 32])),
            (0x10, 1, U256::from_be_bytes([0xB2; 32])),
            (0x11, 1, U256::from_be_bytes([0xB3; 32])),
            (0x12, 1, U256::from_be_bytes([0xB4; 32])),
            (0x13, 1, U256::from(4u64)),
            (0x14, 0, U256::from(1_000u64)),
            (0x15, 0, address_b),
        ];

        for (param, in2, want) in cases {
            let mut host = DummyHost::new(SpecId::default()).with_frame_tx(post_tx_ctx(), 0x3);
            let mut interpreter = Interpreter::default();
            let _ = interpreter.stack.push(U256::from(param));
            let _ = interpreter.stack.push(U256::from(in2));
            txtrace(Ictx {
                interpreter: &mut interpreter,
                host: &mut host,
            })
            .unwrap();
            assert_eq!(interpreter.stack.data(), &[want], "TXTRACE({param:#x})");
        }
    }

    #[test]
    fn txtrace_rejects_reserved_inputs_and_bad_indices() {
        for (param, in2) in [
            (0x00u64, 1u64), // reserved input must be zero
            (0x03, 2),       // balance index out of bounds
            (0x10, 0),       // event 0 has no topic 1
            (0x16, 0),       // undefined selector
        ] {
            let mut host = DummyHost::new(SpecId::default()).with_frame_tx(post_tx_ctx(), 0x3);
            let mut interpreter = Interpreter::default();
            let _ = interpreter.stack.push(U256::from(param));
            let _ = interpreter.stack.push(U256::from(in2));
            assert_eq!(
                txtrace(Ictx {
                    interpreter: &mut interpreter,
                    host: &mut host
                }),
                Err(InstructionResult::InvalidFEOpcode),
                "TXTRACE({param:#x}, {in2})"
            );
        }
    }

    #[test]
    fn txdiff_reads_direct_and_per_address_views() {
        let address_a = address!("1000000000000000000000000000000000000000");
        let address_b = address!("2000000000000000000000000000000000000000");
        let cases = [
            (0x00u64, address_a, 1u64, U256::ZERO),
            (0x01, address_a, 1, U256::from(11u64)),
            (0x02, address_a, 0, U256::from(10u64)),
            (0x03, address_a, 0, U256::from(7u64)),
            (0x04, address_b, 0, hash_word(KECCAK_EMPTY)),
            (0x05, address_b, 0, U256::from_be_bytes([0xD1; 32])),
            (0x06, address_a, 0, U256::from(2u64)),
            (0x07, address_a, 1, U256::from(1u64)),
            (0x08, address_a, 0, U256::from(1u64)),
            (0x09, address_a, 0, U256::from(1u64)),
            (0x0A, address_a, 0, U256::from(0b0111u64)),
            (0x0A, address_b, 0, U256::from(0b1110u64)),
        ];

        for (param, address, in3, want) in cases {
            let mut host = DummyHost::new(SpecId::default()).with_frame_tx(post_tx_ctx(), 0x3);
            let mut interpreter = Interpreter::default();
            let address_word: U256 = address.into_word().into();
            // in3 is deepest, address second, param on top.
            let _ = interpreter.stack.push(U256::from(in3));
            let _ = interpreter.stack.push(address_word);
            let _ = interpreter.stack.push(U256::from(param));
            txdiff(Ictx {
                interpreter: &mut interpreter,
                host: &mut host,
            })
            .unwrap();
            assert_eq!(interpreter.stack.data(), &[want], "TXDIFF({param:#x})");
        }
    }

    #[test]
    fn txdiff_rejects_reserved_inputs_and_bad_local_indices() {
        let address_a: U256 = address!("1000000000000000000000000000000000000000")
            .into_word()
            .into();
        for (param, in3) in [(0x06u64, 1u64), (0x07, 2), (0x0B, 0)] {
            let mut host = DummyHost::new(SpecId::default()).with_frame_tx(post_tx_ctx(), 0x3);
            let mut interpreter = Interpreter::default();
            let _ = interpreter.stack.push(U256::from(in3));
            let _ = interpreter.stack.push(address_a);
            let _ = interpreter.stack.push(U256::from(param));
            assert_eq!(
                txdiff(Ictx {
                    interpreter: &mut interpreter,
                    host: &mut host
                }),
                Err(InstructionResult::InvalidFEOpcode),
                "TXDIFF({param:#x})"
            );
        }
    }

    #[test]
    fn txdiff_live_fallback_uses_eip2929_warm_and_cold_gas() {
        let lookup_address = address!("3000000000000000000000000000000000000000");
        let code_hash = B256::repeat_byte(0xC1);
        let mut host = DummyHost::new(SpecId::BERLIN).with_frame_tx(post_tx_ctx(), 0x3);
        host.account_info = AccountInfo {
            balance: U256::from(77u64),
            nonce: 1,
            code_hash,
            ..Default::default()
        };
        host.account_is_cold = true;
        host.account_is_empty = false;
        host.storage_value = U256::from(88u64);
        host.storage_is_cold = true;

        let balance = execute_txdiff(&mut host, 0x02, lookup_address, U256::ZERO);
        assert_eq!(balance.stack.data(), &[U256::from(77u64)]);
        assert_eq!(balance.gas.total_gas_spent(), 2_600);

        let code = execute_txdiff(&mut host, 0x05, lookup_address, U256::ZERO);
        assert_eq!(code.stack.data(), &[hash_word(code_hash)]);
        assert_eq!(code.gas.total_gas_spent(), 100);

        let storage = execute_txdiff(&mut host, 0x00, lookup_address, U256::from(99u64));
        assert_eq!(storage.stack.data(), &[U256::from(88u64)]);
        assert_eq!(storage.gas.total_gas_spent(), 2_100);

        let warm_storage = execute_txdiff(&mut host, 0x01, lookup_address, U256::from(99u64));
        assert_eq!(warm_storage.stack.data(), &[U256::from(88u64)]);
        assert_eq!(warm_storage.gas.total_gas_spent(), 100);
    }

    #[test]
    fn txdiff_storage_access_does_not_warm_the_account() {
        let lookup_address = address!("3000000000000000000000000000000000000000");
        let mut host = DummyHost::new(SpecId::BERLIN).with_frame_tx(post_tx_ctx(), 0x3);
        host.account_info.balance = U256::from(77u64);
        host.account_is_cold = true;
        host.storage_value = U256::from(88u64);
        host.storage_is_cold = true;

        let storage = execute_txdiff(&mut host, 0x00, lookup_address, U256::from(99u64));
        assert_eq!(storage.stack.data(), &[U256::from(88u64)]);
        assert_eq!(storage.gas.total_gas_spent(), 2_100);

        let account = execute_txdiff(&mut host, 0x02, lookup_address, U256::ZERO);
        assert_eq!(account.stack.data(), &[U256::from(77u64)]);
        assert_eq!(account.gas.total_gas_spent(), 2_600);
    }

    #[test]
    fn txdiff_diff_hits_still_apply_eip2929_warmth_and_gas() {
        let address_a = address!("1000000000000000000000000000000000000000");
        let address_b = address!("2000000000000000000000000000000000000000");

        let mut storage_host = DummyHost::new(SpecId::BERLIN).with_frame_tx(post_tx_ctx(), 0x3);
        storage_host.storage_is_cold = true;
        let storage_before = execute_txdiff(&mut storage_host, 0x00, address_a, U256::from(1u64));
        assert_eq!(storage_before.stack.data(), &[U256::ZERO]);
        assert_eq!(storage_before.gas.total_gas_spent(), 2_100);
        let storage_after = execute_txdiff(&mut storage_host, 0x01, address_a, U256::from(1u64));
        assert_eq!(storage_after.stack.data(), &[U256::from(11u64)]);
        assert_eq!(storage_after.gas.total_gas_spent(), 100);

        let mut balance_host = DummyHost::new(SpecId::BERLIN).with_frame_tx(post_tx_ctx(), 0x3);
        balance_host.account_is_cold = true;
        let balance_before = execute_txdiff(&mut balance_host, 0x02, address_a, U256::ZERO);
        assert_eq!(balance_before.stack.data(), &[U256::from(10u64)]);
        assert_eq!(balance_before.gas.total_gas_spent(), 2_600);
        let balance_after = execute_txdiff(&mut balance_host, 0x03, address_a, U256::ZERO);
        assert_eq!(balance_after.stack.data(), &[U256::from(7u64)]);
        assert_eq!(balance_after.gas.total_gas_spent(), 100);

        let mut code_host = DummyHost::new(SpecId::BERLIN).with_frame_tx(post_tx_ctx(), 0x3);
        code_host.account_is_cold = true;
        let code_before = execute_txdiff(&mut code_host, 0x04, address_b, U256::ZERO);
        assert_eq!(code_before.stack.data(), &[hash_word(KECCAK_EMPTY)]);
        assert_eq!(code_before.gas.total_gas_spent(), 2_600);
        let code_after = execute_txdiff(&mut code_host, 0x05, address_b, U256::ZERO);
        assert_eq!(code_after.stack.data(), &[U256::from_be_bytes([0xD1; 32])]);
        assert_eq!(code_after.gas.total_gas_spent(), 100);
    }

    #[test]
    fn txdiff_event_index_ignores_unrelated_emitters() {
        let target = address!("1000000000000000000000000000000000000000");
        let unrelated = address!("3000000000000000000000000000000000000000");
        let mut frame = post_tx_ctx();
        frame.trace.events = (0..128)
            .map(|_| FrameTxEvent {
                address: unrelated,
                ..Default::default()
            })
            .collect();
        frame.trace.events.insert(
            17,
            FrameTxEvent {
                address: target,
                ..Default::default()
            },
        );
        frame.trace.events.push(FrameTxEvent {
            address: target,
            ..Default::default()
        });
        let mut host = DummyHost::new(SpecId::BERLIN).with_frame_tx(frame, 0x3);

        let count = execute_txdiff(&mut host, 0x08, target, U256::ZERO);
        assert_eq!(count.stack.data(), &[U256::from(2u64)]);
        let first = execute_txdiff(&mut host, 0x09, target, U256::ZERO);
        assert_eq!(first.stack.data(), &[U256::from(17u64)]);
        let second = execute_txdiff(&mut host, 0x09, target, U256::from(1u64));
        assert_eq!(second.stack.data(), &[U256::from(129u64)]);
        let unrelated_count = execute_txdiff(&mut host, 0x08, unrelated, U256::ZERO);
        assert_eq!(unrelated_count.stack.data(), &[U256::from(128u64)]);
        let unrelated_last = execute_txdiff(&mut host, 0x09, unrelated, U256::from(127u64));
        assert_eq!(unrelated_last.stack.data(), &[U256::from(128u64)]);
    }

    #[test]
    fn dummy_host_returns_the_same_prepared_context_allocation() {
        let target = address!("1000000000000000000000000000000000000000");
        let host = DummyHost::new(SpecId::BERLIN).with_frame_tx(post_tx_ctx(), 0x3);
        let first = host.frame_context().unwrap();
        let second = host.frame_context().unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.event_count_for_address(target), Some(1));
    }

    #[test]
    fn prepared_event_snapshot_stays_consistent_after_in_place_source_mutation() {
        let target = address!("1000000000000000000000000000000000000000");
        let original_emitter = address!("2000000000000000000000000000000000000000");
        let mut host = DummyHost::new(SpecId::BERLIN).with_frame_tx(post_tx_ctx(), 0x3);

        let source = host.frame_tx_context_mut().unwrap();
        source.trace.events[0].address = target;
        source.trace.events[0].data = Bytes::from_static(&[0xFF]);
        source.trace.events.push(FrameTxEvent {
            address: target,
            data: Bytes::from_static(&[0xEE]),
            ..Default::default()
        });

        let shared = host.frame_context().unwrap();
        assert_eq!(shared.trace.events.len(), 3);
        let snapshot = shared.event_snapshot().unwrap();
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot[0].address, original_emitter);
        assert_eq!(snapshot[0].data.as_ref(), &[1, 2, 3]);
        assert_eq!(shared.event_count_for_address(target), Some(1));
        assert_eq!(shared.event_global_index_for_address(target, 0), Some(1));
        drop(shared);

        let count = execute_txdiff(&mut host, 0x08, target, U256::ZERO);
        assert_eq!(count.stack.data(), &[U256::from(1u64)]);

        let trace_count = execute_txtrace(&mut host, 0x0C, U256::ZERO);
        assert_eq!(trace_count.stack.data(), &[U256::from(2u64)]);
        let trace_emitter = execute_txtrace(&mut host, 0x0D, U256::ZERO);
        let emitter_word: U256 = original_emitter.into_word().into();
        assert_eq!(trace_emitter.stack.data(), &[emitter_word]);
        let trace_data_len = execute_txtrace(&mut host, 0x13, U256::ZERO);
        assert_eq!(trace_data_len.stack.data(), &[U256::from(3u64)]);

        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::from(3u64));
        let _ = interpreter.stack.push(U256::ZERO);
        let _ = interpreter.stack.push(U256::ZERO);
        let _ = interpreter.stack.push(U256::ZERO);
        eventdatacopy(Ictx {
            interpreter: &mut interpreter,
            host: &mut host,
        })
        .unwrap();
        assert_eq!(interpreter.memory.slice_len(0, 3).as_ref(), &[1, 2, 3]);
    }

    #[test]
    fn native_host_context_controls_static_call_requirement() {
        let target = Address::repeat_byte(0x11);
        let unrelated = Address::repeat_byte(0x22);

        for (mode, expected) in [
            (0, false),
            (FRAME_MODE_VERIFY, true),
            (2, false),
            (FRAME_MODE_POST_TX, true),
        ] {
            let host = DummyHost::new(SpecId::BERLIN).with_frame_tx(
                FrameTxContext {
                    frames: vec![FrameInfo {
                        resolved_target: target,
                        mode,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                0,
            );
            assert_eq!(
                frame_tx_call_requires_static(&host, target),
                expected,
                "frame mode {mode}"
            );
            assert!(!frame_tx_call_requires_static(&host, unrelated));
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn external_event_indices_are_ignored_and_rebuilt() {
        let target = address!("1000000000000000000000000000000000000000");
        let shared = post_tx_ctx().into_shared();
        let mut wire = serde_json::to_value(&*shared).unwrap();
        let object = wire.as_object_mut().unwrap();
        assert!(!object.contains_key("event_index"));
        object.insert(
            "event_index".to_owned(),
            serde_json::json!({ "entries": [[target, 0]], "source": [0, 1] }),
        );

        let mut decoded: FrameTxContext = serde_json::from_value(wire).unwrap();
        assert_eq!(decoded.event_count_for_address(target), None);
        assert_eq!(decoded.event_snapshot(), None);
        decoded.trace.events.push(FrameTxEvent {
            address: target,
            ..Default::default()
        });
        let shared = decoded.into_shared();
        assert_eq!(shared.event_count_for_address(target), Some(2));
        assert_eq!(shared.event_snapshot().unwrap().len(), 3);
    }

    #[test]
    fn eventdatacopy_uses_declared_stack_order_and_strict_bounds() {
        let bytecode = Bytecode::new_raw(Bytes::from_static(&[EVENTDATACOPY]));
        let mut interpreter = Interpreter::default().with_bytecode(bytecode);
        // length is deepest; eventIndex is on top.
        let _ = interpreter.stack.push(U256::from(3u64));
        let _ = interpreter.stack.push(U256::from(1u64));
        let _ = interpreter.stack.push(U256::from(4u64));
        let _ = interpreter.stack.push(U256::from(1u64));
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(post_tx_ctx(), 0x3);
        let instructions = instruction_table::<EthInterpreter, DummyHost>();
        let gas = gas_table();

        interpreter.step(&instructions, &gas, &mut host).unwrap();

        assert!(interpreter.stack.data().is_empty());
        assert_eq!(interpreter.memory.slice_len(4, 3).as_ref(), &[5, 6, 7]);
        assert_eq!(interpreter.gas.total_gas_spent(), 9);

        for (event_index, data_offset, len, want) in [
            (2u64, 0u64, 0u64, InstructionResult::InvalidFEOpcode),
            (1, 2, 3, InstructionResult::OutOfOffset),
        ] {
            let mut interpreter = Interpreter::default();
            let _ = interpreter.stack.push(U256::from(len));
            let _ = interpreter.stack.push(U256::from(data_offset));
            let _ = interpreter.stack.push(U256::ZERO);
            let _ = interpreter.stack.push(U256::from(event_index));
            assert_eq!(
                eventdatacopy(Ictx {
                    interpreter: &mut interpreter,
                    host: &mut host
                }),
                Err(want)
            );
        }
    }

    #[test]
    fn trace_opcodes_require_current_post_tx_frame() {
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(ctx(), 0x3);

        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::ZERO);
        let _ = interpreter.stack.push(U256::ZERO);
        assert_eq!(
            txtrace(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::InvalidFEOpcode)
        );

        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::ZERO);
        let _ = interpreter.stack.push(U256::ZERO);
        let _ = interpreter.stack.push(U256::ZERO);
        assert_eq!(
            txdiff(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::InvalidFEOpcode)
        );

        let mut interpreter = Interpreter::default();
        for _ in 0..4 {
            let _ = interpreter.stack.push(U256::ZERO);
        }
        assert_eq!(
            eventdatacopy(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::InvalidFEOpcode)
        );
    }

    #[test]
    fn txtrace_charges_provisional_flat_gas() {
        let bytecode = Bytecode::new_raw(Bytes::from_static(&[TXTRACE]));
        let mut interpreter = Interpreter::default().with_bytecode(bytecode);
        let _ = interpreter.stack.push(U256::ZERO); // param
        let _ = interpreter.stack.push(U256::ZERO); // in2
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(post_tx_ctx(), 0x3);
        let instructions = instruction_table::<EthInterpreter, DummyHost>();
        let gas = gas_table();

        interpreter.step(&instructions, &gas, &mut host).unwrap();

        assert_eq!(interpreter.gas.total_gas_spent(), 100);
    }

    #[test]
    fn approve_native_host_requires_current_target_and_intersected_scopes() {
        let mut verify_context = ctx();
        let current = verify_context.frame_index as usize;
        let target = verify_context.frames[current].resolved_target;
        verify_context.frames[current].mode = FRAME_MODE_VERIFY;
        verify_context.frames[current].flags = 0x03;
        verify_context.approvable_scopes = 0x01;
        // The native host permits both scopes, but the context permits only payment.
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(verify_context, 0x03);

        let mut interpreter = approve_interpreter(Address::repeat_byte(0xFE), U256::from(1u64));
        assert_eq!(
            approve(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::Revert)
        );
        assert_eq!(host.frame_approve_calls, 0);
        assert_eq!(interpreter.stack.len(), 3);

        let mut interpreter = approve_interpreter(target, U256::from(3u64));
        assert_eq!(
            approve(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::Revert)
        );
        assert_eq!(host.frame_approve_calls, 0);

        let mut interpreter = approve_interpreter(target, U256::from(1u64));
        assert_eq!(
            approve(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::Return)
        );
        assert_eq!(host.frame_approve_calls, 1);

        let mut verify_context = ctx();
        let current = verify_context.frame_index as usize;
        let target = verify_context.frames[current].resolved_target;
        verify_context.frames[current].mode = FRAME_MODE_VERIFY;
        verify_context.frames[current].flags = 0x01;
        verify_context.approvable_scopes = 0x03;
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(verify_context, 0x03);
        let mut interpreter = approve_interpreter(target, U256::from(2u64));
        assert_eq!(
            approve(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::Revert)
        );
        assert_eq!(host.frame_approve_calls, 0);
    }

    #[test]
    fn approve_exceptionally_halts_in_post_tx_before_host_callback() {
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(post_tx_ctx(), 0x3);
        let mut interpreter = Interpreter::default();
        interpreter.runtime_flag.is_static = true;
        let _ = interpreter.stack.push(U256::from(1u64));
        let _ = interpreter.stack.push(U256::ZERO);
        let _ = interpreter.stack.push(U256::ZERO);

        assert_eq!(
            approve(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::InvalidFEOpcode)
        );
        assert_eq!(host.frame_approve_calls, 0);
        assert_eq!(interpreter.stack.len(), 3);
    }

    #[test]
    fn approve_rejects_malformed_current_frame_before_host_callback() {
        let mut out_of_range = ctx();
        out_of_range.frame_index = out_of_range.frames.len() as u64;
        let mut unrepresentable = ctx();
        unrepresentable.frame_index = u64::MAX;
        let mut invalid_mode = ctx();
        invalid_mode.frames[invalid_mode.frame_index as usize].mode = u8::MAX;

        for frame in [
            FrameTxContext::default(),
            out_of_range,
            unrepresentable,
            invalid_mode,
        ] {
            let mut host = DummyHost::new(SpecId::default()).with_frame_tx(frame, 0x3);
            let mut interpreter = Interpreter::default();
            let _ = interpreter.stack.push(U256::from(1u64));
            let _ = interpreter.stack.push(U256::ZERO);
            let _ = interpreter.stack.push(U256::ZERO);

            assert_eq!(
                approve(Ictx {
                    interpreter: &mut interpreter,
                    host: &mut host
                }),
                Err(InstructionResult::InvalidFEOpcode)
            );
            assert_eq!(host.frame_approve_calls, 0);
            assert_eq!(interpreter.stack.len(), 3);
        }
    }

    #[test]
    fn approve_rejects_out_of_range_scope_before_host_callback() {
        let mut verify_context = ctx();
        let current = verify_context.frame_index as usize;
        let target = verify_context.frames[current].resolved_target;
        verify_context.frames[current].mode = FRAME_MODE_VERIFY;
        verify_context.frames[current].flags = 0x03;
        verify_context.approvable_scopes = 0x03;
        let mut host = DummyHost::new(SpecId::default()).with_frame_tx(verify_context, u64::MAX);
        let mut interpreter = approve_interpreter(target, U256::MAX);

        assert_eq!(
            approve(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::Revert)
        );
        assert_eq!(host.frame_approve_calls, 0);
    }
}

#[cfg(all(test, feature = "std"))]
mod slot_tests {
    use super::*;
    use crate::{host::DummyHost, InstructionContext as Ictx, Interpreter};
    use primitives::hardfork::SpecId;

    /// The tooling slot supplies a context to a host that knows nothing about
    /// frame transactions, which is how a test runner drives these opcodes
    /// without any shared revm type being changed.
    #[test]
    fn slot_supplies_context_to_an_unaware_host() {
        let mut host = DummyHost::new(SpecId::default());
        // No context anywhere: the opcode halts.
        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::from(0x01u64));
        assert_eq!(
            txparam(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::InvalidFEOpcode)
        );

        set_frame_tx_context(Some(FrameTxContext {
            nonce: 42,
            approvable_scopes: 0x3,
            frames: vec![FrameInfo {
                mode: FRAME_MODE_VERIFY,
                flags: 0x03,
                ..Default::default()
            }],
            ..Default::default()
        }));

        let mut interpreter = Interpreter::default();
        let _ = interpreter.stack.push(U256::from(0x01u64)); // TXPARAM nonce
        txparam(Ictx {
            interpreter: &mut interpreter,
            host: &mut host,
        })
        .unwrap();
        assert_eq!(interpreter.stack.data()[0], U256::from(42u64));

        // APPROVE honours the slot's permitted scopes.
        let mut interpreter = Interpreter::default();
        interpreter.runtime_flag.is_static = true;
        let _ = interpreter.stack.push(U256::from(3u64));
        let _ = interpreter.stack.push(U256::ZERO);
        let _ = interpreter.stack.push(U256::ZERO);
        assert_eq!(
            approve(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::Return)
        );

        // A scope outside the permitted mask reverts.
        set_frame_tx_context(Some(FrameTxContext {
            approvable_scopes: 0x1,
            frames: vec![FrameInfo {
                mode: FRAME_MODE_VERIFY,
                flags: 0x03,
                ..Default::default()
            }],
            ..Default::default()
        }));
        let mut interpreter = Interpreter::default();
        interpreter.runtime_flag.is_static = true;
        let _ = interpreter.stack.push(U256::from(3u64));
        let _ = interpreter.stack.push(U256::ZERO);
        let _ = interpreter.stack.push(U256::ZERO);
        assert_eq!(
            approve(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::Revert)
        );

        set_frame_tx_context(None);
    }

    #[test]
    fn tls_approve_rejects_nested_foreign_target_and_disallowed_scope() {
        let outer = Address::repeat_byte(0x11);
        let current = Address::repeat_byte(0x22);
        let _guard = install_frame_tx_context(FrameTxContext {
            frame_index: 1,
            approvable_scopes: 0x03,
            frames: vec![
                FrameInfo {
                    resolved_target: outer,
                    mode: FRAME_MODE_VERIFY,
                    flags: 0x03,
                    ..Default::default()
                },
                FrameInfo {
                    resolved_target: current,
                    mode: FRAME_MODE_VERIFY,
                    flags: 0x01,
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let mut host = DummyHost::new(SpecId::default());

        let mut interpreter = Interpreter::default();
        interpreter.input.target_address = outer;
        let _ = interpreter.stack.push(U256::from(1u64));
        let _ = interpreter.stack.push(U256::ZERO);
        let _ = interpreter.stack.push(U256::ZERO);
        assert_eq!(
            approve(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::Revert)
        );
        assert_eq!(interpreter.stack.len(), 3);

        let mut interpreter = Interpreter::default();
        interpreter.input.target_address = current;
        let _ = interpreter.stack.push(U256::from(2u64));
        let _ = interpreter.stack.push(U256::ZERO);
        let _ = interpreter.stack.push(U256::ZERO);
        assert_eq!(
            approve(Ictx {
                interpreter: &mut interpreter,
                host: &mut host
            }),
            Err(InstructionResult::Revert)
        );

        let mut interpreter = Interpreter::default();
        interpreter.input.target_address = current;
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
        assert_eq!(host.frame_approve_calls, 0);
    }

    #[test]
    fn static_call_requirement_matches_only_current_verify_and_post_tx_targets() {
        let target = Address::repeat_byte(0x11);
        let unrelated = Address::repeat_byte(0x22);
        let host = DummyHost::new(SpecId::default());
        set_frame_tx_context(None);

        for (mode, expected) in [
            (0, false),
            (FRAME_MODE_VERIFY, true),
            (2, false),
            (FRAME_MODE_POST_TX, true),
        ] {
            let _guard = install_frame_tx_context(FrameTxContext {
                frames: vec![FrameInfo {
                    resolved_target: target,
                    mode,
                    ..Default::default()
                }],
                ..Default::default()
            });
            assert_eq!(frame_tx_call_requires_static(&host, target), expected);
            assert!(!frame_tx_call_requires_static(&host, unrelated));
        }

        let _guard = install_frame_tx_context(FrameTxContext::default());
        assert!(!frame_tx_call_requires_static(&host, target));
    }

    #[test]
    fn scoped_contexts_share_and_restore_across_nesting_and_unwind() {
        set_frame_tx_context(Some(FrameTxContext {
            nonce: 10,
            ..Default::default()
        }));

        let outer = install_frame_tx_context(FrameTxContext {
            nonce: 20,
            ..Default::default()
        });
        let first = frame_tx_context().unwrap();
        let second = frame_tx_context().unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.nonce, 20);

        {
            let _inner = install_frame_tx_context(FrameTxContext {
                nonce: 30,
                ..Default::default()
            });
            assert_eq!(frame_tx_context().unwrap().nonce, 30);
        }
        assert_eq!(frame_tx_context().unwrap().nonce, 20);

        let unwind = std::panic::catch_unwind(|| {
            let _guard = install_frame_tx_context(FrameTxContext {
                nonce: 40,
                ..Default::default()
            });
            panic!("test unwind");
        });
        assert!(unwind.is_err());
        assert_eq!(frame_tx_context().unwrap().nonce, 20);

        drop(outer);
        assert_eq!(frame_tx_context().unwrap().nonce, 10);
        set_frame_tx_context(None);
    }
}
