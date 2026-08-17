use crate::InstructionContext as Ictx;
use crate::{
    instructions::utility::{IntoAddress, IntoU256},
    interpreter_types::{InputsTr, InterpreterTypes as ITy, MemoryTr, RuntimeFlag, StackTr},
    Gas, Host, InstructionExecResult as Result, InstructionResult,
};
use context_interface::{
    context::{SStoreResult, StateLoad},
    host::{LoadError, SetDelegateError},
    journaled_state::AccountInfoLoad,
};
use core::cmp::min;
use primitives::{
    hardfork::SpecId::{self, *},
    Address, Bytes, Log, LogData, B256, BLOCK_HASH_HISTORY, U256,
};

/// Loads an account, handling cold load gas accounting.
///
/// Pre-Berlin, `cold_account_additional_cost` is 0, so the cold load logic is a no-op.
fn load_account<'a, H: Host + ?Sized>(
    gas: &mut Gas,
    host: &'a mut H,
    address: primitives::Address,
    load_code: bool,
) -> core::result::Result<AccountInfoLoad<'a>, LoadError> {
    let cold_load_gas = host.gas_params().cold_account_additional_cost();
    let skip_cold_load = gas.remaining() < cold_load_gas;
    let account = host.load_account_info_skip_cold_load(address, load_code, skip_cold_load)?;
    if account.is_cold && !gas.record_regular_cost(cold_load_gas) {
        return Err(LoadError::ColdLoadSkipped);
    }
    Ok(account)
}

/// Implements the BALANCE instruction.
///
/// Gets the balance of the given account.
pub fn balance<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    popn_top!([], top, context.interpreter);
    let address = top.into_address();
    let account = load_account(&mut context.interpreter.gas, context.host, address, false)?;
    *top = account.balance;
    Ok(())
}

/// EIP-1884: Repricing for trie-size-dependent opcodes
pub fn selfbalance<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    check!(context.interpreter, ISTANBUL);

    let balance = context
        .host
        .balance(context.interpreter.input.target_address())
        .ok_or(InstructionResult::FatalExternalError)?;
    push!(context.interpreter, balance.data);
    Ok(())
}

/// Implements the EXTCODESIZE instruction.
///
/// Gets the size of an account's code.
pub fn extcodesize<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    popn_top!([], top, context.interpreter);
    let address = top.into_address();
    let account = load_account(&mut context.interpreter.gas, context.host, address, true)?;
    // safe to unwrap because we are loading code
    *top = U256::from(account.code.as_ref().unwrap().len());
    Ok(())
}

/// EIP-1052: EXTCODEHASH opcode
pub fn extcodehash<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    check!(context.interpreter, PETERSBURG);
    popn_top!([], top, context.interpreter);
    let address = top.into_address();
    let account = load_account(&mut context.interpreter.gas, context.host, address, false)?;
    // if account is empty, code hash is zero
    let code_hash = if account.is_empty() {
        B256::ZERO
    } else {
        account.code_hash
    };
    *top = code_hash.into_u256();
    Ok(())
}

/// Implements the EXTCODECOPY instruction.
///
/// Copies a portion of an account's code to memory.
pub fn extcodecopy<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    popn!(
        [address, memory_offset, code_offset, len_u256],
        context.interpreter
    );
    let address = address.into_address();

    let len = as_usize_or_fail!(context.interpreter, len_u256);
    gas!(
        context.interpreter,
        context.host.gas_params().extcodecopy(len)
    );

    let mut memory_offset_usize = 0;
    // resize memory only if len is not zero
    if len != 0 {
        // fail on casting of memory_offset only if len is not zero.
        memory_offset_usize = as_usize_or_fail!(context.interpreter, memory_offset);
        // Resize memory to fit the code
        context
            .interpreter
            .resize_memory(context.host.gas_params(), memory_offset_usize, len)?;
    }

    let account = load_account(&mut context.interpreter.gas, context.host, address, true)?;
    let code = account.code.as_ref().unwrap().original_bytes();

    let code_offset_usize = min(as_usize_saturated!(code_offset), code.len());

    // Note: This can't panic because we resized memory to fit.
    // len zero is handled in set_data
    context
        .interpreter
        .memory
        .set_data(memory_offset_usize, code_offset_usize, len, &code);
    Ok(())
}

/// Implements the BLOCKHASH instruction.
///
/// Gets the hash of one of the 256 most recent complete blocks.
pub fn blockhash<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    popn_top!([], number, context.interpreter);

    let requested_number = *number;
    let block_number = context.host.block_number();

    let Some(diff) = block_number.checked_sub(requested_number) else {
        *number = U256::ZERO;
        return Ok(());
    };

    let diff = as_u64_saturated!(diff);

    // blockhash should push zero if number is same as current block number.
    if diff == 0 {
        *number = U256::ZERO;
        return Ok(());
    }

    *number = if diff <= BLOCK_HASH_HISTORY {
        let hash = context
            .host
            .block_hash(as_u64_saturated!(requested_number))
            .ok_or(InstructionResult::FatalExternalError)?;
        U256::from_be_bytes(hash.0)
    } else {
        U256::ZERO
    };
    Ok(())
}

/// Implements the SLOAD instruction.
///
/// Loads a word from storage.
pub fn sload<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    popn_top!([], index, context.interpreter);
    let spec_id = context.interpreter.runtime_flag.spec_id();
    let target = context.interpreter.input.target_address();

    if spec_id.is_enabled_in(BERLIN) {
        let additional_cold_cost = context.host.gas_params().cold_storage_additional_cost();
        let skip_cold = context.interpreter.gas.remaining() < additional_cold_cost;
        let storage = context
            .host
            .sload_skip_cold_load(target, *index, skip_cold)?;
        if storage.is_cold {
            gas!(context.interpreter, additional_cold_cost);
        }
        *index = storage.data;
    } else {
        let storage = context
            .host
            .sload(target, *index)
            .ok_or(InstructionResult::FatalExternalError)?;
        *index = storage.data;
    };
    Ok(())
}

/// Implements the SSTORE instruction.
///
/// Stores a word to storage.
pub fn sstore<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    sstore_with_gas_accounting(context, sstore_default_gas_accounting)
}

/// Implements SSTORE, delegating dynamic gas and refund accounting to `gas_accounting`.
///
/// This helper performs the common SSTORE instruction flow: static-call checks,
/// stack pops, stipend/static-gas charging, and the journaled storage write.
/// Custom instruction sets can override the SSTORE opcode and call this helper
/// with their own gas accounting closure.
pub fn sstore_with_gas_accounting<'a, IT, H, F>(
    mut context: Ictx<'a, H, IT>,
    gas_accounting: F,
) -> Result
where
    IT: ITy,
    H: Host + ?Sized,
    F: for<'ctx, 'load> FnOnce(
        &'ctx mut Ictx<'a, H, IT>,
        Address,
        &'load StateLoad<SStoreResult>,
    ) -> Result,
{
    require_non_staticcall!(context.interpreter);
    popn!([index, value], context.interpreter);

    let target = context.interpreter.input.target_address();
    let spec_id = context.interpreter.runtime_flag.spec_id();

    // EIP-2200: Structured Definitions for Net Gas Metering
    // If gasleft is less than or equal to gas stipend, fail the current call frame with 'out of gas' exception.
    if spec_id.is_enabled_in(ISTANBUL)
        && context.interpreter.gas.remaining() <= context.host.gas_params().call_stipend()
    {
        return Err(InstructionResult::ReentrancySentryOOG);
    }

    gas!(
        context.interpreter,
        context.host.gas_params().sstore_static_gas()
    );

    let state_load = if spec_id.is_enabled_in(BERLIN) {
        let skip_cold_load =
            context.interpreter.gas.remaining() < context.host.gas_params().cold_storage_cost();
        context
            .host
            .sstore_skip_cold_load(target, index, value, skip_cold_load)?
    } else {
        context
            .host
            .sstore(target, index, value)
            .ok_or(InstructionResult::FatalExternalError)?
    };

    gas_accounting(&mut context, target, &state_load)
}

/// Default dynamic gas and refund accounting for SSTORE.
pub fn sstore_default_gas_accounting<IT, H>(
    context: &mut Ictx<'_, H, IT>,
    _target: Address,
    state_load: &StateLoad<SStoreResult>,
) -> Result
where
    IT: ITy,
    H: Host + ?Sized,
{
    let spec_id = context.interpreter.runtime_flag.spec_id();
    let is_istanbul = spec_id.is_enabled_in(ISTANBUL);

    // dynamic gas
    gas!(
        context.interpreter,
        context.host.gas_params().sstore_dynamic_gas(
            is_istanbul,
            &state_load.data,
            state_load.is_cold
        )
    );

    // state gas for new slot creation (EIP-8037)
    if context.host.is_amsterdam_eip8037_enabled() {
        state_gas!(
            context.interpreter,
            context.host.gas_params().sstore_state_gas(&state_load.data)
        );

        // EIP-8037 issue #2: 0→x→0 storage restoration refills the reservoir
        // directly rather than routing the state gas through the capped refund
        // counter. The regular-gas portion of the restoration still flows
        // through `sstore_refund` below.
        let refill = context
            .host
            .gas_params()
            .sstore_state_gas_refill(&state_load.data);
        if refill > 0 {
            context.interpreter.gas.refill_reservoir(refill);
        }
    }

    // refund
    context.interpreter.gas.record_refund(
        context
            .host
            .gas_params()
            .sstore_refund(is_istanbul, &state_load.data),
    );
    Ok(())
}

/// EIP-1153: Transient storage opcodes
/// Store value to transient storage
pub fn tstore<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    check!(context.interpreter, CANCUN);
    require_non_staticcall!(context.interpreter);
    popn!([index, value], context.interpreter);

    context
        .host
        .tstore(context.interpreter.input.target_address(), index, value);
    Ok(())
}

/// EIP-1153: Transient storage opcodes
/// Load value from transient storage
pub fn tload<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    check!(context.interpreter, CANCUN);
    popn_top!([], index, context.interpreter);

    *index = context
        .host
        .tload(context.interpreter.input.target_address(), *index);
    Ok(())
}

/// EIP-7819: Sets delegation code at a deterministic address.
pub fn setdelegate<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    if !context.host.is_eip7819_enabled()
        || !context
            .interpreter
            .runtime_flag
            .spec_id()
            .is_enabled_in(SpecId::PRAGUE)
    {
        return Err(InstructionResult::NotActivated);
    }
    require_non_staticcall!(context.interpreter);
    popn!([salt, target], context.interpreter);

    let target = target.into_address();
    let location =
        primitives::eip7819::setdelegate_address(context.interpreter.input.target_address(), salt);
    let existed = context
        .host
        .set_delegate(location, target)
        .ok_or(InstructionResult::FatalExternalError)?
        .map_err(|err| match err {
            SetDelegateError::AddressCollision => InstructionResult::AddressCollision,
        })?;
    if existed {
        context
            .interpreter
            .gas
            .record_refund(primitives::eip7819::EXISTING_ACCOUNT_REFUND);
    }
    push!(context.interpreter, location.into_word().into());
    Ok(())
}

/// EIP-7851: Updates the current execution-context account's delegation.
pub fn setselfdelegate<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    if !context.host.is_eip7851_enabled()
        || !context
            .interpreter
            .runtime_flag
            .spec_id()
            .is_enabled_in(SpecId::PRAGUE)
    {
        return Err(InstructionResult::NotActivated);
    }
    require_non_staticcall!(context.interpreter);
    popn!([target], context.interpreter);

    let target = target.into_address();
    if target.is_zero() {
        push!(context.interpreter, U256::ZERO);
        return Ok(());
    }

    let authority = context.interpreter.input.target_address();
    let success = context
        .host
        .set_self_delegate(authority, target)
        .ok_or(InstructionResult::FatalExternalError)?;
    push!(context.interpreter, U256::from(u8::from(success)));
    Ok(())
}

/// Implements the LOG0-LOG4 instructions.
///
/// Appends log record with N topics.
pub fn log<const N: usize, H: Host + ?Sized>(context: Ictx<'_, H, impl ITy>) -> Result {
    require_non_staticcall!(context.interpreter);

    popn!([offset, len], context.interpreter);
    let len = as_usize_or_fail!(context.interpreter, len);
    gas!(
        context.interpreter,
        context.host.gas_params().log_cost(N as u8, len as u64)
    );
    let data = if len == 0 {
        Bytes::new()
    } else {
        let offset = as_usize_or_fail!(context.interpreter, offset);
        // Resize memory to fit the data
        context
            .interpreter
            .resize_memory(context.host.gas_params(), offset, len)?;
        Bytes::copy_from_slice(context.interpreter.memory.slice_len(offset, len).as_ref())
    };
    let Some(topics) = context.interpreter.stack.popn::<N>() else {
        return Err(InstructionResult::StackUnderflow);
    };

    let log = Log {
        address: context.interpreter.input.target_address(),
        data: LogData::new(topics.into_iter().map(B256::from).collect(), data)
            .expect("LogData should have <=4 topics"),
    };

    context.host.log(log);
    Ok(())
}

/// Implements the SELFDESTRUCT instruction.
///
/// Halt execution and register account for later deletion.
pub fn selfdestruct<IT: ITy, H: Host + ?Sized>(context: Ictx<'_, H, IT>) -> Result {
    require_non_staticcall!(context.interpreter);
    popn!([target], context.interpreter);
    let target = target.into_address();
    let spec = context.interpreter.runtime_flag.spec_id();

    let cold_load_gas = context.host.gas_params().selfdestruct_cold_cost();

    let skip_cold_load = context.interpreter.gas.remaining() < cold_load_gas;
    let res = context.host.selfdestruct(
        context.interpreter.input.target_address(),
        target,
        skip_cold_load,
    )?;

    // EIP-161: State trie clearing (invariant-preserving alternative)
    let should_charge_topup = if spec.is_enabled_in(SpecId::SPURIOUS_DRAGON) {
        res.had_value && !res.target_exists
    } else {
        !res.target_exists
    };

    gas!(
        context.interpreter,
        context
            .host
            .gas_params()
            .selfdestruct_cost(should_charge_topup, res.is_cold)
    );

    // State gas for new account creation (EIP-8037)
    if context.host.is_amsterdam_eip8037_enabled() && should_charge_topup {
        state_gas!(
            context.interpreter,
            context.host.gas_params().new_account_state_gas()
        );
    }

    if !res.previously_destroyed {
        context
            .interpreter
            .gas
            .record_refund(context.host.gas_params().selfdestruct_refund());
    }

    Err(InstructionResult::SelfDestruct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gas_table, host::DummyHost, instruction_table, interpreter::EthInterpreter, Gas,
        Interpreter,
    };
    use bytecode::{
        opcode::{SETDELEGATE, SETSELFDELEGATE},
        Bytecode,
    };
    use primitives::{address, eip7819, eip7851, Bytes};

    const EXECUTION_ADDRESS: Address = address!("1111111111111111111111111111111111111111");
    const TARGET: Address = address!("2222222222222222222222222222222222222222");

    fn setdelegate_interpreter(gas_limit: u64, target: U256, salt: U256) -> Interpreter {
        let mut interpreter = Interpreter::default()
            .with_bytecode(Bytecode::new_raw(Bytes::from_static(&[SETDELEGATE])));
        interpreter.gas = Gas::new(gas_limit);
        interpreter.runtime_flag.spec_id = SpecId::PRAGUE;
        interpreter.input.target_address = EXECUTION_ADDRESS;
        assert!(interpreter.stack.push(target));
        assert!(interpreter.stack.push(salt));
        interpreter
    }

    fn step_setdelegate(
        interpreter: &mut Interpreter,
        host: &mut DummyHost,
    ) -> core::result::Result<(), InstructionResult> {
        interpreter.step(
            &instruction_table::<EthInterpreter, DummyHost>(),
            &gas_table(),
            host,
        )
    }

    fn setselfdelegate_interpreter(gas_limit: u64, target: U256, spec: SpecId) -> Interpreter {
        let mut interpreter = Interpreter::default()
            .with_bytecode(Bytecode::new_raw(Bytes::from_static(&[SETSELFDELEGATE])));
        interpreter.gas = Gas::new(gas_limit);
        interpreter.runtime_flag.spec_id = spec;
        interpreter.input.target_address = EXECUTION_ADDRESS;
        assert!(interpreter.stack.push(target));
        interpreter
    }

    #[test]
    fn setdelegate_uses_reference_address_and_operand_order() {
        let mut host = DummyHost::default();
        host.enable_eip7819 = true;
        let mut interpreter = setdelegate_interpreter(
            eip7819::EMPTY_ACCOUNT_COST,
            TARGET.into_word().into(),
            U256::ZERO,
        );

        step_setdelegate(&mut interpreter, &mut host).unwrap();

        let location = address!("7a41c03bf3062738d4ad052749101d8ec0f5639d");
        let location_word: U256 = location.into_word().into();
        assert_eq!(interpreter.stack.data(), &[location_word]);
        assert_eq!(host.set_delegate_calls, [(location, TARGET)]);
        assert_eq!(
            interpreter.gas.total_gas_spent(),
            eip7819::EMPTY_ACCOUNT_COST
        );
        assert_eq!(interpreter.gas.refunded(), 0);
    }

    #[test]
    fn setdelegate_truncates_target_and_refunds_existing_account() {
        let target = (U256::from(0xabu64) << 160) | U256::from_be_slice(TARGET.as_slice());
        let mut host = DummyHost::default();
        host.enable_eip7819 = true;
        host.set_delegate_result = Some(Ok(true));
        let mut interpreter = setdelegate_interpreter(30_000, target, U256::from(1u64));

        step_setdelegate(&mut interpreter, &mut host).unwrap();

        assert_eq!(host.set_delegate_calls[0].1, TARGET);
        assert_eq!(interpreter.gas.refunded(), eip7819::EXISTING_ACCOUNT_REFUND);
    }

    #[test]
    fn setdelegate_charges_before_activation_static_and_stack_checks() {
        let mut disabled = DummyHost::default();
        let mut interpreter =
            setdelegate_interpreter(30_000, TARGET.into_word().into(), U256::ZERO);
        assert_eq!(
            step_setdelegate(&mut interpreter, &mut disabled),
            Err(InstructionResult::NotActivated)
        );
        assert_eq!(
            interpreter.gas.total_gas_spent(),
            eip7819::EMPTY_ACCOUNT_COST
        );
        assert!(disabled.set_delegate_calls.is_empty());

        let mut enabled = DummyHost::default();
        enabled.enable_eip7819 = true;
        let mut interpreter = setdelegate_interpreter(30_000, U256::ZERO, U256::ZERO);
        interpreter.stack.clear();
        interpreter.runtime_flag.is_static = true;
        assert_eq!(
            step_setdelegate(&mut interpreter, &mut enabled),
            Err(InstructionResult::StateChangeDuringStaticCall)
        );
        assert_eq!(
            interpreter.gas.total_gas_spent(),
            eip7819::EMPTY_ACCOUNT_COST
        );
        assert!(enabled.set_delegate_calls.is_empty());

        let mut interpreter = setdelegate_interpreter(
            eip7819::EMPTY_ACCOUNT_COST - 1,
            TARGET.into_word().into(),
            U256::ZERO,
        );
        assert_eq!(
            step_setdelegate(&mut interpreter, &mut enabled),
            Err(InstructionResult::OutOfGas)
        );
        assert_eq!(interpreter.stack.len(), 2, "out of gas popped operands");
        assert!(enabled.set_delegate_calls.is_empty());
    }

    #[test]
    fn setdelegate_collision_halts_after_popping_operands() {
        let mut host = DummyHost::default();
        host.enable_eip7819 = true;
        host.set_delegate_result = Some(Err(SetDelegateError::AddressCollision));
        let mut interpreter = setdelegate_interpreter(
            eip7819::EMPTY_ACCOUNT_COST,
            TARGET.into_word().into(),
            U256::ZERO,
        );

        assert_eq!(
            step_setdelegate(&mut interpreter, &mut host),
            Err(InstructionResult::AddressCollision)
        );
        assert!(interpreter.stack.data().is_empty());
        assert_eq!(interpreter.gas.refunded(), 0);
    }

    #[test]
    fn setselfdelegate_activation_requires_opt_in_and_prague() {
        let mut disabled = DummyHost::new(SpecId::PRAGUE);
        let mut interpreter = setselfdelegate_interpreter(
            eip7851::SETSELFDELEGATE_GAS,
            TARGET.into_word().into(),
            SpecId::PRAGUE,
        );
        assert_eq!(
            step_setdelegate(&mut interpreter, &mut disabled),
            Err(InstructionResult::NotActivated)
        );
        assert_eq!(
            interpreter.gas.total_gas_spent(),
            eip7851::SETSELFDELEGATE_GAS
        );
        assert_eq!(interpreter.stack.len(), 1);

        let mut pre_prague = DummyHost::new(SpecId::CANCUN);
        pre_prague.enable_eip7851 = true;
        let mut interpreter = setselfdelegate_interpreter(
            eip7851::SETSELFDELEGATE_GAS,
            TARGET.into_word().into(),
            SpecId::CANCUN,
        );
        assert_eq!(
            step_setdelegate(&mut interpreter, &mut pre_prague),
            Err(InstructionResult::NotActivated)
        );
        assert!(pre_prague.set_self_delegate_calls.is_empty());
    }

    #[test]
    fn setselfdelegate_charges_9500_before_oog_and_static_halt() {
        let mut host = DummyHost::new(SpecId::PRAGUE);
        host.enable_eip7851 = true;
        host.set_self_delegate_result = Some(true);

        let mut oog = setselfdelegate_interpreter(
            eip7851::SETSELFDELEGATE_GAS - 1,
            TARGET.into_word().into(),
            SpecId::PRAGUE,
        );
        assert_eq!(
            step_setdelegate(&mut oog, &mut host),
            Err(InstructionResult::OutOfGas)
        );
        assert_eq!(oog.stack.len(), 1);
        assert!(host.set_self_delegate_calls.is_empty());

        let mut exact = setselfdelegate_interpreter(
            eip7851::SETSELFDELEGATE_GAS,
            TARGET.into_word().into(),
            SpecId::PRAGUE,
        );
        step_setdelegate(&mut exact, &mut host).unwrap();
        assert_eq!(exact.stack.data(), &[U256::from(1)]);
        assert_eq!(exact.gas.total_gas_spent(), eip7851::SETSELFDELEGATE_GAS);
        assert_eq!(host.set_self_delegate_calls, [(EXECUTION_ADDRESS, TARGET)]);

        host.set_self_delegate_calls.clear();
        let mut static_call = setselfdelegate_interpreter(
            eip7851::SETSELFDELEGATE_GAS,
            TARGET.into_word().into(),
            SpecId::PRAGUE,
        );
        static_call.runtime_flag.is_static = true;
        assert_eq!(
            step_setdelegate(&mut static_call, &mut host),
            Err(InstructionResult::StateChangeDuringStaticCall)
        );
        assert_eq!(static_call.stack.len(), 1);
        assert_eq!(
            static_call.gas.total_gas_spent(),
            eip7851::SETSELFDELEGATE_GAS
        );
        assert!(host.set_self_delegate_calls.is_empty());
    }

    #[test]
    fn setselfdelegate_uses_context_authority_truncates_low160_and_pushes_status() {
        let target_word = (U256::from(0xabu64) << 160) | U256::from_be_slice(TARGET.as_slice());
        let mut host = DummyHost::new(SpecId::PRAGUE);
        host.enable_eip7851 = true;
        host.set_self_delegate_result = Some(false);
        let mut interpreter =
            setselfdelegate_interpreter(eip7851::SETSELFDELEGATE_GAS, target_word, SpecId::PRAGUE);

        step_setdelegate(&mut interpreter, &mut host).unwrap();
        assert_eq!(host.set_self_delegate_calls, [(EXECUTION_ADDRESS, TARGET)]);
        assert_eq!(interpreter.stack.data(), &[U256::ZERO]);

        host.set_self_delegate_calls.clear();
        host.set_self_delegate_result = Some(true);
        let mut zero =
            setselfdelegate_interpreter(eip7851::SETSELFDELEGATE_GAS, U256::ZERO, SpecId::PRAGUE);
        step_setdelegate(&mut zero, &mut host).unwrap();
        assert_eq!(zero.stack.data(), &[U256::ZERO]);
        assert!(host.set_self_delegate_calls.is_empty());
    }
}
