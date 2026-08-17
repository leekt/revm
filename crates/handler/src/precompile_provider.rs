use auto_impl::auto_impl;
use context::{Cfg, LocalContextTr};
use context_interface::{
    cfg::gas::{COLD_ACCOUNT_ACCESS_COST, WARM_STORAGE_READ_COST},
    host::LoadError,
    ContextTr, JournalTr,
};
use interpreter::{CallInputs, Gas, InstructionResult, InterpreterResult};
use precompile::{
    secp256k1::is_ecrecover_code_eligible, PrecompileHalt, PrecompileId, PrecompileOutput,
    PrecompileSpecId, PrecompileStatus, Precompiles,
};
use primitives::{hardfork::SpecId, Address, AddressSet, Bytes};
use std::string::{String, ToString};

/// Provider for precompiled contracts in the EVM.
#[auto_impl(&mut, Box)]
pub trait PrecompileProvider<CTX: ContextTr> {
    /// The output type returned by precompile execution.
    type Output;

    /// Sets the spec id and returns true if the spec id was changed. Initial call to set_spec will always return true.
    ///
    /// Returns `true` if precompile addresses should be injected into the journal.
    fn set_spec(&mut self, spec: <CTX::Cfg as Cfg>::Spec) -> bool;

    /// Run the precompile.
    fn run(
        &mut self,
        context: &mut CTX,
        inputs: &CallInputs,
    ) -> Result<Option<Self::Output>, String>;

    /// Get the warm addresses.
    fn warm_addresses(&self) -> &AddressSet;

    /// Check if the address is a precompile.
    fn contains(&self, address: &Address) -> bool {
        self.warm_addresses().contains(address)
    }
}

/// The [`PrecompileProvider`] for ethereum precompiles.
#[derive(Debug)]
pub struct EthPrecompiles {
    /// Contains precompiles for the current spec.
    pub precompiles: &'static Precompiles,
    /// Current spec. None means that spec was not set yet.
    pub spec: SpecId,
}

impl EthPrecompiles {
    /// Create a new precompile provider with the given spec.
    pub fn new(spec: SpecId) -> Self {
        Self {
            precompiles: Precompiles::new(PrecompileSpecId::from_spec_id(spec)),
            spec,
        }
    }

    /// Returns addresses of the precompiles.
    pub const fn warm_addresses(&self) -> &AddressSet {
        self.precompiles.addresses_set()
    }

    /// Returns whether the address is a precompile.
    pub fn contains(&self, address: &Address) -> bool {
        self.precompiles.contains(address)
    }
}

impl Clone for EthPrecompiles {
    fn clone(&self) -> Self {
        Self {
            precompiles: self.precompiles,
            spec: self.spec,
        }
    }
}

/// Converts a [`PrecompileOutput`] into an [`InterpreterResult`] for a call frame
/// with `gas_limit` regular gas.
///
/// Maps precompile status to the corresponding instruction result:
/// - `Success` -> [`InstructionResult::Return`]
/// - `Revert` -> [`InstructionResult::Revert`]
/// - `Halt(OOG)` -> [`InstructionResult::PrecompileOOG`]
/// - `Halt(other)` -> [`InstructionResult::PrecompileError`]
///
/// A precompile that reports more gas than it was given is downgraded to
/// [`InstructionResult::PrecompileOOG`]. Anything but a success or revert consumes
/// all regular gas and returns no output bytes.
pub fn precompile_output_to_interpreter_result(
    output: PrecompileOutput,
    gas_limit: u64,
) -> InterpreterResult {
    // A precompile lying about its usage must not leave the frame with gas it
    // never had: charging more regular gas than the limit is an OOG halt.
    let result = if output.gas_used > gas_limit {
        InstructionResult::PrecompileOOG
    } else {
        match &output.status {
            PrecompileStatus::Success => InstructionResult::Return,
            PrecompileStatus::Revert => InstructionResult::Revert,
            PrecompileStatus::Halt(reason) if reason.is_oog() => InstructionResult::PrecompileOOG,
            PrecompileStatus::Halt(_) => InstructionResult::PrecompileError,
        }
    };

    // Gas used, refund, state gas (with its spilled portion, so a later rollback
    // credits it back to regular gas per EIP-8037) and the reservoir all come from
    // the precompile's own accounting.
    let mut gas = Gas::new(gas_limit);
    *gas.tracker_mut() = output.to_gas_tracker(gas_limit);

    // Only a success or revert returns output bytes and keeps its unspent gas.
    if result.is_halt() {
        gas.spend_all();
        return InterpreterResult::new(result, Bytes::new(), gas);
    }

    InterpreterResult::new(result, output.bytes, gas)
}

impl<CTX: ContextTr> PrecompileProvider<CTX> for EthPrecompiles {
    type Output = InterpreterResult;

    fn set_spec(&mut self, spec: <CTX::Cfg as Cfg>::Spec) -> bool {
        let spec = spec.into();
        // generate new precompiles only on new spec
        if spec == self.spec {
            return false;
        }
        self.precompiles = Precompiles::new(PrecompileSpecId::from_spec_id(spec));
        self.spec = spec;
        true
    }

    fn run(
        &mut self,
        context: &mut CTX,
        inputs: &CallInputs,
    ) -> Result<Option<InterpreterResult>, String> {
        let Some(precompile) = self.precompiles.get(&inputs.bytecode_address) else {
            return Ok(None);
        };

        let mut output = precompile
            .execute(
                &inputs.input.as_bytes(context),
                inputs.gas_limit,
                inputs.reservoir,
            )
            .map_err(|e| e.to_string())?;

        let is_eip8151_enabled = context.cfg().is_eip8151_enabled()
            && context.cfg().spec().into().is_enabled_in(SpecId::PRAGUE);
        if is_eip8151_enabled
            && matches!(precompile.id(), PrecompileId::EcRec)
            && output.is_success()
        {
            if output.bytes.is_empty() {
                output.bytes = Bytes::from_static(&[0; 32]);
            } else if output.bytes.len() == 32 {
                let recovered_address = Address::from_slice(&output.bytes[12..]);
                let warm_gas_used = output.gas_used + WARM_STORAGE_READ_COST;
                let cold_gas_used = output.gas_used + COLD_ACCOUNT_ACCESS_COST;

                if inputs.gas_limit < warm_gas_used {
                    output = PrecompileOutput::halt(PrecompileHalt::OutOfGas, inputs.reservoir);
                } else {
                    let account = match context.load_account_info_skip_cold_load(
                        recovered_address,
                        true,
                        inputs.gas_limit < cold_gas_used,
                    ) {
                        Ok(account) => account,
                        Err(LoadError::ColdLoadSkipped) => {
                            output =
                                PrecompileOutput::halt(PrecompileHalt::OutOfGas, inputs.reservoir);
                            return Ok(Some(precompile_output_to_interpreter_result(
                                output,
                                inputs.gas_limit,
                            )));
                        }
                        Err(LoadError::DBError) => {
                            return Ok(Some(InterpreterResult::new(
                                InstructionResult::FatalExternalError,
                                Bytes::new(),
                                Gas::new_with_regular_gas_and_reservoir(
                                    inputs.gas_limit,
                                    inputs.reservoir,
                                ),
                            )));
                        }
                    };

                    output.gas_used = if account.is_cold {
                        cold_gas_used
                    } else {
                        warm_gas_used
                    };
                    let raw_code = account
                        .code
                        .as_ref()
                        .map_or(&[][..], |code| code.original_byte_slice());
                    if !is_ecrecover_code_eligible(raw_code) {
                        output.bytes = Bytes::from_static(&[0; 32]);
                    }
                }
            }
        }

        // If this is a top-level precompile call (depth == 1), persist the error message
        // into the local context so it can be returned as output in the final result.
        // Only do this for non-OOG halt errors.
        if let Some(halt_reason) = output.halt_reason() {
            if !halt_reason.is_oog() && context.journal().depth() == 1 {
                context
                    .local_mut()
                    .set_precompile_error_context(halt_reason.to_string());
            }
        }

        let result = precompile_output_to_interpreter_result(output, inputs.gas_limit);
        Ok(Some(result))
    }

    fn warm_addresses(&self) -> &AddressSet {
        Self::warm_addresses(self)
    }

    fn contains(&self, address: &Address) -> bool {
        Self::contains(self, address)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instructions::EthInstructions, ExecuteEvm, MainContext};
    use bytecode::Bytecode;
    use context::{BlockEnv, CfgEnv, Context, Evm, FrameStack, TxEnv};
    use context_interface::{
        result::{EVMError, ExecutionResult, HaltReason, OutOfGasError},
        DBErrorMarker, Database,
    };
    use database::InMemoryDB;
    use interpreter::{
        interpreter::EthInterpreter, CallInput, CallScheme, CallValue, InstructionResult,
    };
    use primitives::{
        address, bytes, hardfork::SpecId, AddressMap, HashSet, StorageKey, TxKind, B256, U256,
    };
    use state::AccountInfo;
    use std::string::String;

    /// Test-only address that hosts an over-spending precompile.
    const OVERSPEND_PRECOMPILE: Address = address!("0000000000000000000000000000000000000100");
    const ECRECOVER_ADDRESS: Address = address!("0000000000000000000000000000000000000001");
    const RECOVERED_ADDRESS: Address = address!("7e5f4552091a69125d5dfcb7b8c2659029395bdf");
    const DELEGATE_ADDRESS: Address = address!("2222222222222222222222222222222222222222");

    type Eip8151Context = Context<BlockEnv, TxEnv, CfgEnv, InMemoryDB>;

    /// Deterministic private key 1 signing the 32-byte hash `0x11..11`.
    fn valid_ecrecover_input() -> Bytes {
        bytes!(
            "1111111111111111111111111111111111111111111111111111111111111111\
                000000000000000000000000000000000000000000000000000000000000001c\
                e7c93726a865578504442b1a6827f676e0ed74bdff2be3960d1e253bbcfc4462\
                6aa772b878bc912bdbb33a0014ec507c4b3896ea85aa914b74dee9b7ac3e56da"
        )
    }

    fn recovered_output() -> Bytes {
        let mut output = [0; 32];
        output[12..].copy_from_slice(RECOVERED_ADDRESS.as_slice());
        Bytes::copy_from_slice(&output)
    }

    fn invalid_ecrecover_input() -> Bytes {
        let mut input = [0; 128];
        input[63] = 27;
        Bytes::copy_from_slice(&input)
    }

    fn ecrecover_call(input: Bytes, gas_limit: u64) -> CallInputs {
        CallInputs {
            input: CallInput::Bytes(input),
            return_memory_offset: 0..32,
            gas_limit,
            reservoir: 0,
            bytecode_address: ECRECOVER_ADDRESS,
            known_bytecode: (B256::ZERO, Bytecode::new()),
            target_address: ECRECOVER_ADDRESS,
            caller: Address::ZERO,
            value: CallValue::default(),
            scheme: CallScheme::Call,
            is_static: false,
            charged_new_account_state_gas: false,
        }
    }

    fn eip8151_context(spec: SpecId, enabled: bool, db: InMemoryDB) -> Eip8151Context {
        let mut context: Eip8151Context = Context::new(db, spec);
        context.cfg.enable_eip8151 = enabled;
        context
    }

    fn run_ecrecover<CTX: ContextTr>(
        context: &mut CTX,
        precompiles: &mut EthPrecompiles,
        input: Bytes,
        gas_limit: u64,
    ) -> InterpreterResult {
        <EthPrecompiles as PrecompileProvider<CTX>>::run(
            precompiles,
            context,
            &ecrecover_call(input, gas_limit),
        )
        .expect("precompile provider error")
        .expect("ecrecover must be registered")
    }

    fn insert_code(db: &mut InMemoryDB, address: Address, code: Bytecode) {
        db.insert_account_info(
            address,
            AccountInfo {
                code_hash: code.hash_slow(),
                code: Some(code),
                ..Default::default()
            },
        );
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct TestDbError;

    impl core::fmt::Display for TestDbError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("recovered account read failed")
        }
    }

    impl core::error::Error for TestDbError {}
    impl DBErrorMarker for TestDbError {}

    #[derive(Debug)]
    struct FailingRecoveredDb {
        fail_address: Option<Address>,
        basic_calls: usize,
    }

    impl FailingRecoveredDb {
        fn always() -> Self {
            Self {
                fail_address: None,
                basic_calls: 0,
            }
        }

        fn recovered_only() -> Self {
            Self {
                fail_address: Some(RECOVERED_ADDRESS),
                basic_calls: 0,
            }
        }
    }

    impl Database for FailingRecoveredDb {
        type Error = TestDbError;

        fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
            self.basic_calls += 1;
            if self.fail_address.is_none_or(|failed| address == failed) {
                Err(TestDbError)
            } else {
                Ok(None)
            }
        }

        fn code_by_hash(&mut self, _code_hash: B256) -> Result<Bytecode, Self::Error> {
            Ok(Bytecode::new())
        }

        fn storage(&mut self, _address: Address, _index: StorageKey) -> Result<U256, Self::Error> {
            Ok(U256::ZERO)
        }

        fn block_hash(&mut self, _number: u64) -> Result<B256, Self::Error> {
            Ok(B256::ZERO)
        }
    }

    /// Stateful expectations are pinned to
    /// ethereum/EIPs@bf7a4067f263bf7ce01c1511de48473e281d885d.
    #[test]
    fn eip8151_invalid_recovery_keeps_legacy_gas_and_skips_state() {
        let mut context = eip8151_context(SpecId::PRAGUE, true, InMemoryDB::default());
        let mut precompiles = EthPrecompiles::new(SpecId::PRAGUE);

        let below_base = run_ecrecover(
            &mut context,
            &mut precompiles,
            invalid_ecrecover_input(),
            2_999,
        );
        assert_eq!(below_base.result, InstructionResult::PrecompileOOG);
        assert_eq!(below_base.gas.total_gas_spent(), 2_999);
        assert!(below_base.output.is_empty());
        assert!(context.journal().evm_state().is_empty());

        let exact_base = run_ecrecover(
            &mut context,
            &mut precompiles,
            invalid_ecrecover_input(),
            3_000,
        );
        assert_eq!(exact_base.result, InstructionResult::Return);
        assert_eq!(exact_base.gas.total_gas_spent(), 3_000);
        assert_eq!(exact_base.output, Bytes::from_static(&[0; 32]));
        assert!(context.journal().evm_state().is_empty());
    }

    #[test]
    fn eip8151_warm_and_cold_access_gas_boundaries() {
        let mut warm_db = InMemoryDB::default();
        warm_db.insert_account_info(RECOVERED_ADDRESS, AccountInfo::default());
        let mut warm_context = eip8151_context(SpecId::PRAGUE, true, warm_db);
        let mut access_list = AddressMap::default();
        access_list.insert(RECOVERED_ADDRESS, HashSet::default());
        warm_context.journal_mut().warm_access_list(access_list);
        let mut precompiles = EthPrecompiles::new(SpecId::PRAGUE);

        let warm_oog = run_ecrecover(
            &mut warm_context,
            &mut precompiles,
            valid_ecrecover_input(),
            3_099,
        );
        assert_eq!(warm_oog.result, InstructionResult::PrecompileOOG);
        assert_eq!(warm_oog.gas.total_gas_spent(), 3_099);

        let warm = run_ecrecover(
            &mut warm_context,
            &mut precompiles,
            valid_ecrecover_input(),
            3_100,
        );
        assert_eq!(warm.result, InstructionResult::Return);
        assert_eq!(warm.gas.total_gas_spent(), 3_100);
        assert_eq!(warm.output, recovered_output());

        let mut cold_context = eip8151_context(SpecId::PRAGUE, true, InMemoryDB::default());
        let cold_oog = run_ecrecover(
            &mut cold_context,
            &mut precompiles,
            valid_ecrecover_input(),
            5_599,
        );
        assert_eq!(cold_oog.result, InstructionResult::PrecompileOOG);
        assert_eq!(cold_oog.gas.total_gas_spent(), 5_599);
        assert!(
            !cold_context
                .journal()
                .evm_state()
                .contains_key(&RECOVERED_ADDRESS),
            "an unaffordable cold access must not query or warm the account"
        );

        let cold = run_ecrecover(
            &mut cold_context,
            &mut precompiles,
            valid_ecrecover_input(),
            5_600,
        );
        assert_eq!(cold.result, InstructionResult::Return);
        assert_eq!(cold.gas.total_gas_spent(), 5_600);
        assert_eq!(cold.output, recovered_output());
    }

    #[test]
    fn eip8151_flag_and_prague_gate_preserve_legacy_ecrecover() {
        for (case, spec, enabled) in [
            ("default off", SpecId::PRAGUE, false),
            ("pre-Prague", SpecId::CANCUN, true),
        ] {
            let mut db = InMemoryDB::default();
            insert_code(
                &mut db,
                RECOVERED_ADDRESS,
                Bytecode::new_legacy(bytes!("00")),
            );
            let mut context = eip8151_context(spec, enabled, db);
            let mut precompiles = EthPrecompiles::new(spec);

            let invalid = run_ecrecover(
                &mut context,
                &mut precompiles,
                invalid_ecrecover_input(),
                3_000,
            );
            assert_eq!(invalid.result, InstructionResult::Return, "{case}");
            assert_eq!(invalid.gas.total_gas_spent(), 3_000, "{case}");
            assert!(invalid.output.is_empty(), "{case}");

            let result = run_ecrecover(
                &mut context,
                &mut precompiles,
                valid_ecrecover_input(),
                3_000,
            );

            assert_eq!(result.result, InstructionResult::Return, "{case}");
            assert_eq!(result.gas.total_gas_spent(), 3_000, "{case}");
            assert_eq!(result.output, recovered_output(), "{case}");
            assert!(
                !context
                    .journal()
                    .evm_state()
                    .contains_key(&RECOVERED_ADDRESS),
                "{case}: disabled restriction must remain pure"
            );
        }
    }

    #[test]
    fn eip8151_checks_exact_raw_code_without_changing_call_success() {
        let mut short = vec![0xef, 0x01, 0x00];
        short.extend_from_slice(&[0x44; 19]);
        let mut long = vec![0xef, 0x01, 0x00];
        long.extend_from_slice(&[0x44; 21]);
        let mut trailing = Bytecode::new_eip7702(DELEGATE_ADDRESS)
            .original_bytes()
            .to_vec();
        trailing.push(0);

        let cases = [
            ("absent", None, true),
            ("empty", Some(Bytecode::new()), true),
            (
                "ordinary code",
                Some(Bytecode::new_legacy(bytes!("00"))),
                false,
            ),
            (
                "ef0100 zero delegate",
                Some(Bytecode::new_eip7702(Address::ZERO)),
                true,
            ),
            (
                "ef0100 nonzero delegate",
                Some(Bytecode::new_eip7702(DELEGATE_ADDRESS)),
                true,
            ),
            (
                "exact ef0101",
                Some(Bytecode::new_eip7851(DELEGATE_ADDRESS)),
                false,
            ),
            (
                "short ef0100",
                Some(Bytecode::new_legacy(short.into())),
                false,
            ),
            (
                "long ef0100",
                Some(Bytecode::new_legacy(long.into())),
                false,
            ),
            (
                "trailing ef0100",
                Some(Bytecode::new_legacy(trailing.into())),
                false,
            ),
        ];

        for (case, code, allowed) in cases {
            let mut db = InMemoryDB::default();
            if let Some(code) = code {
                insert_code(&mut db, RECOVERED_ADDRESS, code);
            }
            let mut context = eip8151_context(SpecId::PRAGUE, true, db);
            let mut precompiles = EthPrecompiles::new(SpecId::PRAGUE);

            let result = run_ecrecover(
                &mut context,
                &mut precompiles,
                valid_ecrecover_input(),
                5_600,
            );

            assert_eq!(result.result, InstructionResult::Return, "{case}");
            assert_eq!(result.gas.total_gas_spent(), 5_600, "{case}");
            let expected = if allowed {
                recovered_output()
            } else {
                Bytes::from_static(&[0; 32])
            };
            assert_eq!(result.output, expected, "{case}");
        }
    }

    #[test]
    fn eip8151_rejected_account_is_warm_on_repeat() {
        let mut db = InMemoryDB::default();
        insert_code(
            &mut db,
            RECOVERED_ADDRESS,
            Bytecode::new_legacy(bytes!("00")),
        );
        let mut context = eip8151_context(SpecId::PRAGUE, true, db);
        let mut precompiles = EthPrecompiles::new(SpecId::PRAGUE);

        let cold = run_ecrecover(
            &mut context,
            &mut precompiles,
            valid_ecrecover_input(),
            5_600,
        );
        let warm = run_ecrecover(
            &mut context,
            &mut precompiles,
            valid_ecrecover_input(),
            3_100,
        );

        assert_eq!(cold.result, InstructionResult::Return);
        assert_eq!(cold.gas.total_gas_spent(), 5_600);
        assert_eq!(cold.output, Bytes::from_static(&[0; 32]));
        assert_eq!(warm.result, InstructionResult::Return);
        assert_eq!(warm.gas.total_gas_spent(), 3_100);
        assert_eq!(warm.output, Bytes::from_static(&[0; 32]));
    }

    #[test]
    fn eip8151_does_not_follow_or_warm_delegation_target() {
        let mut db = InMemoryDB::default();
        insert_code(
            &mut db,
            RECOVERED_ADDRESS,
            Bytecode::new_eip7702(DELEGATE_ADDRESS),
        );
        insert_code(
            &mut db,
            DELEGATE_ADDRESS,
            Bytecode::new_legacy(bytes!("00")),
        );
        let mut context = eip8151_context(SpecId::PRAGUE, true, db);
        let mut precompiles = EthPrecompiles::new(SpecId::PRAGUE);

        let result = run_ecrecover(
            &mut context,
            &mut precompiles,
            valid_ecrecover_input(),
            5_600,
        );

        assert_eq!(result.output, recovered_output());
        assert!(context
            .journal()
            .evm_state()
            .contains_key(&RECOVERED_ADDRESS));
        assert!(!context
            .journal()
            .evm_state()
            .contains_key(&DELEGATE_ADDRESS));
    }

    #[test]
    fn eip8151_reverts_recovered_account_warmth_with_checkpoint() {
        let mut context = eip8151_context(SpecId::PRAGUE, true, InMemoryDB::default());
        let mut precompiles = EthPrecompiles::new(SpecId::PRAGUE);
        let checkpoint = context.journal_mut().checkpoint();

        let first = run_ecrecover(
            &mut context,
            &mut precompiles,
            valid_ecrecover_input(),
            5_600,
        );
        assert_eq!(first.result, InstructionResult::Return);
        context.journal_mut().checkpoint_revert(checkpoint);

        let after_revert = run_ecrecover(
            &mut context,
            &mut precompiles,
            valid_ecrecover_input(),
            5_599,
        );
        assert_eq!(after_revert.result, InstructionResult::PrecompileOOG);
        assert_eq!(after_revert.gas.total_gas_spent(), 5_599);
    }

    #[test]
    fn eip8151_database_error_occurs_only_after_successful_recovery() {
        type FailingContext = Context<BlockEnv, TxEnv, CfgEnv, FailingRecoveredDb>;

        let mut invalid_context: FailingContext =
            Context::new(FailingRecoveredDb::always(), SpecId::PRAGUE);
        invalid_context.cfg.enable_eip8151 = true;
        let mut precompiles = EthPrecompiles::new(SpecId::PRAGUE);
        let invalid = run_ecrecover(
            &mut invalid_context,
            &mut precompiles,
            invalid_ecrecover_input(),
            3_000,
        );
        assert_eq!(invalid.result, InstructionResult::Return);
        assert_eq!(invalid.output, Bytes::from_static(&[0; 32]));
        assert_eq!(invalid_context.error, Ok(()));
        assert_eq!(invalid_context.db().basic_calls, 0);

        let mut context: FailingContext =
            Context::new(FailingRecoveredDb::recovered_only(), SpecId::PRAGUE);
        context.cfg.enable_eip8151 = true;
        let mut evm = Evm {
            ctx: context,
            inspector: (),
            instruction: EthInstructions::<EthInterpreter, _>::new_mainnet_with_spec(
                SpecId::PRAGUE,
            ),
            precompiles: EthPrecompiles::new(SpecId::PRAGUE),
            frame_stack: FrameStack::new_prealloc(8),
            #[cfg(feature = "asyncdb")]
            async_stack: database_interface::async_db::FiberStack::default(),
        };
        let tx = TxEnv::builder()
            .caller(Address::repeat_byte(0xaa))
            .kind(TxKind::Call(ECRECOVER_ADDRESS))
            .data(valid_ecrecover_input())
            .gas_limit(100_000)
            .build()
            .unwrap();

        let error = evm.transact_one(tx).unwrap_err();
        assert_eq!(error, EVMError::Database(TestDbError));
    }

    /// Custom precompile provider that drives the bug path: it returns a
    /// `PrecompileOutput` with `status = Success` and `gas_used = u64::MAX` while
    /// `gas_limit` is finite. Without the fix, `record_regular_cost`'s `false` return
    /// is discarded so the call lands as `Return` with the gas tracker untouched —
    /// the transaction succeeds and refunds the precompile's "free" gas. With the fix,
    /// the helper converts the over-spend into `PrecompileOOG`, halting the tx.
    #[derive(Debug)]
    struct OverspendingPrecompiles {
        inner: EthPrecompiles,
        warm: AddressSet,
    }

    impl OverspendingPrecompiles {
        fn new(spec: SpecId) -> Self {
            let inner = EthPrecompiles::new(spec);
            let mut warm = AddressSet::default();
            warm.clone_from(inner.warm_addresses());
            warm.insert(OVERSPEND_PRECOMPILE);
            Self { inner, warm }
        }
    }

    impl<CTX> PrecompileProvider<CTX> for OverspendingPrecompiles
    where
        CTX: ContextTr<Cfg: Cfg<Spec = SpecId>>,
    {
        type Output = InterpreterResult;

        fn set_spec(&mut self, spec: <CTX::Cfg as Cfg>::Spec) -> bool {
            let changed =
                <EthPrecompiles as PrecompileProvider<CTX>>::set_spec(&mut self.inner, spec);
            self.warm.clone_from(self.inner.warm_addresses());
            self.warm.insert(OVERSPEND_PRECOMPILE);
            changed
        }

        fn run(
            &mut self,
            context: &mut CTX,
            inputs: &CallInputs,
        ) -> Result<Option<Self::Output>, String> {
            if inputs.bytecode_address == OVERSPEND_PRECOMPILE {
                let output = PrecompileOutput {
                    status: PrecompileStatus::Success,
                    gas_used: u64::MAX,
                    gas_refunded: 0,
                    state_gas_used: 0,
                    state_gas_spilled: 0,
                    reservoir: inputs.reservoir,
                    bytes: Bytes::from_static(b"unreliable"),
                };
                return Ok(Some(precompile_output_to_interpreter_result(
                    output,
                    inputs.gas_limit,
                )));
            }
            <EthPrecompiles as PrecompileProvider<CTX>>::run(&mut self.inner, context, inputs)
        }

        fn warm_addresses(&self) -> &AddressSet {
            &self.warm
        }
    }

    /// The spilled portion of a precompile's state gas must reach the frame's gas
    /// tracker, otherwise a rollback credits it to the reservoir instead of regular
    /// gas (EIP-8037).
    #[test]
    fn precompile_output_propagates_spilled_state_gas() {
        let output = PrecompileOutput {
            status: PrecompileStatus::Success,
            // 10 regular + 30 state gas, of which 20 spilled out of the 10 gas reservoir
            gas_used: 40,
            gas_refunded: 0,
            state_gas_used: 30,
            state_gas_spilled: 20,
            reservoir: 0,
            bytes: Bytes::new(),
        };
        let mut result = precompile_output_to_interpreter_result(output, 100);

        assert_eq!(result.result, InstructionResult::Return);
        assert_eq!(result.gas.state_gas_spent(), 30);
        assert_eq!(result.gas.state_gas_spilled(), 20);
        assert_eq!(result.gas.remaining(), 60);

        // rollback returns the spilled part to regular gas and the rest to the reservoir
        result.gas.rollback_state_gas();
        assert_eq!(result.gas.remaining(), 80);
        assert_eq!(result.gas.reservoir(), 10);
        assert_eq!(result.gas.state_gas_spent(), 0);
        assert_eq!(result.gas.state_gas_spilled(), 0);
    }

    /// A precompile that reports more gas than its limit is turned into an OOG halt
    /// with all gas consumed and no output bytes.
    #[test]
    fn precompile_output_overspend_is_oog() {
        let output = PrecompileOutput::new(u64::MAX, Bytes::from_static(b"out"), 0);
        let result = precompile_output_to_interpreter_result(output, 100);
        assert_eq!(result.result, InstructionResult::PrecompileOOG);
        assert_eq!(result.gas.remaining(), 0);
        assert!(result.output.is_empty());
    }

    /// End-to-end regression test for Bug 3. A transaction targets a custom precompile
    /// that lies about its gas usage. The fix turns this into an `OutOfGas(Precompile)`
    /// halt; without the fix it is silently treated as a successful call.
    #[test]
    fn overspending_precompile_halts_tx_with_precompile_oog() {
        let caller = address!("0000000000000000000000000000000000000001");
        let mut db = InMemoryDB::default();
        db.insert_account_info(
            caller,
            AccountInfo {
                balance: U256::from(10).pow(U256::from(18)),
                ..Default::default()
            },
        );

        let spec = SpecId::default();
        let ctx = Context::mainnet().with_db(db);
        let mut evm = Evm {
            ctx,
            inspector: (),
            instruction: EthInstructions::<EthInterpreter, _>::new_mainnet_with_spec(spec),
            precompiles: OverspendingPrecompiles::new(spec),
            frame_stack: FrameStack::new_prealloc(8),
            #[cfg(feature = "asyncdb")]
            async_stack: database_interface::async_db::FiberStack::default(),
        };

        let tx = TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(OVERSPEND_PRECOMPILE))
            .gas_limit(100_000)
            .build()
            .unwrap();

        let exec = evm.transact_one(tx).expect("handler returned an error");

        match exec {
            ExecutionResult::Halt { reason, .. } => {
                assert_eq!(
                    reason,
                    HaltReason::OutOfGas(OutOfGasError::Precompile),
                    "expected precompile OOG halt for over-spending precompile",
                );
            }
            ExecutionResult::Success { .. } => panic!(
                "before-fix behavior leaked: over-spending precompile reported Success \
                 instead of halting with PrecompileOOG"
            ),
            ExecutionResult::Revert { .. } => panic!("expected Halt(PrecompileOOG), got Revert"),
        }
    }
}
