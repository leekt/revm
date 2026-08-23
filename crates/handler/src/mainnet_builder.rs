use crate::{frame::EthFrame, instructions::EthInstructions, EthPrecompiles};
use context::{BlockEnv, Cfg, CfgEnv, Context, Evm, FrameStack, Journal, TxEnv};
use context_interface::{Block, Database, JournalTr, Transaction};
use database_interface::EmptyDB;
use interpreter::interpreter::EthInterpreter;
use primitives::hardfork::SpecId;

/// Type alias for a mainnet EVM instance with standard Ethereum components.
pub type MainnetEvm<CTX, INSP = ()> =
    Evm<CTX, INSP, EthInstructions<EthInterpreter, CTX>, EthPrecompiles, EthFrame<EthInterpreter>>;

/// Type alias for a mainnet context with standard Ethereum environment types.
pub type MainnetContext<DB> = Context<BlockEnv, TxEnv, CfgEnv, DB, Journal<DB>, ()>;

/// Trait for building mainnet EVM instances from contexts.
pub trait MainBuilder: Sized {
    /// The context type that will be used in the EVM.
    type Context;

    /// Builds a mainnet EVM instance without an inspector.
    fn build_mainnet(self) -> MainnetEvm<Self::Context>;

    /// Builds a mainnet EVM instance with the provided inspector.
    fn build_mainnet_with_inspector<INSP>(self, inspector: INSP)
        -> MainnetEvm<Self::Context, INSP>;
}

impl<BLOCK, TX, CFG, DB, JOURNAL, CHAIN> MainBuilder for Context<BLOCK, TX, CFG, DB, JOURNAL, CHAIN>
where
    BLOCK: Block,
    TX: Transaction,
    CFG: Cfg,
    DB: Database,
    JOURNAL: JournalTr<Database = DB>,
{
    type Context = Self;

    fn build_mainnet(self) -> MainnetEvm<Self::Context> {
        let spec = self.cfg.spec().into();
        Evm {
            ctx: self,
            inspector: (),
            instruction: EthInstructions::new_mainnet_with_spec(spec),
            precompiles: EthPrecompiles::new(spec),
            frame_stack: FrameStack::new_prealloc(8),
            #[cfg(feature = "asyncdb")]
            async_stack: database_interface::async_db::FiberStack::default(),
        }
    }

    fn build_mainnet_with_inspector<INSP>(
        self,
        inspector: INSP,
    ) -> MainnetEvm<Self::Context, INSP> {
        let spec = self.cfg.spec().into();
        Evm {
            ctx: self,
            inspector,
            instruction: EthInstructions::new_mainnet_with_spec(spec),
            precompiles: EthPrecompiles::new(spec),
            frame_stack: FrameStack::new_prealloc(8),
            #[cfg(feature = "asyncdb")]
            async_stack: database_interface::async_db::FiberStack::default(),
        }
    }
}

/// Trait used to initialize Context with default mainnet types.
pub trait MainContext {
    /// Creates a new mainnet context with default configuration.
    fn mainnet() -> Self;
}

impl MainContext for Context<BlockEnv, TxEnv, CfgEnv, EmptyDB, Journal<EmptyDB>, ()> {
    fn mainnet() -> Self {
        Context::new(EmptyDB::new(), SpecId::default())
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use crate::{
        instructions::EthInstructions, EthFrame, EthPrecompiles, ExecuteEvm, Handler, MainBuilder,
        MainContext, MainnetHandler, PrecompileProvider,
    };
    use alloy_signer::{Either, SignerSync};
    use alloy_signer_local::PrivateKeySigner;
    use bytecode::{
        opcode::{
            CALL, CALLDATALOAD, CREATE, GAS, MSTORE, POP, PUSH0, PUSH1, PUSH20, SETDELEGATE,
            SSTORE, STOP,
        },
        Bytecode,
    };
    use context::{
        result::{EVMError, ExecutionResult, HaltReason, InvalidTransaction},
        Context, Evm, TxEnv,
    };
    use context_interface::{
        cfg::GasParams,
        context::{ContextError, ContextSetters, SStoreResult, SelfDestructResult, StateLoad},
        host::{FrameInfo, FrameTxContext, LoadError},
        journaled_state::AccountInfoLoad,
        transaction::{AccessList, AccessListItem, Authorization},
        Cfg, ContextTr, Database, Host, JournalTr,
    };
    use database::{BenchmarkDB, CacheDB, EmptyDB, EEADDRESS, FFADDRESS};
    use interpreter::{
        instructions::frame_tx::install_frame_tx_context, CallInputs, Gas, InstructionResult,
        InterpreterResult,
    };
    use primitives::{
        address, eip7819, hardfork::SpecId, Address, AddressSet, Bytes, Log, StorageKey,
        StorageValue, TxKind, B256, U256,
    };
    use state::AccountInfo;
    use std::{cell::Cell, string::String, sync::Arc};

    const STATIC_TEST_PRECOMPILE: Address = address!("0000000000000000000000000000000000000100");

    fn matching_frame(caller: Address, target: Address, gas_limit: u64) -> FrameInfo {
        FrameInfo {
            resolved_target: target,
            expected_caller: caller,
            gas_limit,
            ..Default::default()
        }
    }

    fn matching_context(sender: Address, frame: FrameInfo) -> FrameTxContext {
        FrameTxContext {
            sender,
            frames: vec![frame],
            ..Default::default()
        }
    }

    fn matching_tx(caller: Address, target: Address, gas_limit: u64) -> TxEnv {
        TxEnv::builder()
            .caller(caller)
            .kind(TxKind::Call(target))
            .gas_limit(gas_limit)
            .chain_id(Some(1))
            .gas_priority_fee(Some(0))
            .build()
            .unwrap()
    }

    fn begin_outer_frame_transaction<CTX: ContextTr>(context: &mut CTX, sender: Address) {
        assert!(context.journal_mut().begin_frame_transaction(sender));
    }

    struct NativeFrameContext<CTX> {
        inner: CTX,
        frame_tx: Arc<FrameTxContext>,
        frame_tx_after_first_read: Option<Arc<FrameTxContext>>,
        frame_context_reads: Cell<usize>,
    }

    impl<CTX> NativeFrameContext<CTX> {
        fn new(inner: CTX, frame_tx: FrameTxContext) -> Self {
            Self {
                inner,
                frame_tx: frame_tx.into_shared(),
                frame_tx_after_first_read: None,
                frame_context_reads: Cell::new(0),
            }
        }

        fn new_mutating(
            inner: CTX,
            frame_tx: FrameTxContext,
            frame_tx_after_first_read: FrameTxContext,
        ) -> Self {
            Self {
                inner,
                frame_tx: frame_tx.into_shared(),
                frame_tx_after_first_read: Some(frame_tx_after_first_read.into_shared()),
                frame_context_reads: Cell::new(0),
            }
        }
    }

    impl<CTX: ContextTr> ContextTr for NativeFrameContext<CTX> {
        type Block = CTX::Block;
        type Tx = CTX::Tx;
        type Cfg = CTX::Cfg;
        type Db = CTX::Db;
        type Journal = CTX::Journal;
        type Chain = CTX::Chain;
        type Local = CTX::Local;

        fn all(
            &self,
        ) -> (
            &Self::Block,
            &Self::Tx,
            &Self::Cfg,
            &Self::Db,
            &Self::Journal,
            &Self::Chain,
            &Self::Local,
        ) {
            self.inner.all()
        }

        fn all_mut(
            &mut self,
        ) -> (
            &Self::Block,
            &Self::Tx,
            &Self::Cfg,
            &mut Self::Journal,
            &mut Self::Chain,
            &mut Self::Local,
        ) {
            self.inner.all_mut()
        }

        fn error(&mut self) -> &mut Result<(), ContextError<<Self::Db as Database>::Error>> {
            self.inner.error()
        }
    }

    impl<CTX: ContextTr + ContextSetters> ContextSetters for NativeFrameContext<CTX> {
        fn set_tx(&mut self, tx: Self::Tx) {
            self.inner.set_tx(tx);
        }

        fn set_block(&mut self, block: Self::Block) {
            self.inner.set_block(block);
        }
    }

    impl<CTX: ContextTr> Host for NativeFrameContext<CTX> {
        fn basefee(&self) -> U256 {
            Host::basefee(&self.inner)
        }

        fn blob_gasprice(&self) -> U256 {
            Host::blob_gasprice(&self.inner)
        }

        fn gas_limit(&self) -> U256 {
            Host::gas_limit(&self.inner)
        }

        fn difficulty(&self) -> U256 {
            Host::difficulty(&self.inner)
        }

        fn prevrandao(&self) -> Option<U256> {
            Host::prevrandao(&self.inner)
        }

        fn block_number(&self) -> U256 {
            Host::block_number(&self.inner)
        }

        fn timestamp(&self) -> U256 {
            Host::timestamp(&self.inner)
        }

        fn beneficiary(&self) -> Address {
            Host::beneficiary(&self.inner)
        }

        fn slot_num(&self) -> U256 {
            Host::slot_num(&self.inner)
        }

        fn chain_id(&self) -> U256 {
            Host::chain_id(&self.inner)
        }

        fn effective_gas_price(&self) -> U256 {
            Host::effective_gas_price(&self.inner)
        }

        fn caller(&self) -> Address {
            Host::caller(&self.inner)
        }

        fn blob_hash(&self, number: usize) -> Option<U256> {
            Host::blob_hash(&self.inner, number)
        }

        fn max_initcode_size(&self) -> usize {
            Host::max_initcode_size(&self.inner)
        }

        fn gas_params(&self) -> &GasParams {
            Host::gas_params(&self.inner)
        }

        fn is_amsterdam_eip8037_enabled(&self) -> bool {
            Host::is_amsterdam_eip8037_enabled(&self.inner)
        }

        fn frame_context(&self) -> Option<Arc<FrameTxContext>> {
            let reads = self.frame_context_reads.get();
            self.frame_context_reads.set(reads + 1);
            if reads != 0 {
                if let Some(frame_tx) = &self.frame_tx_after_first_read {
                    return Some(frame_tx.clone());
                }
            }
            Some(self.frame_tx.clone())
        }

        fn block_hash(&mut self, number: u64) -> Option<B256> {
            Host::block_hash(&mut self.inner, number)
        }

        fn selfdestruct(
            &mut self,
            address: Address,
            target: Address,
            skip_cold_load: bool,
        ) -> Result<StateLoad<SelfDestructResult>, LoadError> {
            Host::selfdestruct(&mut self.inner, address, target, skip_cold_load)
        }

        fn log(&mut self, log: Log) {
            Host::log(&mut self.inner, log);
        }

        fn sstore_skip_cold_load(
            &mut self,
            address: Address,
            key: StorageKey,
            value: StorageValue,
            skip_cold_load: bool,
        ) -> Result<StateLoad<SStoreResult>, LoadError> {
            Host::sstore_skip_cold_load(&mut self.inner, address, key, value, skip_cold_load)
        }

        fn sload_skip_cold_load(
            &mut self,
            address: Address,
            key: StorageKey,
            skip_cold_load: bool,
        ) -> Result<StateLoad<StorageValue>, LoadError> {
            Host::sload_skip_cold_load(&mut self.inner, address, key, skip_cold_load)
        }

        fn tstore(&mut self, address: Address, key: StorageKey, value: StorageValue) {
            Host::tstore(&mut self.inner, address, key, value);
        }

        fn tload(&mut self, address: Address, key: StorageKey) -> StorageValue {
            Host::tload(&mut self.inner, address, key)
        }

        fn load_account_info_skip_cold_load(
            &mut self,
            address: Address,
            load_code: bool,
            skip_cold_load: bool,
        ) -> Result<AccountInfoLoad<'_>, LoadError> {
            Host::load_account_info_skip_cold_load(
                &mut self.inner,
                address,
                load_code,
                skip_cold_load,
            )
        }
    }

    #[derive(Debug)]
    struct StaticRecordingPrecompile {
        warm: AddressSet,
        saw_static: Option<bool>,
    }

    impl StaticRecordingPrecompile {
        fn new() -> Self {
            let mut warm = AddressSet::default();
            warm.insert(STATIC_TEST_PRECOMPILE);
            Self {
                warm,
                saw_static: None,
            }
        }
    }

    impl<CTX: ContextTr> PrecompileProvider<CTX> for StaticRecordingPrecompile {
        type Output = InterpreterResult;

        fn set_spec(&mut self, _spec: <CTX::Cfg as Cfg>::Spec) -> bool {
            false
        }

        fn run(
            &mut self,
            _context: &mut CTX,
            inputs: &CallInputs,
        ) -> Result<Option<Self::Output>, String> {
            if inputs.bytecode_address != STATIC_TEST_PRECOMPILE {
                return Ok(None);
            }
            self.saw_static = Some(inputs.is_static);
            Ok(Some(InterpreterResult::new(
                InstructionResult::Return,
                Bytes::new(),
                Gas::new(inputs.gas_limit),
            )))
        }

        fn warm_addresses(&self) -> &AddressSet {
            &self.warm
        }
    }

    #[test]
    fn sanity_eip7702_tx() {
        let signer = PrivateKeySigner::random();
        let auth = Authorization {
            chain_id: U256::ZERO,
            nonce: 0,
            address: FFADDRESS,
        };
        let signature = signer.sign_hash_sync(&auth.signature_hash()).unwrap();
        let auth = auth.into_signed(signature);

        let bytecode = Bytecode::new_legacy([PUSH1, 0x01, PUSH1, 0x01, SSTORE].into());

        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(BenchmarkDB::new_bytecode(bytecode));

        let mut evm = ctx.build_mainnet();

        let state = evm
            .transact(
                TxEnv::builder()
                    .gas_limit(100_000)
                    .authorization_list(vec![Either::Left(auth)])
                    .caller(EEADDRESS)
                    .kind(TxKind::Call(signer.address()))
                    .build()
                    .unwrap(),
            )
            .unwrap()
            .state;

        let auth_acc = state.get(&signer.address()).unwrap();
        assert_eq!(auth_acc.info.code, Some(Bytecode::new_eip7702(FFADDRESS)));
        assert_eq!(auth_acc.info.nonce, 1);
        assert_eq!(
            auth_acc
                .storage
                .get(&StorageKey::from(1))
                .unwrap()
                .present_value,
            StorageValue::from(1)
        );
    }

    #[test]
    fn setdelegate_is_effective_immediately_in_the_same_transaction() {
        let caller = address!("1000000000000000000000000000000000000001");
        let factory = address!("1000000000000000000000000000000000000002");
        let target = address!("1000000000000000000000000000000000000003");
        let location = eip7819::setdelegate_address(factory, U256::ZERO);
        let implementation =
            Bytecode::new_legacy([PUSH0, CALLDATALOAD, PUSH0, SSTORE, STOP].into());
        let mut factory_code = vec![PUSH20];
        factory_code.extend_from_slice(target.as_slice());
        factory_code.extend_from_slice(&[
            PUSH1,
            0x00,
            SETDELEGATE,
            POP,
            PUSH1,
            0x2a,
            PUSH0,
            MSTORE,
            PUSH0,
            PUSH0,
            PUSH1,
            0x20,
            PUSH0,
            PUSH0,
            PUSH20,
        ]);
        factory_code.extend_from_slice(location.as_slice());
        factory_code.extend_from_slice(&[GAS, CALL, POP, STOP]);
        let factory_code = Bytecode::new_legacy(factory_code.into());
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(
            caller,
            AccountInfo {
                balance: U256::from(1_000_000_000u64),
                ..Default::default()
            },
        );
        db.insert_account_info(
            factory,
            AccountInfo {
                nonce: 1,
                code_hash: factory_code.hash_slow(),
                code: Some(factory_code),
                ..Default::default()
            },
        );
        db.insert_account_info(
            target,
            AccountInfo {
                nonce: 1,
                code_hash: implementation.hash_slow(),
                code: Some(implementation),
                ..Default::default()
            },
        );
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| {
                cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE);
                cfg.enable_eip7819 = true;
            })
            .with_db(db);
        let mut evm = ctx.build_mainnet();

        let output = evm.transact(matching_tx(caller, factory, 200_000)).unwrap();

        assert!(matches!(output.result, ExecutionResult::Success { .. }));
        let delegated = &output.state[&location];
        assert_eq!(delegated.info.nonce, 1);
        assert_eq!(
            delegated.info.code.as_ref().unwrap().eip7702_address(),
            Some(target)
        );
        assert_eq!(
            delegated.storage[&StorageKey::ZERO].present_value,
            StorageValue::from(0x2au64),
            "the immediate call did not execute in the delegated account's storage context"
        );
        assert!(
            output.state[&target]
                .storage
                .get(&StorageKey::ZERO)
                .is_none_or(|slot| slot.present_value.is_zero()),
            "the implementation account's storage was modified"
        );
    }

    #[test]
    fn frame_tx_static_modes_halt_sstore_only_for_the_current_target() {
        let bytecode =
            Bytecode::new_legacy([PUSH1, 0x01, PUSH1, 0x01, SSTORE, STOP].as_slice().into());
        for (mode, must_be_static) in [(1, true), (3, true), (0, false), (2, false)] {
            let mut frame = matching_frame(EEADDRESS, FFADDRESS, 100_000);
            frame.mode = mode;
            let _guard = install_frame_tx_context(matching_context(EEADDRESS, frame));
            let ctx = Context::mainnet()
                .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
                .with_db(BenchmarkDB::new_bytecode(bytecode.clone()));
            let mut evm = ctx.build_mainnet();
            begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);
            let output = evm
                .transact(matching_tx(EEADDRESS, FFADDRESS, 100_000))
                .unwrap();

            if must_be_static {
                assert!(matches!(
                    output.result,
                    ExecutionResult::Halt {
                        reason: HaltReason::StateChangeDuringStaticCall,
                        ..
                    }
                ));
                assert_ne!(
                    output
                        .state
                        .get(&FFADDRESS)
                        .and_then(|account| account.storage.get(&StorageKey::from(1)))
                        .map(|slot| slot.present_value),
                    Some(StorageValue::from(1)),
                );
            } else {
                assert!(matches!(output.result, ExecutionResult::Success { .. }));
                assert_eq!(
                    output.state[&FFADDRESS].storage[&StorageKey::from(1)].present_value,
                    StorageValue::from(1),
                );
            }
            let _ = evm.ctx.journal_mut().finish_frame_transaction();
        }
    }

    #[test]
    fn frame_tx_forced_static_reaches_precompile_inputs() {
        let mut frame = matching_frame(EEADDRESS, STATIC_TEST_PRECOMPILE, 100_000);
        frame.mode = 1;
        let _guard = install_frame_tx_context(matching_context(EEADDRESS, frame));
        let spec = SpecId::PRAGUE;
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(spec))
            .with_db(BenchmarkDB::default());
        let mut evm = Evm::<_, (), _, _, EthFrame>::new(
            ctx,
            EthInstructions::new_mainnet_with_spec(spec),
            StaticRecordingPrecompile::new(),
        );
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);

        let output = evm
            .transact(matching_tx(EEADDRESS, STATIC_TEST_PRECOMPILE, 100_000))
            .unwrap();

        assert!(matches!(output.result, ExecutionResult::Success { .. }));
        assert_eq!(evm.precompiles.saw_static, Some(true));
        let _ = evm.ctx.journal_mut().finish_frame_transaction();
    }

    #[test]
    fn frame_tx_forced_static_rejects_value_before_transfer_or_precompile() {
        let mut frame = matching_frame(EEADDRESS, STATIC_TEST_PRECOMPILE, 100_000);
        frame.mode = 3;
        frame.value = U256::from(1u64);
        let _guard = install_frame_tx_context(matching_context(EEADDRESS, frame));
        let spec = SpecId::PRAGUE;
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(spec))
            .with_db(BenchmarkDB::default());
        let mut evm = Evm::<_, (), _, _, EthFrame>::new(
            ctx,
            EthInstructions::new_mainnet_with_spec(spec),
            StaticRecordingPrecompile::new(),
        );
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);

        let mut tx = matching_tx(EEADDRESS, STATIC_TEST_PRECOMPILE, 100_000);
        tx.value = U256::from(1u64);
        let output = evm.transact(tx).unwrap();

        assert!(matches!(
            output.result,
            ExecutionResult::Halt {
                reason: HaltReason::CallNotAllowedInsideStatic,
                ..
            }
        ));
        assert_eq!(evm.precompiles.saw_static, None);
        assert_eq!(
            output
                .state
                .get(&STATIC_TEST_PRECOMPILE)
                .map(|account| account.info.balance)
                .unwrap_or_default(),
            U256::ZERO
        );
        let _ = evm.ctx.journal_mut().finish_frame_transaction();
    }

    #[test]
    fn frame_context_cold_target_access_uses_exact_frame_gas_without_nonce_bump() {
        const COLD_TARGET_ACCESS: u64 = 2_600;
        for (bytecode, execution_gas) in [
            (Bytecode::default(), 0),
            (Bytecode::new_legacy([STOP].into()), 0),
            (Bytecode::new_legacy([PUSH1, 0x00, POP, STOP].into()), 5),
        ] {
            let gas_limit = COLD_TARGET_ACCESS + execution_gas;
            let _guard = install_frame_tx_context(matching_context(
                EEADDRESS,
                matching_frame(EEADDRESS, FFADDRESS, gas_limit),
            ));
            let ctx = Context::mainnet()
                .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
                .with_db(BenchmarkDB::new_bytecode(bytecode));
            let mut evm = ctx.build_mainnet();
            begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);

            let output = evm
                .transact(matching_tx(EEADDRESS, FFADDRESS, gas_limit))
                .unwrap();

            let ExecutionResult::Success { gas, .. } = output.result else {
                panic!("frame call should succeed with {gas_limit} gas")
            };
            assert_eq!(gas.total_gas_spent(), gas_limit);
            assert_eq!(gas.floor_gas(), 0);
            assert!(evm.ctx.journal().is_frame_transaction_active());
            let _ = evm.ctx.journal_mut().finish_frame_transaction();
        }
    }

    #[test]
    fn frame_context_cold_target_access_halts_below_exact_cost() {
        let gas_limit = 2_599;
        let _guard = install_frame_tx_context(matching_context(
            EEADDRESS,
            matching_frame(EEADDRESS, FFADDRESS, gas_limit),
        ));
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(BenchmarkDB::new_bytecode(Bytecode::new_legacy(
                [STOP].into(),
            )));
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);

        let output = evm
            .transact(matching_tx(EEADDRESS, FFADDRESS, gas_limit))
            .unwrap();

        assert!(matches!(
            &output.result,
            ExecutionResult::Halt {
                reason: HaltReason::OutOfGas(_),
                ..
            }
        ));
        assert_eq!(output.result.gas().total_gas_spent(), gas_limit);
        let (state, logs) = evm.ctx.journal_mut().finish_frame_transaction();
        assert!(state.is_empty());
        assert!(logs.is_empty());
    }

    #[test]
    fn frame_two_target_is_warm_after_frame_one_success() {
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(BenchmarkDB::new_bytecode(Bytecode::new_legacy(
                [STOP].into(),
            )));
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);

        let first = {
            let gas_limit = 2_600;
            let _guard = install_frame_tx_context(matching_context(
                EEADDRESS,
                matching_frame(EEADDRESS, FFADDRESS, gas_limit),
            ));
            evm.transact(matching_tx(EEADDRESS, FFADDRESS, gas_limit))
                .unwrap()
        };
        assert_eq!(first.result.gas().total_gas_spent(), 2_600);

        let second = {
            let gas_limit = 100;
            let _guard = install_frame_tx_context(matching_context(
                EEADDRESS,
                matching_frame(EEADDRESS, FFADDRESS, gas_limit),
            ));
            evm.transact(matching_tx(EEADDRESS, FFADDRESS, gas_limit))
                .unwrap()
        };
        let ExecutionResult::Success { gas, .. } = second.result else {
            panic!("second frame call to the retained warm target should succeed")
        };
        assert_eq!(gas.total_gas_spent(), 100);
        let _ = evm.ctx.journal_mut().finish_frame_transaction();
    }

    #[test]
    fn outer_sender_is_prewarmed_without_warming_zero_value_entry_point() {
        let sender = FFADDRESS;
        let entry_point = EEADDRESS;
        let gas_limit = 100;
        let _guard = install_frame_tx_context(matching_context(
            sender,
            matching_frame(entry_point, sender, gas_limit),
        ));
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(BenchmarkDB::new_bytecode(Bytecode::new_legacy(
                [STOP].into(),
            )));
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, sender);

        let output = evm
            .transact(matching_tx(entry_point, sender, gas_limit))
            .unwrap();

        assert!(matches!(output.result, ExecutionResult::Success { .. }));
        assert_eq!(output.result.gas().total_gas_spent(), 100);
        assert!(!evm.ctx.journal().evm_state().contains_key(&entry_point));
        let _ = evm.ctx.journal_mut().finish_frame_transaction();
    }

    #[test]
    fn nonzero_frame_value_loads_the_synthetic_caller_for_transfer() {
        let sender = Address::repeat_byte(0x31);
        let caller = Address::repeat_byte(0x32);
        let target = Address::repeat_byte(0x33);
        let gas_limit = 2_600;
        let bytecode = Bytecode::new_legacy([STOP].into());
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(
            caller,
            AccountInfo {
                balance: U256::from(5u64),
                ..Default::default()
            },
        );
        db.insert_account_info(
            target,
            AccountInfo {
                nonce: 1,
                code_hash: bytecode.hash_slow(),
                code: Some(bytecode),
                ..Default::default()
            },
        );
        let mut frame = matching_frame(caller, target, gas_limit);
        frame.value = U256::from(1u64);
        let _guard = install_frame_tx_context(matching_context(sender, frame));
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(db);
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, sender);
        let mut tx = matching_tx(caller, target, gas_limit);
        tx.value = U256::from(1u64);

        let output = evm.transact(tx).unwrap();

        assert!(matches!(output.result, ExecutionResult::Success { .. }));
        assert_eq!(output.state[&caller].info.balance, U256::from(4u64));
        assert_eq!(output.state[&target].info.balance, U256::from(1u64));
        let _ = evm.ctx.journal_mut().finish_frame_transaction();
    }

    #[test]
    fn frame_two_target_is_cold_after_frame_one_revert() {
        let bytecode = Bytecode::new_legacy([PUSH1, 0x00, PUSH1, 0x00, 0xfd].into());
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(BenchmarkDB::new_bytecode(bytecode));
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);

        for frame_index in 0..2 {
            let gas_limit = 2_606;
            let _guard = install_frame_tx_context(matching_context(
                EEADDRESS,
                matching_frame(EEADDRESS, FFADDRESS, gas_limit),
            ));
            let output = evm
                .transact(matching_tx(EEADDRESS, FFADDRESS, gas_limit))
                .unwrap();
            assert!(matches!(output.result, ExecutionResult::Revert { .. }));
            assert_eq!(
                output.result.gas().total_gas_spent(),
                gas_limit,
                "frame {frame_index} must pay a cold target access"
            );
        }

        let _ = evm.ctx.journal_mut().finish_frame_transaction();
    }

    #[test]
    fn frame_storage_keeps_outer_eip2200_original_across_finalize_calls() {
        let target = address!("1000000000000000000000000000000000000055");
        let key = StorageKey::ZERO;
        let bytecode = Bytecode::new_legacy([PUSH1, 0x00, 0x35, PUSH1, 0x00, SSTORE, STOP].into());
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(
            target,
            AccountInfo {
                nonce: 1,
                code_hash: bytecode.hash_slow(),
                code: Some(bytecode),
                ..Default::default()
            },
        );
        db.insert_account_storage(target, key, StorageValue::from(5u64))
            .unwrap();
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(db);
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);

        for (frame_index, value) in [1u64, 2].into_iter().enumerate() {
            let mut data = [0u8; 32];
            data[31] = value as u8;
            let data = Bytes::copy_from_slice(&data);
            let mut frame = matching_frame(EEADDRESS, target, 100_000);
            frame.data = data.clone();
            let _guard = install_frame_tx_context(matching_context(EEADDRESS, frame));
            let mut tx = matching_tx(EEADDRESS, target, 100_000);
            tx.data = data;
            let output = evm.transact(tx).unwrap();
            let slot = &output.state[&target].storage[&key];
            assert_eq!(slot.present_value, StorageValue::from(value));
            assert_eq!(
                slot.original_value,
                StorageValue::from(if frame_index == 0 { 5 } else { 1 }),
                "per-frame delta must be based on that frame's pre-state"
            );
        }

        let retained = &evm.ctx.journal().evm_state()[&target].storage[&key];
        assert_eq!(retained.original_value, StorageValue::from(5u64));
        assert_eq!(retained.present_value, StorageValue::from(2u64));
        let (state, _) = evm.ctx.journal_mut().finish_frame_transaction();
        assert_eq!(
            state[&target].storage[&key].original_value,
            StorageValue::from(5u64)
        );
    }

    #[test]
    fn transient_storage_is_cleared_between_outer_frames() {
        let target = address!("1000000000000000000000000000000000000056");
        // Non-empty calldata stores 1 in transient slot 0. Empty calldata reads
        // that slot and persists the result to storage slot 0.
        let bytecode = Bytecode::new_legacy(
            [
                0x36, PUSH1, 0x0b, 0x57, PUSH1, 0x00, 0x5c, PUSH1, 0x00, SSTORE, STOP, 0x5b, PUSH1,
                0x01, PUSH1, 0x00, 0x5d, STOP,
            ]
            .into(),
        );
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(
            target,
            AccountInfo {
                nonce: 1,
                code_hash: bytecode.hash_slow(),
                code: Some(bytecode),
                ..Default::default()
            },
        );
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(db);
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);

        let mut first_frame = matching_frame(EEADDRESS, target, 100_000);
        first_frame.data = Bytes::from_static(&[1]);
        {
            let _guard = install_frame_tx_context(matching_context(EEADDRESS, first_frame));
            let mut tx = matching_tx(EEADDRESS, target, 100_000);
            tx.data = Bytes::from_static(&[1]);
            let output = evm.transact(tx).unwrap();
            assert!(matches!(output.result, ExecutionResult::Success { .. }));
        }
        assert_eq!(
            evm.ctx.journal_mut().tload(target, StorageKey::ZERO),
            StorageValue::ZERO
        );

        {
            let _guard = install_frame_tx_context(matching_context(
                EEADDRESS,
                matching_frame(EEADDRESS, target, 100_000),
            ));
            let output = evm
                .transact(matching_tx(EEADDRESS, target, 100_000))
                .unwrap();
            assert!(matches!(output.result, ExecutionResult::Success { .. }));
        }
        assert_eq!(
            evm.ctx.journal().evm_state()[&target].storage[&StorageKey::ZERO].present_value,
            StorageValue::ZERO
        );
        let _ = evm.ctx.journal_mut().finish_frame_transaction();
    }

    #[test]
    fn frame_results_have_local_logs_and_finish_returns_ordered_cumulative_logs() {
        let first_target = address!("1000000000000000000000000000000000000061");
        let second_target = address!("1000000000000000000000000000000000000062");
        let bytecode = Bytecode::new_legacy([PUSH1, 0x00, PUSH1, 0x00, 0xa0, STOP].into());
        let mut db = CacheDB::<EmptyDB>::default();
        for target in [first_target, second_target] {
            db.insert_account_info(
                target,
                AccountInfo {
                    nonce: 1,
                    code_hash: bytecode.hash_slow(),
                    code: Some(bytecode.clone()),
                    ..Default::default()
                },
            );
        }
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(db);
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);

        for target in [first_target, second_target] {
            let _guard = install_frame_tx_context(matching_context(
                EEADDRESS,
                matching_frame(EEADDRESS, target, 100_000),
            ));
            let output = evm
                .transact(matching_tx(EEADDRESS, target, 100_000))
                .unwrap();
            assert_eq!(output.result.logs().len(), 1);
            assert_eq!(output.result.logs()[0].address, target);
        }

        let (_, logs) = evm.ctx.journal_mut().finish_frame_transaction();
        assert_eq!(logs.len(), 2);
        assert_eq!(logs[0].address, first_target);
        assert_eq!(logs[1].address, second_target);
    }

    #[test]
    fn raw_handler_calls_finalize_each_frame_without_clearing_outer_journal() {
        let first_target = address!("1000000000000000000000000000000000000063");
        let second_target = address!("1000000000000000000000000000000000000064");
        let bytecode =
            Bytecode::new_legacy([PUSH1, 0x01, PUSH1, 0x00, SSTORE, STOP].as_slice().into());
        let mut db = CacheDB::<EmptyDB>::default();
        for target in [first_target, second_target] {
            db.insert_account_info(
                target,
                AccountInfo {
                    nonce: 1,
                    code_hash: bytecode.hash_slow(),
                    code: Some(bytecode.clone()),
                    ..Default::default()
                },
            );
        }
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(db);
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);

        for (target, other_target) in [(first_target, second_target), (second_target, first_target)]
        {
            let result = {
                let _guard = install_frame_tx_context(matching_context(
                    EEADDRESS,
                    matching_frame(EEADDRESS, target, 100_000),
                ));
                evm.transact_one(matching_tx(EEADDRESS, target, 100_000))
                    .unwrap()
            };
            assert!(matches!(result, ExecutionResult::Success { .. }));
            assert!(evm.ctx.journal().is_frame_transaction_active());

            // Foundry's raw handler path must explicitly take the settled delta
            // before the next frame because it does not call ExecuteEvm::finalize.
            let delta = evm.ctx.journal_mut().finalize_frame_transaction_call();
            assert_eq!(
                delta[&target].storage[&StorageKey::ZERO].present_value,
                StorageValue::from(1)
            );
            assert!(!delta.contains_key(&other_target));
        }

        let (state, _) = evm.ctx.journal_mut().finish_frame_transaction();
        for target in [first_target, second_target] {
            assert_eq!(
                state[&target].storage[&StorageKey::ZERO].present_value,
                StorageValue::from(1)
            );
        }
    }

    #[test]
    fn created_local_survives_between_frames_for_later_eip6780_selfdestruct() {
        let factory = address!("1000000000000000000000000000000000000071");
        let beneficiary = address!("1000000000000000000000000000000000000072");
        let factory_nonce = 1;

        let mut runtime = vec![0x73];
        runtime.extend_from_slice(beneficiary.as_slice());
        runtime.push(0xff);
        let mut initcode = vec![
            PUSH1,
            runtime.len() as u8,
            PUSH1,
            0x0c,
            PUSH1,
            0x00,
            0x39,
            PUSH1,
            runtime.len() as u8,
            PUSH1,
            0x00,
            0xf3,
        ];
        initcode.extend_from_slice(&runtime);
        let mut factory_code = vec![
            PUSH1,
            initcode.len() as u8,
            PUSH1,
            0x0f,
            PUSH1,
            0x00,
            0x39,
            PUSH1,
            initcode.len() as u8,
            PUSH1,
            0x00,
            PUSH1,
            0x00,
            CREATE,
            STOP,
        ];
        factory_code.extend_from_slice(&initcode);
        let factory_code = Bytecode::new_legacy(factory_code.into());
        let created = factory.create(factory_nonce);
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(
            factory,
            AccountInfo {
                nonce: factory_nonce,
                code_hash: factory_code.hash_slow(),
                code: Some(factory_code),
                ..Default::default()
            },
        );
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(db);
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);

        {
            let _guard = install_frame_tx_context(matching_context(
                EEADDRESS,
                matching_frame(EEADDRESS, factory, 300_000),
            ));
            let output = evm
                .transact(matching_tx(EEADDRESS, factory, 300_000))
                .unwrap();
            assert!(matches!(output.result, ExecutionResult::Success { .. }));
            assert!(evm.ctx.journal().evm_state()[&created].is_created_locally());
        }

        {
            let _guard = install_frame_tx_context(matching_context(
                EEADDRESS,
                matching_frame(EEADDRESS, created, 100_000),
            ));
            let output = evm
                .transact(matching_tx(EEADDRESS, created, 100_000))
                .unwrap();
            assert!(matches!(output.result, ExecutionResult::Success { .. }));
            assert!(output.state[&created].is_selfdestructed());
        }

        let (state, _) = evm.ctx.journal_mut().finish_frame_transaction();
        let finalized = &state[&created];
        assert!(!finalized.is_created());
        assert!(!finalized.is_selfdestructed());
        assert!(finalized.info.is_empty());
    }

    #[test]
    fn eip8246_delayed_clear_runs_only_when_outer_frame_transaction_finishes() {
        let factory = address!("1000000000000000000000000000000000000073");
        let factory_nonce = 1;
        let created = factory.create(factory_nonce);

        let mut runtime = vec![0x73];
        runtime.extend_from_slice(created.as_slice());
        runtime.push(0xff);
        let mut initcode = vec![
            PUSH1,
            runtime.len() as u8,
            PUSH1,
            0x0c,
            PUSH1,
            0x00,
            0x39,
            PUSH1,
            runtime.len() as u8,
            PUSH1,
            0x00,
            0xf3,
        ];
        initcode.extend_from_slice(&runtime);
        let mut factory_code = vec![
            PUSH1,
            initcode.len() as u8,
            PUSH1,
            0x0f,
            PUSH1,
            0x00,
            0x39,
            PUSH1,
            initcode.len() as u8,
            PUSH1,
            0x00,
            PUSH1,
            0x01,
            CREATE,
            STOP,
        ];
        factory_code.extend_from_slice(&initcode);
        let factory_code = Bytecode::new_legacy(factory_code.into());
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(
            factory,
            AccountInfo {
                balance: U256::from(1u64),
                nonce: factory_nonce,
                code_hash: factory_code.hash_slow(),
                code: Some(factory_code),
                ..Default::default()
            },
        );
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| {
                cfg.set_spec_and_mainnet_gas_params(SpecId::AMSTERDAM);
                cfg.enable_amsterdam_eip2780 = false;
                cfg.enable_amsterdam_eip8037 = false;
            })
            .with_db(db);
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);

        {
            let _guard = install_frame_tx_context(matching_context(
                EEADDRESS,
                matching_frame(EEADDRESS, factory, 2_000_000),
            ));
            let output = evm
                .transact(matching_tx(EEADDRESS, factory, 2_000_000))
                .unwrap();
            assert!(matches!(output.result, ExecutionResult::Success { .. }));
        }
        {
            let _guard = install_frame_tx_context(matching_context(
                EEADDRESS,
                matching_frame(EEADDRESS, created, 200_000),
            ));
            let output = evm
                .transact(matching_tx(EEADDRESS, created, 200_000))
                .unwrap();
            assert!(matches!(output.result, ExecutionResult::Success { .. }));
        }

        let retained = &evm.ctx.journal().evm_state()[&created];
        assert!(retained.is_selfdestructed());
        assert_eq!(retained.info.balance, U256::from(1u64));
        assert_eq!(retained.info.nonce, 1);
        assert!(!retained.info.code.as_ref().unwrap().is_empty());

        let (state, _) = evm.ctx.journal_mut().finish_frame_transaction();
        let cleared = &state[&created];
        assert!(!cleared.is_selfdestructed());
        assert_eq!(cleared.info.balance, U256::from(1u64));
        assert_eq!(cleared.info.nonce, 0);
        assert!(cleared.info.code.as_ref().unwrap().is_empty());
    }

    #[test]
    fn frame_transaction_error_aborts_and_fully_resets_the_outer_journal() {
        let bytecode = Bytecode::new_legacy([PUSH1, 0x01, PUSH1, 0x00, SSTORE, STOP].into());
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(BenchmarkDB::new_bytecode(bytecode));
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);

        {
            let _guard = install_frame_tx_context(matching_context(
                EEADDRESS,
                matching_frame(EEADDRESS, FFADDRESS, 100_000),
            ));
            let output = evm
                .transact(matching_tx(EEADDRESS, FFADDRESS, 100_000))
                .unwrap();
            assert_eq!(
                output.state[&FFADDRESS].storage[&StorageKey::ZERO].present_value,
                StorageValue::from(1u64)
            );
        }

        {
            let _guard = install_frame_tx_context(matching_context(
                EEADDRESS,
                matching_frame(EEADDRESS, FFADDRESS, 100_000),
            ));
            let err = evm
                .transact(matching_tx(EEADDRESS, FFADDRESS, 99_999))
                .unwrap_err();
            assert!(matches!(
                &err,
                EVMError::Transaction(InvalidTransaction::Str(message))
                    if message == "synthetic transaction does not match the current frame"
            ));
        }

        assert!(!evm.ctx.journal().is_frame_transaction_active());
        assert_eq!(evm.ctx.journal().depth(), 0);
        assert!(evm.ctx.journal().logs().is_empty());
        assert!(evm.ctx.journal().evm_state().is_empty());
        evm.ctx.journal_mut().abort_frame_transaction();
        assert!(evm.ctx.journal().evm_state().is_empty());

        let ordinary = evm
            .transact(
                TxEnv::builder()
                    .caller(EEADDRESS)
                    .kind(TxKind::Call(FFADDRESS))
                    .gas_limit(100_000)
                    .build()
                    .unwrap(),
            )
            .unwrap();
        assert!(matches!(ordinary.result, ExecutionResult::Success { .. }));
        assert_eq!(ordinary.state[&EEADDRESS].info.nonce, 1);
    }

    #[test]
    fn frame_context_nested_create_uses_pre_frame_nonce() {
        let factory = address!("1000000000000000000000000000000000000001");
        let implementation = address!("1000000000000000000000000000000000000002");
        let original_nonce = 7;
        let create_code = Bytecode::new_legacy(
            [
                PUSH1, 0x00, // initcode length
                PUSH1, 0x00, // initcode offset
                PUSH1, 0x00, // value
                CREATE, POP, STOP,
            ]
            .into(),
        );
        let designation = Bytecode::new_eip7702(implementation);
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(
            factory,
            AccountInfo {
                balance: U256::from(1_000_000u64),
                nonce: original_nonce,
                code_hash: designation.hash_slow(),
                code: Some(designation),
                ..Default::default()
            },
        );
        db.insert_account_info(
            implementation,
            AccountInfo {
                nonce: 1,
                code_hash: create_code.hash_slow(),
                code: Some(create_code),
                ..Default::default()
            },
        );
        let _guard = install_frame_tx_context(matching_context(
            factory,
            matching_frame(factory, factory, 100_000),
        ));
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(db);
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, factory);

        let mut tx = matching_tx(factory, factory, 100_000);
        tx.nonce = original_nonce;
        let output = evm.transact(tx).unwrap();

        assert!(matches!(output.result, ExecutionResult::Success { .. }));
        let expected = factory.create(original_nonce);
        let synthetic_bump_address = factory.create(original_nonce + 1);
        assert!(output.state.contains_key(&expected));
        assert!(!output.state.contains_key(&synthetic_bump_address));
        assert_eq!(output.state[&factory].info.nonce, original_nonce + 1);
        let _ = evm.ctx.journal_mut().finish_frame_transaction();
    }

    #[test]
    fn ordinary_call_with_foreign_frame_context_keeps_intrinsic_and_nonce_bump() {
        let _guard = install_frame_tx_context(FrameTxContext {
            frames: vec![FrameInfo {
                resolved_target: EEADDRESS,
                ..Default::default()
            }],
            ..Default::default()
        });
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(BenchmarkDB::new_bytecode(Bytecode::new_legacy(
                [STOP].into(),
            )));
        let mut evm = ctx.build_mainnet();
        let err = evm
            .transact(
                TxEnv::builder()
                    .caller(EEADDRESS)
                    .kind(TxKind::Call(FFADDRESS))
                    .gas_limit(20_999)
                    .build()
                    .unwrap(),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            EVMError::Transaction(InvalidTransaction::CallGasCostMoreThanGasLimit { .. })
        ));

        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(BenchmarkDB::new_bytecode(Bytecode::new_legacy(
                [STOP].into(),
            )));
        let mut evm = ctx.build_mainnet();
        let output = evm
            .transact(
                TxEnv::builder()
                    .caller(EEADDRESS)
                    .kind(TxKind::Call(FFADDRESS))
                    .gas_limit(21_000)
                    .build()
                    .unwrap(),
            )
            .unwrap();

        let ExecutionResult::Success { gas, .. } = output.result else {
            panic!("ordinary call should succeed")
        };
        assert_eq!(gas.total_gas_spent(), 21_000);
        assert_eq!(output.state[&EEADDRESS].info.nonce, 1);
    }

    #[test]
    fn ordinary_same_target_context_is_not_forced_static_without_outer_lifecycle() {
        let _guard = install_frame_tx_context(FrameTxContext {
            frames: vec![FrameInfo {
                resolved_target: FFADDRESS,
                mode: 1,
                ..Default::default()
            }],
            ..Default::default()
        });
        let bytecode = Bytecode::new_legacy([PUSH1, 0x01, PUSH1, 0x01, SSTORE, STOP].into());
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(BenchmarkDB::new_bytecode(bytecode));
        let mut evm = ctx.build_mainnet();

        let output = evm
            .transact(
                TxEnv::builder()
                    .caller(EEADDRESS)
                    .kind(TxKind::Call(FFADDRESS))
                    .gas_limit(100_000)
                    .build()
                    .unwrap(),
            )
            .unwrap();

        assert!(matches!(output.result, ExecutionResult::Success { .. }));
        assert_eq!(
            output.state[&FFADDRESS].storage[&StorageKey::from(1)].present_value,
            StorageValue::from(1)
        );
    }

    #[test]
    fn target_only_tooling_context_still_forces_nested_calls_static() {
        let nested_target = address!("1000000000000000000000000000000000000081");
        let caller_target = address!("1000000000000000000000000000000000000082");
        let nested_code = Bytecode::new_legacy([PUSH1, 0x01, PUSH1, 0x00, SSTORE, STOP].into());
        let mut caller_code = vec![
            PUSH1, 0x00, PUSH1, 0x00, PUSH1, 0x00, PUSH1, 0x00, PUSH1, 0x00, 0x73,
        ];
        caller_code.extend_from_slice(nested_target.as_slice());
        caller_code.extend_from_slice(&[0x61, 0xff, 0xff, 0xf1, POP, STOP]);
        let caller_code = Bytecode::new_legacy(caller_code.into());
        let mut db = CacheDB::<EmptyDB>::default();
        for (target, code) in [(nested_target, nested_code), (caller_target, caller_code)] {
            db.insert_account_info(
                target,
                AccountInfo {
                    nonce: 1,
                    code_hash: code.hash_slow(),
                    code: Some(code),
                    ..Default::default()
                },
            );
        }
        let _guard = install_frame_tx_context(FrameTxContext {
            frames: vec![FrameInfo {
                resolved_target: nested_target,
                mode: 1,
                ..Default::default()
            }],
            ..Default::default()
        });
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(db);
        let mut evm = ctx.build_mainnet();

        let output = evm
            .transact(
                TxEnv::builder()
                    .caller(EEADDRESS)
                    .kind(TxKind::Call(caller_target))
                    .gas_limit(200_000)
                    .build()
                    .unwrap(),
            )
            .unwrap();

        assert!(matches!(output.result, ExecutionResult::Success { .. }));
        assert_ne!(
            output
                .state
                .get(&nested_target)
                .and_then(|account| account.storage.get(&StorageKey::ZERO))
                .map(|slot| slot.present_value),
            Some(StorageValue::from(1u64))
        );
    }

    #[test]
    fn active_frame_context_rejects_every_synthetic_call_field_mismatch() {
        let gas_limit = 100_000;
        let mut invalid_mode = matching_frame(EEADDRESS, FFADDRESS, gas_limit);
        invalid_mode.mode = 4;
        let mut wrong_target = matching_frame(EEADDRESS, EEADDRESS, gas_limit);
        wrong_target.resolved_target = EEADDRESS;
        let wrong_caller = matching_frame(Address::ZERO, FFADDRESS, gas_limit);
        let mut wrong_data = matching_frame(EEADDRESS, FFADDRESS, gas_limit);
        wrong_data.data = Bytes::from_static(&[0x01]);
        let mut wrong_value = matching_frame(EEADDRESS, FFADDRESS, gas_limit);
        wrong_value.value = U256::from(1u64);
        let wrong_gas = matching_frame(EEADDRESS, FFADDRESS, gas_limit - 1);

        for (field, frame) in [
            ("mode", invalid_mode),
            ("target", wrong_target),
            ("caller", wrong_caller),
            ("data", wrong_data),
            ("value", wrong_value),
            ("gas_limit", wrong_gas),
        ] {
            let _guard = install_frame_tx_context(FrameTxContext {
                sender: EEADDRESS,
                frames: vec![frame],
                ..Default::default()
            });
            let ctx = Context::mainnet()
                .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
                .with_db(BenchmarkDB::new_bytecode(Bytecode::new_legacy(
                    [STOP].into(),
                )));
            let mut evm = ctx.build_mainnet();
            begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);
            let err = evm
                .transact(matching_tx(EEADDRESS, FFADDRESS, gas_limit))
                .unwrap_err();

            assert!(
                matches!(
                    &err,
                    EVMError::Transaction(InvalidTransaction::Str(message))
                        if message == "synthetic transaction does not match the current frame"
                ),
                "mismatched {field} must be rejected: {err:?}"
            );
            assert!(!evm.ctx.journal().is_frame_transaction_active());
        }
    }

    #[test]
    fn active_frame_context_rejects_unbound_transaction_metadata() {
        let gas_limit = 100_000;
        let frame = matching_frame(EEADDRESS, FFADDRESS, gas_limit);

        let mut wrong_type = matching_tx(EEADDRESS, FFADDRESS, gas_limit);
        wrong_type.tx_type = 0;
        wrong_type.gas_priority_fee = None;

        let mut access_list = matching_tx(EEADDRESS, FFADDRESS, gas_limit);
        access_list.access_list = AccessList(vec![AccessListItem {
            address: FFADDRESS,
            storage_keys: vec![],
        }]);

        let signer = PrivateKeySigner::random();
        let authorization = Authorization {
            chain_id: U256::ZERO,
            nonce: 0,
            address: FFADDRESS,
        };
        let signature = signer
            .sign_hash_sync(&authorization.signature_hash())
            .unwrap();
        let mut authorization_list = matching_tx(EEADDRESS, FFADDRESS, gas_limit);
        authorization_list.authorization_list =
            vec![Either::Left(authorization.into_signed(signature))];

        let mut max_fee = matching_tx(EEADDRESS, FFADDRESS, gas_limit);
        max_fee.gas_price = 1;

        let mut priority_fee = matching_tx(EEADDRESS, FFADDRESS, gas_limit);
        priority_fee.gas_priority_fee = Some(1);

        let mut unexpected_blob = matching_tx(EEADDRESS, FFADDRESS, gas_limit);
        unexpected_blob.blob_hashes = vec![B256::with_last_byte(1)];
        unexpected_blob.derive_tx_type().unwrap();

        let mut blob_context = matching_context(EEADDRESS, frame.clone());
        blob_context.blob_count = 1;
        let wrong_sender_context = matching_context(Address::ZERO, frame.clone());

        for (field, context, tx) in [
            (
                "tx type",
                matching_context(EEADDRESS, frame.clone()),
                wrong_type,
            ),
            (
                "access list",
                matching_context(EEADDRESS, frame.clone()),
                access_list,
            ),
            (
                "authorization list",
                matching_context(EEADDRESS, frame.clone()),
                authorization_list,
            ),
            (
                "max fee",
                matching_context(EEADDRESS, frame.clone()),
                max_fee,
            ),
            (
                "priority fee",
                matching_context(EEADDRESS, frame.clone()),
                priority_fee,
            ),
            (
                "unexpected blob",
                matching_context(EEADDRESS, frame.clone()),
                unexpected_blob,
            ),
            (
                "missing blob transaction",
                blob_context,
                matching_tx(EEADDRESS, FFADDRESS, gas_limit),
            ),
            (
                "outer sender",
                wrong_sender_context,
                matching_tx(EEADDRESS, FFADDRESS, gas_limit),
            ),
        ] {
            let _guard = install_frame_tx_context(context);
            let ctx = Context::mainnet()
                .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
                .with_db(BenchmarkDB::new_bytecode(Bytecode::new_legacy(
                    [STOP].into(),
                )));
            let mut evm = ctx.build_mainnet();
            begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);
            let err = evm.transact(tx).unwrap_err();
            assert!(
                matches!(
                    &err,
                    EVMError::Transaction(InvalidTransaction::Str(message))
                        if message == "synthetic transaction does not match the current frame"
                ),
                "{field} must be rejected"
            );
            assert!(!evm.ctx.journal().is_frame_transaction_active());
        }
    }

    #[test]
    fn frame_context_accepts_matching_blob_count_and_fee_metadata() {
        let gas_limit = 2_600;
        let mut blob_hash = [0u8; 32];
        blob_hash[0] = 1;
        let mut tx = matching_tx(EEADDRESS, FFADDRESS, gas_limit);
        tx.blob_hashes = vec![B256::from(blob_hash)];
        tx.max_fee_per_blob_gas = 1;
        tx.derive_tx_type().unwrap();
        let mut frame_context =
            matching_context(EEADDRESS, matching_frame(EEADDRESS, FFADDRESS, gas_limit));
        frame_context.blob_count = 1;
        frame_context.max_fee_per_blob_gas = U256::from(1u64);
        let _guard = install_frame_tx_context(frame_context);
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(BenchmarkDB::new_bytecode(Bytecode::new_legacy(
                [STOP].into(),
            )));
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);

        let output = evm.transact(tx).unwrap();

        assert!(matches!(output.result, ExecutionResult::Success { .. }));
        let _ = evm.ctx.journal_mut().finish_frame_transaction();
    }

    #[test]
    fn frame_context_allows_contract_callers_and_matching_max_nonce() {
        let caller = address!("10000000000000000000000000000000000000c1");
        let target = address!("10000000000000000000000000000000000000d1");
        let gas_limit = 2_600;

        for (case, nonce, caller_code) in [
            (
                "contract caller",
                7,
                Some(Bytecode::new_legacy([STOP].into())),
            ),
            ("maximum nonce", u64::MAX, None),
        ] {
            let mut db = CacheDB::<EmptyDB>::default();
            let (code_hash, code) = caller_code
                .map(|code| (code.hash_slow(), Some(code)))
                .unwrap_or_default();
            db.insert_account_info(
                caller,
                AccountInfo {
                    nonce,
                    code_hash,
                    code,
                    ..Default::default()
                },
            );
            let target_code = Bytecode::new_legacy([STOP].into());
            db.insert_account_info(
                target,
                AccountInfo {
                    nonce: 1,
                    code_hash: target_code.hash_slow(),
                    code: Some(target_code),
                    ..Default::default()
                },
            );
            let _guard = install_frame_tx_context(matching_context(
                caller,
                matching_frame(caller, target, gas_limit),
            ));
            let ctx = Context::mainnet()
                .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
                .with_db(db);
            let mut evm = ctx.build_mainnet();
            begin_outer_frame_transaction(&mut evm.ctx, caller);
            let mut tx = matching_tx(caller, target, gas_limit);
            tx.nonce = nonce;
            let output = evm
                .transact(tx)
                .unwrap_or_else(|err| panic!("{case} should be valid: {err:?}"));

            assert!(matches!(output.result, ExecutionResult::Success { .. }));
            let (state, _) = evm.ctx.journal_mut().finish_frame_transaction();
            assert_eq!(state.get(&caller).map(|account| account.info.nonce), None);
        }
    }

    #[test]
    fn frame_context_still_requires_nonce_equality() {
        let caller = address!("10000000000000000000000000000000000000c1");
        let target = address!("10000000000000000000000000000000000000d1");
        let gas_limit = 2_600;
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(
            caller,
            AccountInfo {
                nonce: 7,
                ..Default::default()
            },
        );
        let _guard = install_frame_tx_context(matching_context(
            caller,
            matching_frame(caller, target, gas_limit),
        ));
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE))
            .with_db(db);
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, caller);
        let mut tx = matching_tx(caller, target, gas_limit);
        tx.nonce = 6;
        let err = evm.transact(tx).unwrap_err();

        assert!(matches!(
            err,
            EVMError::Transaction(InvalidTransaction::NonceTooLow { tx: 6, state: 7 })
        ));
        assert!(!evm.ctx.journal().is_frame_transaction_active());
        assert!(evm.ctx.journal().evm_state().is_empty());
    }

    #[test]
    fn scalar_frame_gas_rejects_amsterdam_state_gas() {
        let gas_limit = 3_000;
        let regular_cap = 2_600;
        let tx = matching_tx(EEADDRESS, FFADDRESS, gas_limit);
        let _guard = install_frame_tx_context(matching_context(
            EEADDRESS,
            matching_frame(EEADDRESS, FFADDRESS, gas_limit),
        ));
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| {
                cfg.set_spec_and_mainnet_gas_params(SpecId::AMSTERDAM);
                cfg.tx_gas_limit_cap = Some(regular_cap);
            })
            .with_tx(tx)
            .with_db(BenchmarkDB::new_bytecode(Bytecode::new_legacy(
                [STOP].into(),
            )));
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);
        let handler: MainnetHandler<_, EVMError<core::convert::Infallible>, EthFrame> =
            MainnetHandler::default();

        let err = handler.validate(&mut evm).unwrap_err();
        assert!(matches!(
            err,
            EVMError::Transaction(InvalidTransaction::Str(message))
                if message == "scalar frame gas does not support Amsterdam state-gas rules"
        ));
        evm.ctx.journal_mut().abort_frame_transaction();
    }

    #[test]
    fn scalar_frame_gas_rejects_independently_enabled_eip2780() {
        let gas_limit = 3_000;
        let tx = matching_tx(EEADDRESS, FFADDRESS, gas_limit);
        let _guard = install_frame_tx_context(matching_context(
            EEADDRESS,
            matching_frame(EEADDRESS, FFADDRESS, gas_limit),
        ));
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| {
                cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE);
                cfg.enable_amsterdam_eip2780 = true;
            })
            .with_tx(tx)
            .with_db(BenchmarkDB::new_bytecode(Bytecode::new_legacy(
                [STOP].into(),
            )));
        let mut evm = ctx.build_mainnet();
        begin_outer_frame_transaction(&mut evm.ctx, EEADDRESS);
        let handler: MainnetHandler<_, EVMError<core::convert::Infallible>, EthFrame> =
            MainnetHandler::default();

        let err = handler.validate(&mut evm).unwrap_err();
        assert!(matches!(
            err,
            EVMError::Transaction(InvalidTransaction::Str(message))
                if message == "scalar frame gas does not support Amsterdam state-gas rules"
        ));
        evm.ctx.journal_mut().abort_frame_transaction();
    }

    #[test]
    fn native_context_mutation_cannot_change_latched_frame_classification() {
        let caller = EEADDRESS;
        let target = FFADDRESS;
        let beneficiary = Address::repeat_byte(0x43);
        let gas_limit = 100_000;
        let bytecode = Bytecode::new_legacy([PUSH1, 0x01, PUSH1, 0x00, SSTORE, STOP].into());
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(caller, AccountInfo::default());
        db.insert_account_info(
            target,
            AccountInfo {
                nonce: 1,
                code_hash: bytecode.hash_slow(),
                code: Some(bytecode),
                ..Default::default()
            },
        );
        let spec = SpecId::PRAGUE;
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(spec))
            .modify_block_chained(|block| block.beneficiary = beneficiary)
            .with_db(db);
        let mut initial_frame = matching_frame(caller, target, gas_limit);
        initial_frame.mode = 1;
        let mut initial_context = matching_context(caller, initial_frame);
        initial_context.max_fee_per_gas = U256::from(7u64);
        let changed_context = matching_context(
            caller,
            matching_frame(caller, Address::repeat_byte(0x99), gas_limit - 1),
        );
        let ctx = NativeFrameContext::new_mutating(ctx, initial_context, changed_context);
        let mut evm = Evm::<_, (), _, _, EthFrame>::new(
            ctx,
            EthInstructions::new_mainnet_with_spec(spec),
            EthPrecompiles::new(spec),
        );
        begin_outer_frame_transaction(&mut evm.ctx, caller);
        let mut tx = matching_tx(caller, target, gas_limit);
        tx.gas_price = 7;

        let output = evm.transact(tx).unwrap();

        assert!(matches!(
            output.result,
            ExecutionResult::Halt {
                reason: HaltReason::StateChangeDuringStaticCall,
                ..
            }
        ));
        assert_eq!(evm.ctx.frame_context_reads.get(), 1);
        assert_eq!(
            output
                .state
                .get(&caller)
                .map(|account| account.info.balance)
                .unwrap_or_default(),
            U256::ZERO
        );
        assert_eq!(
            output
                .state
                .get(&beneficiary)
                .map(|account| account.info.balance)
                .unwrap_or_default(),
            U256::ZERO
        );
        let _ = evm.ctx.journal_mut().finish_frame_transaction();
    }

    #[test]
    fn native_frame_context_does_not_mint_gas_fees_during_settlement() {
        let caller = EEADDRESS;
        let target = FFADDRESS;
        let beneficiary = Address::repeat_byte(0x42);
        let gas_limit = 3_000;
        let target_code = Bytecode::new_legacy([STOP].into());
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(caller, AccountInfo::default());
        db.insert_account_info(
            target,
            AccountInfo {
                nonce: 1,
                code_hash: target_code.hash_slow(),
                code: Some(target_code),
                ..Default::default()
            },
        );
        let spec = SpecId::PRAGUE;
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(spec))
            .modify_block_chained(|block| block.beneficiary = beneficiary)
            .with_db(db);
        let ctx = NativeFrameContext::new(
            ctx,
            FrameTxContext {
                sender: caller,
                max_fee_per_gas: U256::from(7u64),
                frames: vec![matching_frame(caller, target, gas_limit)],
                ..Default::default()
            },
        );
        let mut evm = Evm::<_, (), _, _, EthFrame>::new(
            ctx,
            EthInstructions::new_mainnet_with_spec(spec),
            EthPrecompiles::new(spec),
        );
        begin_outer_frame_transaction(&mut evm.ctx, caller);

        let mut tx = matching_tx(caller, target, gas_limit);
        tx.gas_price = 7;
        let output = evm.transact(tx).unwrap();

        let ExecutionResult::Success { gas, .. } = output.result else {
            panic!("native frame call should succeed")
        };
        assert_eq!(gas.total_gas_spent(), 2_600);
        assert_eq!(
            output
                .state
                .get(&caller)
                .map(|account| account.info.balance)
                .unwrap_or_default(),
            U256::ZERO
        );
        assert_eq!(
            output
                .state
                .get(&beneficiary)
                .map(|account| account.info.balance)
                .unwrap_or_default(),
            U256::ZERO
        );
        let _ = evm.ctx.journal_mut().finish_frame_transaction();
    }
}
