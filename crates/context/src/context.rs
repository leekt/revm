//! This module contains [`Context`] struct and implements [`ContextTr`] trait for it.
use crate::{block::BlockEnv, cfg::CfgEnv, journal::Journal, tx::TxEnv, LocalContext};
use context_interface::{
    cfg::GasParams,
    context::{ContextError, ContextSetters, SStoreResult, SelfDestructResult, StateLoad},
    host::{LoadError, SetDelegateError},
    journaled_state::{account::JournaledAccountTr, AccountInfoLoad},
    Block, Cfg, ContextTr, Host, JournalTr, LocalContextTr, Transaction, TransactionType,
};
use database_interface::{Database, DatabaseRef, EmptyDB, WrapDatabaseRef};
use derive_where::derive_where;
use primitives::{
    eip7819::DELEGATION_PREFIX, hardfork::SpecId, hints_util::cold_path, Address, Log, StorageKey,
    StorageValue, B256, U256,
};
use state::Bytecode;

/// EVM context contains data that EVM needs for execution.
#[derive_where(Clone, Debug; BLOCK, CFG, CHAIN, TX, DB, JOURNAL, <DB as Database>::Error, LOCAL)]
pub struct Context<
    BLOCK = BlockEnv,
    TX = TxEnv,
    CFG = CfgEnv,
    DB: Database = EmptyDB,
    JOURNAL: JournalTr<Database = DB> = Journal<DB>,
    CHAIN = (),
    LOCAL: LocalContextTr = LocalContext,
> {
    /// Block information.
    pub block: BLOCK,
    /// Transaction information.
    pub tx: TX,
    /// Configurations.
    pub cfg: CFG,
    /// EVM State with journaling support and database.
    pub journaled_state: JOURNAL,
    /// Inner context.
    pub chain: CHAIN,
    /// Local context that is filled by execution.
    pub local: LOCAL,
    /// Error that happened during execution.
    pub error: Result<(), ContextError<DB::Error>>,
}

#[inline]
fn sync_cfg_to_journal<CFG: Cfg, JOURNAL: JournalTr>(cfg: &CFG, journal: &mut JOURNAL) {
    let spec = cfg.spec().into();
    journal.set_spec_id(spec);
    journal.set_eip7851_enabled(cfg.is_eip7851_enabled() && spec.is_enabled_in(SpecId::PRAGUE));
    journal.set_eip7708_config(
        cfg.is_eip7708_disabled(),
        cfg.is_eip8246_delayed_clear_disabled(),
    );
}

impl<
        BLOCK: Block,
        TX: Transaction,
        DB: Database,
        CFG: Cfg,
        JOURNAL: JournalTr<Database = DB>,
        CHAIN,
        LOCAL: LocalContextTr,
    > ContextTr for Context<BLOCK, TX, CFG, DB, JOURNAL, CHAIN, LOCAL>
{
    type Block = BLOCK;
    type Tx = TX;
    type Cfg = CFG;
    type Db = DB;
    type Journal = JOURNAL;
    type Chain = CHAIN;
    type Local = LOCAL;

    #[inline]
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
        let block = &self.block;
        let tx = &self.tx;
        let cfg = &self.cfg;
        let db = self.journaled_state.db();
        let journal = &self.journaled_state;
        let chain = &self.chain;
        let local = &self.local;

        (block, tx, cfg, db, journal, chain, local)
    }

    #[inline]
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
        let block = &self.block;
        let tx = &self.tx;
        let cfg = &self.cfg;
        let journal = &mut self.journaled_state;
        let chain = &mut self.chain;
        let local = &mut self.local;

        (block, tx, cfg, journal, chain, local)
    }

    #[inline]
    fn error(&mut self) -> &mut Result<(), ContextError<<Self::Db as Database>::Error>> {
        &mut self.error
    }
}

impl<
        BLOCK: Block,
        TX: Transaction,
        DB: Database,
        CFG: Cfg,
        JOURNAL: JournalTr<Database = DB>,
        CHAIN,
        LOCAL: LocalContextTr,
    > ContextSetters for Context<BLOCK, TX, CFG, DB, JOURNAL, CHAIN, LOCAL>
{
    fn set_tx(&mut self, tx: Self::Tx) {
        self.tx = tx;
    }

    fn set_block(&mut self, block: Self::Block) {
        self.block = block;
    }
}

impl<
        BLOCK: Block + Default,
        TX: Transaction + Default,
        DB: Database,
        JOURNAL: JournalTr<Database = DB>,
        CHAIN: Default,
        LOCAL: LocalContextTr + Default,
        SPEC: Default + Into<SpecId> + Clone,
    > Context<BLOCK, TX, CfgEnv<SPEC>, DB, JOURNAL, CHAIN, LOCAL>
{
    /// Creates a new context with a new database type.
    ///
    /// This will create a new [`Journal`] object.
    pub fn new(db: DB, spec: SPEC) -> Self {
        let cfg = CfgEnv::new_with_spec(spec);
        let mut journaled_state = JOURNAL::new(db);
        sync_cfg_to_journal(&cfg, &mut journaled_state);
        Self {
            tx: TX::default(),
            block: BLOCK::default(),
            cfg,
            local: LOCAL::default(),
            journaled_state,
            chain: Default::default(),
            error: Ok(()),
        }
    }
}

impl<BLOCK, TX, CFG, DB, JOURNAL, CHAIN, LOCAL> Context<BLOCK, TX, CFG, DB, JOURNAL, CHAIN, LOCAL>
where
    BLOCK: Block,
    TX: Transaction,
    CFG: Cfg,
    DB: Database,
    JOURNAL: JournalTr<Database = DB>,
    LOCAL: LocalContextTr,
{
    /// Creates a new context with a new journal type. New journal needs to have the same database type.
    pub fn with_new_journal<OJOURNAL: JournalTr<Database = DB>>(
        self,
        mut journal: OJOURNAL,
    ) -> Context<BLOCK, TX, CFG, DB, OJOURNAL, CHAIN, LOCAL> {
        sync_cfg_to_journal(&self.cfg, &mut journal);
        Context {
            tx: self.tx,
            block: self.block,
            cfg: self.cfg,
            journaled_state: journal,
            local: self.local,
            chain: self.chain,
            error: Ok(()),
        }
    }

    /// Creates a new context with a new database type.
    ///
    /// This will create a new [`Journal`] object.
    pub fn with_db<ODB: Database>(
        self,
        db: ODB,
    ) -> Context<BLOCK, TX, CFG, ODB, Journal<ODB>, CHAIN, LOCAL> {
        let mut journaled_state = Journal::new(db);
        sync_cfg_to_journal(&self.cfg, &mut journaled_state);
        Context {
            tx: self.tx,
            block: self.block,
            cfg: self.cfg,
            journaled_state,
            local: self.local,
            chain: self.chain,
            error: Ok(()),
        }
    }

    /// Creates a new context with a new `DatabaseRef` type.
    pub fn with_ref_db<ODB: DatabaseRef>(
        self,
        db: ODB,
    ) -> Context<BLOCK, TX, CFG, WrapDatabaseRef<ODB>, Journal<WrapDatabaseRef<ODB>>, CHAIN, LOCAL>
    {
        let mut journaled_state = Journal::new(WrapDatabaseRef(db));
        sync_cfg_to_journal(&self.cfg, &mut journaled_state);
        Context {
            tx: self.tx,
            block: self.block,
            cfg: self.cfg,
            journaled_state,
            local: self.local,
            chain: self.chain,
            error: Ok(()),
        }
    }

    /// Creates a new context with a new block type.
    pub fn with_block<OB: Block>(
        self,
        block: OB,
    ) -> Context<OB, TX, CFG, DB, JOURNAL, CHAIN, LOCAL> {
        Context {
            tx: self.tx,
            block,
            cfg: self.cfg,
            journaled_state: self.journaled_state,
            local: self.local,
            chain: self.chain,
            error: Ok(()),
        }
    }
    /// Creates a new context with a new transaction type.
    pub fn with_tx<OTX: Transaction>(
        self,
        tx: OTX,
    ) -> Context<BLOCK, OTX, CFG, DB, JOURNAL, CHAIN, LOCAL> {
        Context {
            tx,
            block: self.block,
            cfg: self.cfg,
            journaled_state: self.journaled_state,
            local: self.local,
            chain: self.chain,
            error: Ok(()),
        }
    }

    /// Creates a new context with a new chain type.
    pub fn with_chain<OC>(self, chain: OC) -> Context<BLOCK, TX, CFG, DB, JOURNAL, OC, LOCAL> {
        Context {
            tx: self.tx,
            block: self.block,
            cfg: self.cfg,
            journaled_state: self.journaled_state,
            local: self.local,
            chain,
            error: Ok(()),
        }
    }

    /// Creates a new context with a new chain type.
    pub fn with_cfg<OCFG: Cfg>(
        mut self,
        cfg: OCFG,
    ) -> Context<BLOCK, TX, OCFG, DB, JOURNAL, CHAIN, LOCAL> {
        sync_cfg_to_journal(&cfg, &mut self.journaled_state);
        Context {
            tx: self.tx,
            block: self.block,
            cfg,
            journaled_state: self.journaled_state,
            local: self.local,
            chain: self.chain,
            error: Ok(()),
        }
    }

    /// Creates a new context with a new local context type.
    pub fn with_local<OL: LocalContextTr>(
        self,
        local: OL,
    ) -> Context<BLOCK, TX, CFG, DB, JOURNAL, CHAIN, OL> {
        Context {
            tx: self.tx,
            block: self.block,
            cfg: self.cfg,
            journaled_state: self.journaled_state,
            local,
            chain: self.chain,
            error: Ok(()),
        }
    }

    /// Modifies the context configuration.
    #[must_use]
    pub fn modify_cfg_chained<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut CFG),
    {
        f(&mut self.cfg);
        sync_cfg_to_journal(&self.cfg, &mut self.journaled_state);
        self
    }

    /// Modifies the context block.
    #[must_use]
    pub fn modify_block_chained<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut BLOCK),
    {
        self.modify_block(f);
        self
    }

    /// Modifies the context transaction.
    #[must_use]
    pub fn modify_tx_chained<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut TX),
    {
        self.modify_tx(f);
        self
    }

    /// Modifies the context chain.
    #[must_use]
    pub fn modify_chain_chained<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut CHAIN),
    {
        self.modify_chain(f);
        self
    }

    /// Modifies the context database.
    #[must_use]
    pub fn modify_db_chained<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut DB),
    {
        self.modify_db(f);
        self
    }

    /// Modifies the context journal.
    #[must_use]
    pub fn modify_journal_chained<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut JOURNAL),
    {
        self.modify_journal(f);
        self
    }

    /// Modifies the context block.
    pub fn modify_block<F>(&mut self, f: F)
    where
        F: FnOnce(&mut BLOCK),
    {
        f(&mut self.block);
    }

    /// Modifies the context transaction.
    pub fn modify_tx<F>(&mut self, f: F)
    where
        F: FnOnce(&mut TX),
    {
        f(&mut self.tx);
    }

    /// Modifies the context configuration.
    pub fn modify_cfg<F>(&mut self, f: F)
    where
        F: FnOnce(&mut CFG),
    {
        f(&mut self.cfg);
        sync_cfg_to_journal(&self.cfg, &mut self.journaled_state);
    }

    /// Modifies the context chain.
    pub fn modify_chain<F>(&mut self, f: F)
    where
        F: FnOnce(&mut CHAIN),
    {
        f(&mut self.chain);
    }

    /// Modifies the context database.
    pub fn modify_db<F>(&mut self, f: F)
    where
        F: FnOnce(&mut DB),
    {
        f(self.journaled_state.db_mut());
    }

    /// Modifies the context journal.
    pub fn modify_journal<F>(&mut self, f: F)
    where
        F: FnOnce(&mut JOURNAL),
    {
        f(&mut self.journaled_state);
    }

    /// Modifies the local context.
    pub fn modify_local<F>(&mut self, f: F)
    where
        F: FnOnce(&mut LOCAL),
    {
        f(&mut self.local);
    }
}

impl<
        BLOCK: Block,
        TX: Transaction,
        CFG: Cfg,
        DB: Database,
        JOURNAL: JournalTr<Database = DB>,
        CHAIN,
        LOCAL: LocalContextTr,
    > Host for Context<BLOCK, TX, CFG, DB, JOURNAL, CHAIN, LOCAL>
{
    /* Block */

    fn basefee(&self) -> U256 {
        U256::from(self.block().basefee())
    }

    fn blob_gasprice(&self) -> U256 {
        U256::from(self.block().blob_gasprice().unwrap_or(0))
    }

    fn gas_limit(&self) -> U256 {
        U256::from(self.block().gas_limit())
    }

    fn difficulty(&self) -> U256 {
        self.block().difficulty()
    }

    fn prevrandao(&self) -> Option<U256> {
        self.block().prevrandao().map(|r| r.into())
    }

    #[inline]
    fn gas_params(&self) -> &GasParams {
        self.cfg().gas_params()
    }

    fn is_amsterdam_eip8037_enabled(&self) -> bool {
        self.cfg().is_amsterdam_eip8037_enabled()
    }

    fn is_eip7819_enabled(&self) -> bool {
        self.cfg().is_eip7819_enabled()
    }

    fn is_eip7851_enabled(&self) -> bool {
        self.cfg().is_eip7851_enabled() && self.cfg().spec().into().is_enabled_in(SpecId::PRAGUE)
    }

    fn block_number(&self) -> U256 {
        self.block().number()
    }

    fn timestamp(&self) -> U256 {
        U256::from(self.block().timestamp())
    }

    fn beneficiary(&self) -> Address {
        self.block().beneficiary()
    }

    fn slot_num(&self) -> U256 {
        U256::from(self.block().slot_num())
    }

    fn chain_id(&self) -> U256 {
        U256::from(self.cfg().chain_id())
    }

    /* Transaction */

    /* EIP-8141 frame transaction */

    fn effective_gas_price(&self) -> U256 {
        let basefee = self.block().basefee();
        U256::from(self.tx().effective_gas_price(basefee as u128))
    }

    fn caller(&self) -> Address {
        self.tx().caller()
    }

    fn blob_hash(&self, number: usize) -> Option<U256> {
        let tx = &self.tx();
        if tx.tx_type() != TransactionType::Eip4844 {
            return None;
        }
        tx.blob_versioned_hashes()
            .get(number)
            .map(|t| U256::from_be_bytes(t.0))
    }

    /* Config */

    fn max_initcode_size(&self) -> usize {
        self.cfg().max_initcode_size()
    }

    /* Database */

    fn block_hash(&mut self, requested_number: u64) -> Option<B256> {
        self.db_mut()
            .block_hash(requested_number)
            .map_err(|e| {
                cold_path();
                *self.error() = Err(e.into());
            })
            .ok()
    }

    /* Journal */

    /// Gets the transient storage value of `address` at `index`.
    fn tload(&mut self, address: Address, index: StorageKey) -> StorageValue {
        self.journal_mut().tload(address, index)
    }

    /// Sets the transient storage value of `address` at `index`.
    fn tstore(&mut self, address: Address, index: StorageKey, value: StorageValue) {
        self.journal_mut().tstore(address, index, value)
    }

    /// Emits a log owned by `address` with given `LogData`.
    fn log(&mut self, log: Log) {
        self.journal_mut().log(log);
    }

    fn set_delegate(
        &mut self,
        location: Address,
        target: Address,
    ) -> Option<Result<bool, SetDelegateError>> {
        let mut account = match self
            .journaled_state
            .load_account_mut_optional_code(location, true)
        {
            Ok(account) => account,
            Err(err) => {
                cold_path();
                self.error = Err(err.into());
                return None;
            }
        };
        // EIP-7523 forbids empty trie accounts on networks where SETDELEGATE can run. Checking
        // account contents also handles databases that materialize missing addresses as defaults.
        let existed = !account.account().is_empty();
        let collision = account.code().is_some_and(|code| {
            let raw = code.original_byte_slice();
            !raw.is_empty() && !raw.starts_with(&DELEGATION_PREFIX)
        });
        if collision {
            return Some(Err(SetDelegateError::AddressCollision));
        }

        let code = if target.is_zero() {
            Bytecode::default()
        } else {
            Bytecode::new_eip7702(target)
        };
        account.set_code_and_hash_slow(code);
        if account.nonce() == 0 {
            account.set_nonce(1);
        }
        Some(Ok(existed))
    }

    fn set_self_delegate(&mut self, authority: Address, target: Address) -> Option<bool> {
        if target.is_zero() {
            return Some(false);
        }
        let mut account = match self
            .journaled_state
            .load_account_mut_optional_code(authority, true)
        {
            Ok(account) => account,
            Err(err) => {
                cold_path();
                self.error = Err(err.into());
                return None;
            }
        };
        if !account.code().is_some_and(Bytecode::is_delegation) {
            return Some(false);
        }

        account.set_code_and_hash_slow(Bytecode::new_eip7851(target));
        Some(true)
    }

    /// Marks `address` to be deleted, with funds transferred to `target`.
    #[inline]
    fn selfdestruct(
        &mut self,
        address: Address,
        target: Address,
        skip_cold_load: bool,
    ) -> Result<StateLoad<SelfDestructResult>, LoadError> {
        self.journal_mut()
            .selfdestruct(address, target, skip_cold_load)
            .map_err(|e| {
                cold_path();
                let (ret, err) = e.into_parts();
                if let Some(err) = err {
                    *self.error() = Err(err.into());
                }
                ret
            })
    }

    #[inline]
    fn sstore_skip_cold_load(
        &mut self,
        address: Address,
        key: StorageKey,
        value: StorageValue,
        skip_cold_load: bool,
    ) -> Result<StateLoad<SStoreResult>, LoadError> {
        self.journal_mut()
            .sstore_skip_cold_load(address, key, value, skip_cold_load)
            .map_err(|e| {
                cold_path();
                let (ret, err) = e.into_parts();
                if let Some(err) = err {
                    *self.error() = Err(err.into());
                }
                ret
            })
    }

    #[inline]
    fn sload_skip_cold_load(
        &mut self,
        address: Address,
        key: StorageKey,
        skip_cold_load: bool,
    ) -> Result<StateLoad<StorageValue>, LoadError> {
        self.journal_mut()
            .sload_skip_cold_load(address, key, skip_cold_load)
            .map_err(|e| {
                cold_path();
                let (ret, err) = e.into_parts();
                if let Some(err) = err {
                    *self.error() = Err(err.into());
                }
                ret
            })
    }

    #[inline]
    fn load_account_info_skip_cold_load(
        &mut self,
        address: Address,
        load_code: bool,
        skip_cold_load: bool,
    ) -> Result<AccountInfoLoad<'_>, LoadError> {
        match self.journaled_state.load_account_info_skip_cold_load(
            address,
            load_code,
            skip_cold_load,
        ) {
            Ok(a) => Ok(a),
            Err(e) => {
                cold_path();
                let (ret, err) = e.into_parts();
                if let Some(err) = err {
                    self.error = Err(err.into());
                }
                Err(ret)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use context_interface::host::{LoadError, SetDelegateError};
    use database::{CacheDB, EmptyDB};
    use primitives::{address, eip7819, hardfork::SpecId, Bytes, KECCAK_EMPTY};
    use state::{AccountInfo, Bytecode};

    #[test]
    fn host_sload_loads_an_untouched_address_and_honors_skip_cold() {
        let address = address!("1000000000000000000000000000000000000000");
        let key = U256::from(7u64);
        let value = U256::from(99u64);
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(
            address,
            AccountInfo {
                nonce: 1,
                ..Default::default()
            },
        );
        db.insert_account_storage(address, key, value).unwrap();
        let mut context: Context<BlockEnv, TxEnv, CfgEnv, CacheDB<EmptyDB>> =
            Context::new(db, SpecId::BERLIN);

        assert_eq!(
            context.sload_skip_cold_load(address, key, true),
            Err(LoadError::ColdLoadSkipped)
        );

        let cold = context
            .sload_skip_cold_load(address, key, false)
            .expect("untouched account and storage load from the database");
        assert!(cold.is_cold);
        assert_eq!(cold.data, value);

        assert_eq!(
            context.load_account_info_skip_cold_load(address, false, true),
            Err(LoadError::ColdLoadSkipped),
            "warming a storage key must not warm its account"
        );
        assert!(
            context
                .load_account_info_skip_cold_load(address, false, false)
                .expect("account can still be cold-loaded")
                .is_cold
        );

        let warm = context
            .sload_skip_cold_load(address, key, true)
            .expect("the previously loaded storage slot is warm");
        assert!(!warm.is_cold);
        assert_eq!(warm.data, value);
    }

    #[test]
    fn setdelegate_journals_updates_clearing_warming_and_revert() {
        let execution_address = address!("1111111111111111111111111111111111111111");
        let target = address!("2222222222222222222222222222222222222222");
        let replacement = address!("3333333333333333333333333333333333333333");
        let location = eip7819::setdelegate_address(execution_address, U256::ZERO);
        let mut context: Context<BlockEnv, TxEnv, CfgEnv, CacheDB<EmptyDB>> =
            Context::new(CacheDB::default(), SpecId::PRAGUE);
        context.cfg.enable_eip7819 = true;
        let checkpoint = context.journal_mut().checkpoint();

        assert_eq!(
            Host::set_delegate(&mut context, location, target),
            Some(Ok(false))
        );
        {
            let account = context
                .journal_mut()
                .load_account_with_code(location)
                .unwrap();
            assert_eq!(account.info.nonce, 1);
            assert_eq!(
                account.info.code.as_ref().unwrap().eip7702_address(),
                Some(target)
            );
        }
        assert!(
            Host::load_account_info_skip_cold_load(&mut context, location, false, true).is_ok(),
            "SETDELEGATE did not warm its location"
        );
        assert_eq!(
            Host::load_account_info_skip_cold_load(&mut context, target, false, true),
            Err(LoadError::ColdLoadSkipped),
            "SETDELEGATE warmed its target"
        );

        assert_eq!(
            Host::set_delegate(&mut context, location, replacement),
            Some(Ok(true))
        );
        {
            let account = context
                .journal_mut()
                .load_account_with_code(location)
                .unwrap();
            assert_eq!(account.info.nonce, 1, "replacement incremented the nonce");
            assert_eq!(
                account.info.code.as_ref().unwrap().eip7702_address(),
                Some(replacement)
            );
        }
        assert_eq!(
            Host::set_delegate(&mut context, location, Address::ZERO),
            Some(Ok(true))
        );
        {
            let account = context
                .journal_mut()
                .load_account_with_code(location)
                .unwrap();
            assert_eq!(account.info.nonce, 1);
            assert_eq!(account.info.code_hash, KECCAK_EMPTY);
            assert!(account.info.code.as_ref().unwrap().is_empty());
        }

        context.journal_mut().checkpoint_revert(checkpoint);
        let account = context
            .journal_mut()
            .load_account_with_code(location)
            .unwrap();
        assert_eq!(account.info.nonce, 0);
        assert!(account.info.code.as_ref().unwrap().is_empty());
        assert!(account.is_loaded_as_not_existing_not_touched());
    }

    #[test]
    fn setdelegate_does_not_refund_a_materialized_empty_account() {
        let location = address!("1000000000000000000000000000000000000000");
        let target = address!("2000000000000000000000000000000000000000");
        let replacement = address!("3000000000000000000000000000000000000000");
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(location, AccountInfo::default());
        let mut context: Context<BlockEnv, TxEnv, CfgEnv, CacheDB<EmptyDB>> =
            Context::new(db, SpecId::PRAGUE);

        assert_eq!(
            Host::set_delegate(&mut context, location, target),
            Some(Ok(false))
        );
        assert_eq!(
            Host::set_delegate(&mut context, location, replacement),
            Some(Ok(true))
        );
    }

    #[test]
    fn setdelegate_preserves_existing_state_and_rejects_only_ordinary_code() {
        let existing = address!("1000000000000000000000000000000000000000");
        let collision = address!("2000000000000000000000000000000000000000");
        let replaceable = address!("3000000000000000000000000000000000000000");
        let target = address!("4444444444444444444444444444444444444444");
        let key = U256::from(7u64);
        let value = U256::from(9u64);
        let ordinary_code = Bytecode::new_legacy(Bytes::from_static(&[0x00]));
        let prefixed_code = Bytecode::new_legacy(Bytes::from_static(&[0xef, 0x01, 0x00, 0xff]));
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(
            existing,
            AccountInfo {
                balance: U256::from(5u64),
                ..Default::default()
            },
        );
        db.insert_account_storage(existing, key, value).unwrap();
        db.insert_account_info(
            collision,
            AccountInfo {
                code_hash: ordinary_code.hash_slow(),
                code: Some(ordinary_code.clone()),
                ..Default::default()
            },
        );
        db.insert_account_info(
            replaceable,
            AccountInfo {
                nonce: 7,
                code_hash: prefixed_code.hash_slow(),
                code: Some(prefixed_code),
                ..Default::default()
            },
        );
        let mut context: Context<BlockEnv, TxEnv, CfgEnv, CacheDB<EmptyDB>> =
            Context::new(db, SpecId::PRAGUE);

        assert_eq!(
            Host::set_delegate(&mut context, existing, target),
            Some(Ok(true))
        );
        {
            let account = context
                .journal_mut()
                .load_account_with_code(existing)
                .unwrap();
            assert_eq!(account.info.balance, U256::from(5u64));
            assert_eq!(account.info.nonce, 1);
            assert_eq!(
                account.info.code.as_ref().unwrap().eip7702_address(),
                Some(target)
            );
        }
        assert_eq!(
            context.journal_mut().sload(existing, key).unwrap().data,
            value
        );

        assert_eq!(
            Host::set_delegate(&mut context, collision, target),
            Some(Err(SetDelegateError::AddressCollision))
        );
        let account = context
            .journal_mut()
            .load_account_with_code(collision)
            .unwrap();
        assert_eq!(
            account.info.code.as_ref().unwrap().original_byte_slice(),
            &[0x00]
        );

        assert_eq!(
            Host::set_delegate(&mut context, replaceable, target),
            Some(Ok(true))
        );
        let account = context
            .journal_mut()
            .load_account_with_code(replaceable)
            .unwrap();
        assert_eq!(account.info.nonce, 7);
        assert_eq!(
            account.info.code.as_ref().unwrap().eip7702_address(),
            Some(target)
        );
    }

    #[test]
    fn setselfdelegate_accepts_exact_enabled_and_disabled_designations_and_reverts() {
        let authority = address!("1000000000000000000000000000000000000011");
        let old_target = address!("2000000000000000000000000000000000000022");
        let new_target = address!("3000000000000000000000000000000000000033");

        for original in [
            Bytecode::new_eip7702(old_target),
            Bytecode::new_eip7851(old_target),
        ] {
            let mut db = CacheDB::<EmptyDB>::default();
            db.insert_account_info(
                authority,
                AccountInfo {
                    nonce: 7,
                    code_hash: original.hash_slow(),
                    code: Some(original.clone()),
                    ..Default::default()
                },
            );
            db.insert_account_info(
                new_target,
                AccountInfo {
                    nonce: 1,
                    ..Default::default()
                },
            );
            let mut context: Context<BlockEnv, TxEnv, CfgEnv, CacheDB<EmptyDB>> =
                Context::new(db, SpecId::PRAGUE);
            let checkpoint = context.journal_mut().checkpoint();

            assert_eq!(
                Host::set_self_delegate(&mut context, authority, new_target),
                Some(true)
            );
            {
                let account = context
                    .journal_mut()
                    .load_account_with_code(authority)
                    .unwrap();
                assert_eq!(account.info.nonce, 7, "SETSELFDELEGATE bumped nonce");
                assert_eq!(
                    account.info.code.as_ref().unwrap().eip7851_address(),
                    Some(new_target)
                );
            }
            assert_eq!(
                Host::load_account_info_skip_cold_load(&mut context, new_target, false, true),
                Err(LoadError::ColdLoadSkipped),
                "SETSELFDELEGATE loaded or warmed its new target"
            );

            context.journal_mut().checkpoint_revert(checkpoint);
            let account = context
                .journal_mut()
                .load_account_with_code(authority)
                .unwrap();
            assert_eq!(account.info.nonce, 7);
            assert_eq!(account.info.code.as_ref(), Some(&original));
        }
    }

    #[test]
    fn setselfdelegate_zero_and_invalid_raw_authority_do_not_mutate() {
        let authority = address!("1000000000000000000000000000000000000011");
        let target = address!("2000000000000000000000000000000000000022");
        let valid = Bytecode::new_eip7702(target);
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(
            authority,
            AccountInfo {
                nonce: 9,
                code_hash: valid.hash_slow(),
                code: Some(valid),
                ..Default::default()
            },
        );
        let mut context: Context<BlockEnv, TxEnv, CfgEnv, CacheDB<EmptyDB>> =
            Context::new(db, SpecId::PRAGUE);
        assert_eq!(
            Host::set_self_delegate(&mut context, authority, Address::ZERO),
            Some(false)
        );
        assert!(
            context.journal().evm_state().get(&authority).is_none(),
            "zero target loaded or mutated the authority"
        );

        let mut malformed = vec![0xef, 0x01, 0x00];
        malformed.extend_from_slice(&[0x44; 21]);
        for invalid in [
            Bytecode::new_legacy(Bytes::from_static(&[0x00])),
            Bytecode::new_legacy(malformed.into()),
        ] {
            let mut db = CacheDB::<EmptyDB>::default();
            db.insert_account_info(
                authority,
                AccountInfo {
                    nonce: 9,
                    code_hash: invalid.hash_slow(),
                    code: Some(invalid.clone()),
                    ..Default::default()
                },
            );
            let mut context: Context<BlockEnv, TxEnv, CfgEnv, CacheDB<EmptyDB>> =
                Context::new(db, SpecId::PRAGUE);

            assert_eq!(
                Host::set_self_delegate(&mut context, authority, target),
                Some(false)
            );
            let account = context
                .journal_mut()
                .load_account_with_code(authority)
                .unwrap();
            assert_eq!(account.info.nonce, 9);
            assert_eq!(account.info.code.as_ref(), Some(&invalid));
        }
    }

    #[test]
    fn setdelegate_treats_eip7851_as_an_address_collision() {
        let location = address!("1000000000000000000000000000000000000011");
        let old_target = address!("2000000000000000000000000000000000000022");
        let new_target = address!("3000000000000000000000000000000000000033");
        let disabled = Bytecode::new_eip7851(old_target);
        let mut db = CacheDB::<EmptyDB>::default();
        db.insert_account_info(
            location,
            AccountInfo {
                nonce: 4,
                code_hash: disabled.hash_slow(),
                code: Some(disabled.clone()),
                ..Default::default()
            },
        );
        let mut context: Context<BlockEnv, TxEnv, CfgEnv, CacheDB<EmptyDB>> =
            Context::new(db, SpecId::PRAGUE);

        assert_eq!(
            Host::set_delegate(&mut context, location, new_target),
            Some(Err(SetDelegateError::AddressCollision))
        );
        let account = context
            .journal_mut()
            .load_account_with_code(location)
            .unwrap();
        assert_eq!(account.info.nonce, 4);
        assert_eq!(account.info.code.as_ref(), Some(&disabled));
    }
}
