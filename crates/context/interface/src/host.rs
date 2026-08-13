//! Host interface for external blockchain state access.

use crate::{
    cfg::GasParams,
    context::{SStoreResult, SelfDestructResult, StateLoad},
    journaled_state::{AccountInfoLoad, AccountLoad},
};
use auto_impl::auto_impl;
use primitives::{hardfork::SpecId, Address, Bytes, Log, StorageKey, StorageValue, B256, U256};
use state::Bytecode;

/// Error that can happen when loading account info.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LoadError {
    /// Cold load skipped.
    ColdLoadSkipped,
    /// Database error.
    DBError,
}

/// EIP-8141 frame transaction context.
///
/// Frame transactions decompose a transaction into frames that validate it,
/// approve gas payment and execute user operations. The introspection opcodes
/// (`TXPARAM`, `FRAMEPARAM`, `SIGPARAM`, `FRAMEDATALOAD`, `FRAMEDATACOPY`) read
/// from this context; outside a frame transaction it is absent and those opcodes
/// halt exceptionally.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameTxContext {
    /// The declared sender of the transaction.
    pub sender: Address,
    /// Sender nonce.
    pub nonce: u64,
    /// Canonical signature hash, `TXPARAM(0x08)`.
    pub sig_hash: B256,
    /// Maximum cost the payer may be charged, `TXPARAM(0x06)`.
    pub max_cost: U256,
    /// `TXPARAM(0x03)`.
    pub max_priority_fee_per_gas: U256,
    /// `TXPARAM(0x04)`.
    pub max_fee_per_gas: U256,
    /// `TXPARAM(0x05)`.
    pub max_fee_per_blob_gas: U256,
    /// Number of blob versioned hashes, `TXPARAM(0x07)`.
    pub blob_count: u64,
    /// Index of the frame currently executing, `TXPARAM(0x0A)`.
    pub frame_index: u64,
    /// Every frame in the transaction, in order.
    pub frames: Vec<FrameInfo>,
    /// Every signature entry in the transaction, in order.
    pub signatures: Vec<FrameSigInfo>,
    /// Scopes `APPROVE` is permitted to grant, mirroring `frame.flags & 0x3`.
    pub approvable_scopes: u64,
    /// Scope `APPROVE` actually granted, or 0. Lets a caller assert what was
    /// approved rather than only that the frame did not revert.
    pub approved_scope: u64,
}

/// A single frame within a frame transaction, as seen by `FRAMEPARAM`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameInfo {
    /// Target after resolving a null target to `tx.sender`.
    pub resolved_target: Address,
    /// Gas limit allotted to this frame.
    pub gas_limit: u64,
    /// Frame mode: 0 DEFAULT, 1 VERIFY, 2 SENDER.
    pub mode: u8,
    /// Frame flags.
    pub flags: u8,
    /// Value transferred by the frame.
    pub value: U256,
    /// Execution status: 0 failed, 1 success, 2 skipped. Only meaningful for a
    /// frame that has already run.
    pub status: u8,
    /// Calldata supplied to the frame.
    pub data: Bytes,
}

/// A signature entry, as seen by `SIGPARAM`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameSigInfo {
    /// Signer after resolving an absent signer to `tx.sender`. `None` for
    /// `ARBITRARY` entries, which have no protocol-assigned signer.
    pub resolved_signer: Option<Address>,
    /// Signature scheme: 0 ARBITRARY, 1 SECP256K1, 2 P256.
    pub scheme: u8,
    /// Explicit 32-byte digest, or zero when the entry signs the canonical hash.
    pub msg: B256,
    /// Raw signature bytes. Only readable for `ARBITRARY` entries.
    pub signature: Bytes,
}

/// Host trait with all methods that are needed by the Interpreter.
///
/// This trait is implemented for all types that have `ContextTr` trait.
///
/// There are few groups of functions which are Block, Transaction, Config, Database and Journal functions.
#[auto_impl(&mut, Box)]
pub trait Host {
    /* Block */

    /// Block basefee, calls ContextTr::block().basefee()
    fn basefee(&self) -> U256;
    /// Block blob gasprice, calls `ContextTr::block().blob_gasprice()`
    fn blob_gasprice(&self) -> U256;
    /// Block gas limit, calls ContextTr::block().gas_limit()
    fn gas_limit(&self) -> U256;
    /// Block difficulty, calls ContextTr::block().difficulty()
    fn difficulty(&self) -> U256;
    /// Block prevrandao, calls ContextTr::block().prevrandao()
    fn prevrandao(&self) -> Option<U256>;
    /// Block number, calls ContextTr::block().number()
    fn block_number(&self) -> U256;
    /// Block timestamp, calls ContextTr::block().timestamp()
    fn timestamp(&self) -> U256;
    /// Block beneficiary, calls ContextTr::block().beneficiary()
    fn beneficiary(&self) -> Address;
    /// Block slot number, calls ContextTr::block().slot_num()
    fn slot_num(&self) -> U256;
    /// Chain id, calls ContextTr::cfg().chain_id()
    fn chain_id(&self) -> U256;

    /* Transaction */

    /// Transaction effective gas price, calls `ContextTr::tx().effective_gas_price(basefee as u128)`
    fn effective_gas_price(&self) -> U256;
    /// Transaction caller, calls `ContextTr::tx().caller()`
    fn caller(&self) -> Address;
    /// Transaction blob hash, calls `ContextTr::tx().blob_hash(number)`
    fn blob_hash(&self, number: usize) -> Option<U256>;

    /* Config */

    /// Max initcode size, calls `ContextTr::cfg().max_code_size().saturating_mul(2)`
    fn max_initcode_size(&self) -> usize;

    /// Gas params contains the dynamic gas constants for the EVM.
    fn gas_params(&self) -> &GasParams;

    /// Returns whether state gas (EIP-8037) is enabled.
    fn is_amsterdam_eip8037_enabled(&self) -> bool;

    /* EIP-8141 frame transaction */

    /// Frame transaction context, or `None` when this is not a frame transaction.
    ///
    /// Defaults to `None` so that hosts which do not model frame transactions
    /// compile unchanged; the frame opcodes then halt exceptionally, which is
    /// what the spec requires outside a frame transaction.
    fn frame_context(&self) -> Option<&FrameTxContext> {
        None
    }

    /// Applies `APPROVE` for the given scope, returning whether it succeeded.
    ///
    /// Defaults to rejecting, so a host that has not opted in cannot silently
    /// approve payment or execution.
    fn frame_approve(&mut self, _scope: u64) -> bool {
        false
    }

    /* Database */

    /// Block hash, calls `ContextTr::journal_mut().db().block_hash(number)`
    fn block_hash(&mut self, number: u64) -> Option<B256>;

    /* Journal */

    /// Selfdestruct account, calls `ContextTr::journal_mut().selfdestruct(address, target)`
    fn selfdestruct(
        &mut self,
        address: Address,
        target: Address,
        skip_cold_load: bool,
    ) -> Result<StateLoad<SelfDestructResult>, LoadError>;

    /// Log, calls `ContextTr::journal_mut().log(log)`
    fn log(&mut self, log: Log);

    /// Sstore with optional fetch from database. Return none if the value is cold or if there is db error.
    fn sstore_skip_cold_load(
        &mut self,
        address: Address,
        key: StorageKey,
        value: StorageValue,
        skip_cold_load: bool,
    ) -> Result<StateLoad<SStoreResult>, LoadError>;

    /// Sstore, calls `ContextTr::journal_mut().sstore(address, key, value)`
    fn sstore(
        &mut self,
        address: Address,
        key: StorageKey,
        value: StorageValue,
    ) -> Option<StateLoad<SStoreResult>> {
        self.sstore_skip_cold_load(address, key, value, false).ok()
    }

    /// Sload with optional fetch from database. Return none if the value is cold or if there is db error.
    fn sload_skip_cold_load(
        &mut self,
        address: Address,
        key: StorageKey,
        skip_cold_load: bool,
    ) -> Result<StateLoad<StorageValue>, LoadError>;

    /// Sload, calls `ContextTr::journal_mut().sload(address, key)`
    fn sload(&mut self, address: Address, key: StorageKey) -> Option<StateLoad<StorageValue>> {
        self.sload_skip_cold_load(address, key, false).ok()
    }

    /// Tstore, calls `ContextTr::journal_mut().tstore(address, key, value)`
    fn tstore(&mut self, address: Address, key: StorageKey, value: StorageValue);

    /// Tload, calls `ContextTr::journal_mut().tload(address, key)`
    fn tload(&mut self, address: Address, key: StorageKey) -> StorageValue;

    /// Main function to load account info.
    ///
    /// If load_code is true, it will load the code fetching it from the database if not done before.
    ///
    /// If skip_cold_load is true, it will not load the account if it is cold. This is needed to short circuit
    /// the load if there is not enough gas.
    ///
    /// Returns AccountInfo, is_cold and is_empty.
    fn load_account_info_skip_cold_load(
        &mut self,
        address: Address,
        load_code: bool,
        skip_cold_load: bool,
    ) -> Result<AccountInfoLoad<'_>, LoadError>;

    /// Balance, calls `ContextTr::journal_mut().load_account(address)`
    #[inline]
    fn balance(&mut self, address: Address) -> Option<StateLoad<U256>> {
        self.load_account_info_skip_cold_load(address, false, false)
            .ok()
            .map(|load| load.into_state_load(|i| i.balance))
    }

    /// Load account delegated, calls `ContextTr::journal_mut().load_account_delegated(address)`
    #[inline]
    fn load_account_delegated(&mut self, address: Address) -> Option<StateLoad<AccountLoad>> {
        let account = self
            .load_account_info_skip_cold_load(address, true, false)
            .ok()?;

        let mut account_load = StateLoad::new(
            AccountLoad {
                is_delegate_account_cold: None,
                is_empty: account.is_empty,
            },
            account.is_cold,
        );

        // load delegate code if account is EIP-7702
        if let Some(address) = account.code.as_ref().and_then(Bytecode::eip7702_address) {
            let delegate_account = self
                .load_account_info_skip_cold_load(address, true, false)
                .ok()?;
            account_load.data.is_delegate_account_cold = Some(delegate_account.is_cold);
        }

        Some(account_load)
    }

    /// Load account code, calls [`Host::load_account_info_skip_cold_load`] with `load_code` set to false.
    #[inline]
    fn load_account_code(&mut self, address: Address) -> Option<StateLoad<Bytes>> {
        self.load_account_info_skip_cold_load(address, true, false)
            .ok()
            .map(|load| {
                load.into_state_load(|i| {
                    i.code
                        .as_ref()
                        .map(|b| b.original_bytes())
                        .unwrap_or_default()
                })
            })
    }

    /// Load account code hash, calls [`Host::load_account_info_skip_cold_load`] with `load_code` set to false.
    #[inline]
    fn load_account_code_hash(&mut self, address: Address) -> Option<StateLoad<B256>> {
        self.load_account_info_skip_cold_load(address, false, false)
            .ok()
            .map(|load| {
                load.into_state_load(|i| {
                    if i.is_empty() {
                        B256::ZERO
                    } else {
                        i.code_hash
                    }
                })
            })
    }
}

/// Dummy host that implements [`Host`] trait and  returns all default values.
#[derive(Default, Debug)]
pub struct DummyHost {
    gas_params: GasParams,
    spec: SpecId,
    /// Optional EIP-8141 frame transaction context, so tests and non-consensus
    /// hosts can exercise the frame instructions. `None` means "not a frame
    /// transaction", which makes those instructions halt.
    pub frame_tx: Option<FrameTxContext>,
    /// Scopes that [`Host::frame_approve`] will accept, as a bitmask.
    pub approvable_scopes: u64,
}

impl DummyHost {
    /// Create a new dummy host with the given spec.
    pub fn new(spec: SpecId) -> Self {
        Self {
            gas_params: GasParams::new_spec(spec),
            spec,
            ..Default::default()
        }
    }
}

impl DummyHost {
    /// Installs a frame transaction context and permits the given approval scopes.
    pub fn with_frame_tx(mut self, frame_tx: FrameTxContext, approvable_scopes: u64) -> Self {
        self.frame_tx = Some(frame_tx);
        self.approvable_scopes = approvable_scopes;
        self
    }
}

impl Host for DummyHost {
    fn frame_context(&self) -> Option<&FrameTxContext> {
        self.frame_tx.as_ref()
    }

    fn frame_approve(&mut self, scope: u64) -> bool {
        scope != 0 && scope & !self.approvable_scopes == 0
    }

    fn basefee(&self) -> U256 {
        U256::ZERO
    }

    fn blob_gasprice(&self) -> U256 {
        U256::ZERO
    }

    fn gas_limit(&self) -> U256 {
        U256::ZERO
    }

    fn gas_params(&self) -> &GasParams {
        &self.gas_params
    }

    fn is_amsterdam_eip8037_enabled(&self) -> bool {
        self.spec.is_enabled_in(SpecId::AMSTERDAM)
    }

    fn difficulty(&self) -> U256 {
        U256::ZERO
    }

    fn prevrandao(&self) -> Option<U256> {
        None
    }

    fn block_number(&self) -> U256 {
        U256::ZERO
    }

    fn timestamp(&self) -> U256 {
        U256::ZERO
    }

    fn beneficiary(&self) -> Address {
        Address::ZERO
    }

    fn slot_num(&self) -> U256 {
        U256::ZERO
    }

    fn chain_id(&self) -> U256 {
        U256::ZERO
    }

    fn effective_gas_price(&self) -> U256 {
        U256::ZERO
    }

    fn caller(&self) -> Address {
        Address::ZERO
    }

    fn blob_hash(&self, _number: usize) -> Option<U256> {
        None
    }

    fn max_initcode_size(&self) -> usize {
        0
    }

    fn block_hash(&mut self, _number: u64) -> Option<B256> {
        None
    }

    fn selfdestruct(
        &mut self,
        _address: Address,
        _target: Address,
        _skip_cold_load: bool,
    ) -> Result<StateLoad<SelfDestructResult>, LoadError> {
        Ok(Default::default())
    }

    fn log(&mut self, _log: Log) {}

    fn tstore(&mut self, _address: Address, _key: StorageKey, _value: StorageValue) {}

    fn tload(&mut self, _address: Address, _key: StorageKey) -> StorageValue {
        StorageValue::ZERO
    }

    fn load_account_info_skip_cold_load(
        &mut self,
        _address: Address,
        _load_code: bool,
        _skip_cold_load: bool,
    ) -> Result<AccountInfoLoad<'_>, LoadError> {
        Ok(Default::default())
    }

    fn sstore_skip_cold_load(
        &mut self,
        _address: Address,
        _key: StorageKey,
        _value: StorageValue,
        _skip_cold_load: bool,
    ) -> Result<StateLoad<SStoreResult>, LoadError> {
        Ok(Default::default())
    }

    fn sload_skip_cold_load(
        &mut self,
        _address: Address,
        _key: StorageKey,
        _skip_cold_load: bool,
    ) -> Result<StateLoad<StorageValue>, LoadError> {
        Ok(Default::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitives::{hardfork::SpecId, Address, U256};
    use state::{AccountInfo, Bytecode};
    use std::borrow::Cow;

    /// Host used to regression-test [`Host::load_account_delegated`].
    ///
    /// `delegated` is a non-empty EIP-7702 account pointing at an empty `delegate`.
    struct Eip7702Host {
        dummy: DummyHost,
        delegated: Address,
        delegate: Address,
        delegated_info: AccountInfo,
    }

    impl Eip7702Host {
        fn new() -> Self {
            let delegated = Address::repeat_byte(0x11);
            let delegate = Address::repeat_byte(0x22);
            let delegated_info = AccountInfo::new(
                U256::from(1),
                1,
                B256::ZERO,
                Bytecode::new_eip7702(delegate),
            );
            Self {
                dummy: DummyHost::new(SpecId::PRAGUE),
                delegated,
                delegate,
                delegated_info,
            }
        }
    }

    impl Host for Eip7702Host {
        fn basefee(&self) -> U256 {
            self.dummy.basefee()
        }
        fn blob_gasprice(&self) -> U256 {
            self.dummy.blob_gasprice()
        }
        fn gas_limit(&self) -> U256 {
            self.dummy.gas_limit()
        }
        fn gas_params(&self) -> &GasParams {
            self.dummy.gas_params()
        }
        fn is_amsterdam_eip8037_enabled(&self) -> bool {
            self.dummy.is_amsterdam_eip8037_enabled()
        }
        fn difficulty(&self) -> U256 {
            self.dummy.difficulty()
        }
        fn prevrandao(&self) -> Option<U256> {
            self.dummy.prevrandao()
        }
        fn block_number(&self) -> U256 {
            self.dummy.block_number()
        }
        fn timestamp(&self) -> U256 {
            self.dummy.timestamp()
        }
        fn beneficiary(&self) -> Address {
            self.dummy.beneficiary()
        }
        fn slot_num(&self) -> U256 {
            self.dummy.slot_num()
        }
        fn chain_id(&self) -> U256 {
            self.dummy.chain_id()
        }
        fn effective_gas_price(&self) -> U256 {
            self.dummy.effective_gas_price()
        }
        fn caller(&self) -> Address {
            self.dummy.caller()
        }
        fn blob_hash(&self, number: usize) -> Option<U256> {
            self.dummy.blob_hash(number)
        }
        fn max_initcode_size(&self) -> usize {
            self.dummy.max_initcode_size()
        }
        fn block_hash(&mut self, number: u64) -> Option<B256> {
            self.dummy.block_hash(number)
        }
        fn selfdestruct(
            &mut self,
            address: Address,
            target: Address,
            skip_cold_load: bool,
        ) -> Result<StateLoad<SelfDestructResult>, LoadError> {
            self.dummy.selfdestruct(address, target, skip_cold_load)
        }
        fn log(&mut self, log: Log) {
            self.dummy.log(log)
        }
        fn tstore(&mut self, address: Address, key: StorageKey, value: StorageValue) {
            self.dummy.tstore(address, key, value)
        }
        fn tload(&mut self, address: Address, key: StorageKey) -> StorageValue {
            self.dummy.tload(address, key)
        }
        fn sstore_skip_cold_load(
            &mut self,
            address: Address,
            key: StorageKey,
            value: StorageValue,
            skip_cold_load: bool,
        ) -> Result<StateLoad<SStoreResult>, LoadError> {
            self.dummy
                .sstore_skip_cold_load(address, key, value, skip_cold_load)
        }
        fn sload_skip_cold_load(
            &mut self,
            address: Address,
            key: StorageKey,
            skip_cold_load: bool,
        ) -> Result<StateLoad<StorageValue>, LoadError> {
            self.dummy
                .sload_skip_cold_load(address, key, skip_cold_load)
        }

        fn load_account_info_skip_cold_load(
            &mut self,
            address: Address,
            _load_code: bool,
            _skip_cold_load: bool,
        ) -> Result<AccountInfoLoad<'_>, LoadError> {
            if address == self.delegated {
                Ok(AccountInfoLoad {
                    account: Cow::Owned(self.delegated_info.clone()),
                    is_cold: false,
                    is_empty: false,
                })
            } else if address == self.delegate {
                // Empty delegated target: must not overwrite the caller's `is_empty`.
                Ok(AccountInfoLoad {
                    account: Cow::Owned(AccountInfo::default()),
                    is_cold: true,
                    is_empty: true,
                })
            } else {
                Ok(Default::default())
            }
        }
    }

    #[test]
    fn load_account_delegated_keeps_caller_is_empty_not_delegate() {
        // Regression: previously `is_empty` was overwritten with the empty
        // delegate account's flag. Gas accounting / account-creation costs
        // must use the EIP-7702 account itself (non-empty here).
        let mut host = Eip7702Host::new();
        let load = host
            .load_account_delegated(host.delegated)
            .expect("delegated account loads");

        assert!(
            load.data.is_delegate_account_cold.is_some(),
            "delegate account must be loaded"
        );
        assert!(
            !load.data.is_empty,
            "is_empty must stay false for the non-empty EIP-7702 account"
        );
    }
}
