//! Host interface for external blockchain state access.

use crate::{
    cfg::GasParams,
    context::{SStoreResult, SelfDestructResult, StateLoad},
    journaled_state::{AccountInfoLoad, AccountLoad},
};
use auto_impl::auto_impl;
use primitives::{hardfork::SpecId, Address, Bytes, Log, StorageKey, StorageValue, B256, U256};
use state::AccountInfo;
use std::{sync::Arc, vec::Vec};

/// Error that can happen when loading account info.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LoadError {
    /// Cold load skipped.
    ColdLoadSkipped,
    /// Database error.
    DBError,
}

/// Exceptional error returned while applying EIP-7819 `SETDELEGATE`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SetDelegateError {
    /// The destination contains non-empty code without the EIP-7702 prefix.
    AddressCollision,
}

/// EIP-8141 frame transaction context.
///
/// Frame transactions decompose a transaction into frames that validate it,
/// approve gas payment and execute user operations. The introspection opcodes
/// (`TXPARAM`, `FRAMEPARAM`, `SIGPARAM`, `FRAMEDATALOAD`, `FRAMEDATACOPY`,
/// `SIGDATACOPY`, `RECENTROOTREFLOAD`, `TXTRACE`, `TXDIFF`, and
/// `EVENTDATACOPY`) read from this context; outside a frame transaction it is
/// absent and those opcodes halt exceptionally.
///
/// This context does not carry a chain ID. Lifecycle callers must validate the
/// synthetic transaction's chain ID against the outer transaction externally.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameTxContext {
    /// The declared sender of the transaction.
    pub sender: Address,
    /// Shared keyed-nonce sequence, `TXPARAM(0x01)`.
    pub nonce: u64,
    /// Sender's legacy account nonce in the transaction pre-state,
    /// fixture `TXPARAM(0x80)`.
    pub legacy_nonce: u64,
    /// Canonically ordered EIP-8250 nonce keys. Their count is fixture
    /// `TXPARAM(0x81)` and the first key is fixture `TXPARAM(0x84)`.
    pub nonce_keys: Vec<U256>,
    /// Canonical hash of `nonce_keys`, fixture `TXPARAM(0x82)`.
    pub nonce_keys_hash: B256,
    /// State gas remaining in the currently executing frame, `TXPARAM(0x0C)`.
    pub state_gas_left: u64,
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
    /// Verified recent-root references in transaction order. Their count is
    /// fixture `TXPARAM(0x83)`.
    pub recent_root_references: Vec<FrameTxRecentRootReference>,
    /// Transaction-local state diff and event trace as of the current frame.
    pub trace: FrameTxTrace,
    /// Scopes `APPROVE` is permitted to grant, mirroring `frame.flags & 0x3`.
    pub approvable_scopes: u64,
    /// Scope `APPROVE` actually granted, or 0. Lets a caller assert what was
    /// approved rather than only that the frame did not revert.
    pub approved_scope: u64,
    /// Frozen event snapshot and derived per-address lookup. This is rebuilt
    /// when the context is prepared for a host and is never accepted from
    /// serialized input.
    #[doc(hidden)]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub event_index: FrameTxEventIndex,
}

impl FrameTxContext {
    /// Freezes event data, rebuilds derived lookups, and wraps this context for
    /// cheap sharing.
    ///
    /// Host implementations that supply frame contexts directly should use
    /// this method before returning the context from [`Host::frame_context`].
    pub fn into_shared(mut self) -> Arc<Self> {
        self.event_index.rebuild(&self.trace.events);
        Arc::new(self)
    }

    /// Returns the number of events emitted by `address`, or `None` if this
    /// context has not had its derived event index prepared.
    pub fn event_count_for_address(&self, address: Address) -> Option<usize> {
        self.event_index.range(address).map(|range| range.len())
    }

    /// Maps an address-local event index to its global transaction event index.
    pub fn event_global_index_for_address(
        &self,
        address: Address,
        local_index: usize,
    ) -> Option<usize> {
        let range = self.event_index.range(address)?;
        let index = range.start.checked_add(local_index)?;
        (index < range.end).then(|| self.event_index.entries[index].1)
    }

    /// Returns the immutable event snapshot used by frame instructions, or
    /// `None` if this context has not been prepared with [`Self::into_shared`].
    pub fn event_snapshot(&self) -> Option<&[FrameTxEvent]> {
        self.event_index.events.as_deref()
    }
}

/// Frozen transaction events and their index, sorted by emitter and then global
/// event index. Address lookups perform two binary partitions over `entries`.
///
/// Its fields are private so external data cannot construct a trusted index.
/// Standard installation paths always rebuild it from [`FrameTxTrace::events`].
#[derive(Clone, Debug, Default)]
#[doc(hidden)]
pub struct FrameTxEventIndex {
    entries: Vec<(Address, usize)>,
    events: Option<Arc<[FrameTxEvent]>>,
}

// This cache is not part of the context's semantic value.
impl PartialEq for FrameTxEventIndex {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for FrameTxEventIndex {}

impl FrameTxEventIndex {
    fn rebuild(&mut self, events: &[FrameTxEvent]) {
        let events: Arc<[FrameTxEvent]> = events.to_vec().into();
        self.entries.clear();
        self.entries.extend(
            events
                .iter()
                .enumerate()
                .map(|(index, event)| (event.address, index)),
        );
        self.entries
            .sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        self.events = Some(events);
    }

    fn range(&self, address: Address) -> Option<core::ops::Range<usize>> {
        self.events.as_ref()?;
        let start = self
            .entries
            .partition_point(|(candidate, _)| *candidate < address);
        let end = self
            .entries
            .partition_point(|(candidate, _)| *candidate <= address);
        Some(start..end)
    }
}

/// A single frame within a frame transaction, as seen by `FRAMEPARAM`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameInfo {
    /// Target after resolving a null target to `tx.sender`.
    pub resolved_target: Address,
    /// Caller expected for the synthetic host call. This is internal binding
    /// metadata and is not exposed through `FRAMEPARAM`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub expected_caller: Address,
    /// Execution gas limit allotted to this frame (`limits.execution`),
    /// `FRAMEPARAM(0x01)`.
    pub gas_limit: u64,
    /// State gas limit allotted to this frame (`limits.state`),
    /// `FRAMEPARAM(0x09)`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub state_gas_limit: u64,
    /// Frame mode: 0 DEFAULT, 1 VERIFY, 2 SENDER, 3 POST_TX.
    pub mode: u8,
    /// Frame flags.
    pub flags: u8,
    /// Value transferred by the frame.
    pub value: U256,
    /// Execution status: 0 failed, 1 success, 2 skipped. Only meaningful for a
    /// frame that has already run.
    pub status: u8,
    /// Execution gas recorded in the frame's receipt (`gas_used.execution`),
    /// `FRAMEPARAM(0x0A)`. Only meaningful for a frame that has already run.
    #[cfg_attr(feature = "serde", serde(default))]
    pub execution_gas_used: u64,
    /// State gas attributed to the frame's receipt (`gas_used.state`),
    /// `FRAMEPARAM(0x0B)`. Only meaningful for a frame that has already run.
    #[cfg_attr(feature = "serde", serde(default))]
    pub state_gas_used: u64,
    /// Calldata supplied to the frame.
    pub data: Bytes,
}

/// A signature entry, as seen by `SIGPARAM` and `SIGDATACOPY`.
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

/// A verified EIP-8272 recent-root reference.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameTxRecentRootReference {
    /// Root source identifier.
    pub source_id: B256,
    /// Consensus slot containing the root.
    pub slot: u64,
    /// Opaque root committed by the source.
    pub root: B256,
}

/// One net balance change, ordered by ascending address in [`FrameTxTrace`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameTxBalanceDiff {
    /// Changed account.
    pub address: Address,
    /// Balance at transaction start.
    pub before: U256,
    /// Balance as of the POST_TX frame.
    pub after: U256,
}

/// One net storage change, ordered by `(address, key)` in [`FrameTxTrace`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameTxStorageDiff {
    /// Changed account.
    pub address: Address,
    /// Changed storage key.
    pub key: StorageKey,
    /// Value at transaction start.
    pub before: StorageValue,
    /// Value as of the POST_TX frame.
    pub after: StorageValue,
}

/// One newly deployed contract, ordered by ascending address in [`FrameTxTrace`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameTxDeployedContract {
    /// Deployed contract address.
    pub address: Address,
    /// Current non-empty, non-delegation code hash.
    pub code_hash: B256,
}

/// Account-level nonce and code-hash diff used by direct lookups and flags.
///
/// Entries are ordered by ascending address. Balance and storage changes remain
/// in their dedicated ordered vectors so their TXTRACE global indices are
/// stable.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameTxAccountDiff {
    /// Changed account.
    pub address: Address,
    /// Whether the nonce differs from transaction pre-state. Nonce values are
    /// deliberately not exposed.
    pub nonce_changed: bool,
    /// Code hash at transaction start.
    pub code_hash_before: B256,
    /// Code hash as of the POST_TX frame.
    pub code_hash_after: B256,
}

/// One event in transaction emission order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameTxEvent {
    /// Contract that emitted the event.
    pub address: Address,
    /// Event topics in LOG order, with at most four entries.
    pub topics: Vec<B256>,
    /// Non-indexed event data.
    pub data: Bytes,
}

/// Precomputed EIP-7906 transaction trace for POST_TX introspection.
///
/// Hosts and tooling must provide `balance_diffs`, `account_diffs`, and
/// `deployed_contracts` in strictly ascending address order, and
/// `storage_diffs` in ascending `(address, key)` order. `events` remain in
/// emission order. Diff vectors contain net changes only.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrameTxTrace {
    /// Net balance changes.
    pub balance_diffs: Vec<FrameTxBalanceDiff>,
    /// Net storage changes.
    pub storage_diffs: Vec<FrameTxStorageDiff>,
    /// Contracts deployed by the transaction.
    pub deployed_contracts: Vec<FrameTxDeployedContract>,
    /// Account-level nonce and code-hash changes.
    pub account_diffs: Vec<FrameTxAccountDiff>,
    /// Events in global transaction log order. [`FrameTxContext::into_shared`]
    /// freezes this source into the snapshot used by frame instructions.
    pub events: Vec<FrameTxEvent>,
    /// Total gas pre-charge deducted from the payer.
    pub gas_pre_charge: U256,
    /// Account charged the gas pre-charge.
    pub gas_payer: Address,
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

    /// Returns whether the experimental EIP-7819 `SETDELEGATE` instruction is enabled.
    fn is_eip7819_enabled(&self) -> bool {
        false
    }

    /// Returns whether EIP-7851 is active for the host's current spec.
    fn is_eip7851_enabled(&self) -> bool {
        false
    }

    /* EIP-8141 frame transaction */

    /// Frame transaction context, or `None` when this is not a frame transaction.
    ///
    /// Defaults to `None` so that hosts which do not model frame transactions
    /// compile unchanged; the frame opcodes then halt exceptionally, which is
    /// what the spec requires outside a frame transaction.
    fn frame_context(&self) -> Option<Arc<FrameTxContext>> {
        None
    }

    /// Applies `APPROVE` for the given scope, returning whether it succeeded.
    ///
    /// Defaults to rejecting, so a host that has not opted in cannot silently
    /// approve payment or execution.
    fn frame_approve(&mut self, _scope: u64) -> bool {
        false
    }

    /// Applies EIP-7819 delegation code at `location`.
    ///
    /// Returns whether `location` existed before the write. `None` represents a
    /// host/database failure; hosts that do not implement EIP-7819 default to `None`.
    fn set_delegate(
        &mut self,
        _location: Address,
        _target: Address,
    ) -> Option<Result<bool, SetDelegateError>> {
        None
    }

    /// Replaces `authority`'s valid delegation with an ECDSA-disabled one.
    ///
    /// Returns `Some(true)` on mutation, `Some(false)` for a zero target or an
    /// invalid raw authority designation, and `None` on host/database failure.
    fn set_self_delegate(&mut self, _authority: Address, _target: Address) -> Option<bool> {
        None
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
        let is_eip7851_enabled = self.is_eip7851_enabled();
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

        let delegated_address = account.code.as_ref().and_then(|code| {
            if is_eip7851_enabled {
                code.delegated_address()
            } else {
                code.eip7702_address()
            }
        });
        if let Some(address) = delegated_address {
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
    spec_id: SpecId,
    /// Optional EIP-8141 frame transaction context, so tests and non-consensus
    /// hosts can exercise the frame instructions. `None` means "not a frame
    /// transaction", which makes those instructions halt.
    frame_tx: Option<Arc<FrameTxContext>>,
    /// Scopes that [`Host::frame_approve`] will accept, as a bitmask.
    pub approvable_scopes: u64,
    /// Number of calls made to [`Host::frame_approve`].
    pub frame_approve_calls: usize,
    /// Enables EIP-7819 for interpreter tests.
    pub enable_eip7819: bool,
    /// Enables EIP-7851 for interpreter tests.
    pub enable_eip7851: bool,
    /// Result returned by the EIP-7819 state mutation fixture.
    pub set_delegate_result: Option<Result<bool, SetDelegateError>>,
    /// Calls made to the EIP-7819 state mutation fixture.
    pub set_delegate_calls: Vec<(Address, Address)>,
    /// Result returned by the EIP-7851 state mutation fixture.
    pub set_self_delegate_result: Option<bool>,
    /// Calls made to the EIP-7851 state mutation fixture.
    pub set_self_delegate_calls: Vec<(Address, Address)>,
    /// Account returned by fixture live-state reads.
    pub account_info: AccountInfo,
    /// Whether the next fixture account read is cold.
    pub account_is_cold: bool,
    /// Whether the fixture account is empty.
    pub account_is_empty: bool,
    /// Value returned by fixture storage reads.
    pub storage_value: StorageValue,
    /// Whether the next fixture storage read is cold.
    pub storage_is_cold: bool,
}

impl DummyHost {
    /// Create a new dummy host with the given spec.
    pub fn new(spec: SpecId) -> Self {
        Self {
            gas_params: GasParams::new_spec(spec),
            spec_id: spec,
            ..Default::default()
        }
    }
}

impl DummyHost {
    /// Installs a frame transaction context and permits the given approval scopes.
    pub fn with_frame_tx(mut self, frame_tx: FrameTxContext, approvable_scopes: u64) -> Self {
        self.set_frame_tx_context(Some(frame_tx));
        self.approvable_scopes = approvable_scopes;
        self
    }

    /// Replaces the frame transaction context, rebuilding all derived lookups.
    pub fn set_frame_tx_context(&mut self, frame_tx: Option<FrameTxContext>) {
        self.frame_tx = frame_tx.map(FrameTxContext::into_shared);
    }

    /// Returns the installed frame context mutably when it is not shared.
    /// Prepared event data remains frozen even if the source trace is changed.
    pub fn frame_tx_context_mut(&mut self) -> Option<&mut FrameTxContext> {
        self.frame_tx.as_mut().and_then(Arc::get_mut)
    }
}

impl Host for DummyHost {
    fn frame_context(&self) -> Option<Arc<FrameTxContext>> {
        self.frame_tx.clone()
    }

    fn frame_approve(&mut self, scope: u64) -> bool {
        self.frame_approve_calls += 1;
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
        self.spec_id.is_enabled_in(SpecId::AMSTERDAM)
    }

    fn is_eip7819_enabled(&self) -> bool {
        self.enable_eip7819
    }

    fn is_eip7851_enabled(&self) -> bool {
        self.enable_eip7851 && self.spec_id.is_enabled_in(SpecId::PRAGUE)
    }

    fn set_delegate(
        &mut self,
        location: Address,
        target: Address,
    ) -> Option<Result<bool, SetDelegateError>> {
        self.set_delegate_calls.push((location, target));
        Some(self.set_delegate_result.unwrap_or(Ok(false)))
    }

    fn set_self_delegate(&mut self, authority: Address, target: Address) -> Option<bool> {
        self.set_self_delegate_calls.push((authority, target));
        self.set_self_delegate_result
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
        skip_cold_load: bool,
    ) -> Result<AccountInfoLoad<'_>, LoadError> {
        if self.account_is_cold && skip_cold_load {
            return Err(LoadError::ColdLoadSkipped);
        }
        let is_cold = core::mem::replace(&mut self.account_is_cold, false);
        Ok(AccountInfoLoad::new(
            &self.account_info,
            is_cold,
            self.account_is_empty,
        ))
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
        skip_cold_load: bool,
    ) -> Result<StateLoad<StorageValue>, LoadError> {
        if self.storage_is_cold && skip_cold_load {
            return Err(LoadError::ColdLoadSkipped);
        }
        let is_cold = core::mem::replace(&mut self.storage_is_cold, false);
        Ok(StateLoad::new(self.storage_value, is_cold))
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

    #[test]
    fn dummy_host_live_state_fixture_tracks_warmth() {
        let mut host = DummyHost::new(SpecId::BERLIN);
        host.account_info.balance = U256::from(7u64);
        host.account_is_cold = true;
        host.storage_value = U256::from(9u64);
        host.storage_is_cold = true;

        let first_account = host
            .load_account_info_skip_cold_load(Address::ZERO, false, false)
            .unwrap();
        assert!(first_account.is_cold);
        assert_eq!(first_account.balance, U256::from(7u64));
        drop(first_account);
        assert!(
            !host
                .load_account_info_skip_cold_load(Address::ZERO, false, false)
                .unwrap()
                .is_cold
        );

        let first_storage = host
            .sload_skip_cold_load(Address::ZERO, U256::ZERO, false)
            .unwrap();
        assert!(first_storage.is_cold);
        assert_eq!(first_storage.data, U256::from(9u64));
        assert!(
            !host
                .sload_skip_cold_load(Address::ZERO, U256::ZERO, false)
                .unwrap()
                .is_cold
        );
    }
}
