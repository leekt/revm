use crate::{
    bytecode::{
        opcode::{
            CALL, DELEGATECALL, GAS, POP, PUSH0, PUSH1, PUSH20, REVERT, SETSELFDELEGATE, SSTORE,
            STOP,
        },
        Bytecode,
    },
    context::{
        result::{EVMError, ExecutionResult, HaltReason},
        Context, TxEnv,
    },
    context_interface::{
        transaction::{Authorization, RecoveredAuthority, RecoveredAuthorization},
        JournalTr,
    },
    database::{CacheDB, EmptyDB},
    handler::pre_execution::{apply_auth_list, apply_auth_list_eip2780},
    interpreter::GasTracker,
    primitives::{address, bytes, hardfork::SpecId, Address, Bytes, HashSet, TxKind, U256},
    state::AccountInfo,
    ExecuteEvm, Journal, MainBuilder, MainContext,
};
use core::convert::Infallible;

#[cfg(feature = "optional_eip3607")]
use crate::context::result::InvalidTransaction;

const CALLER: Address = address!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
const AUTHORITY: Address = address!("1111111111111111111111111111111111111111");
const OLD_TARGET: Address = address!("2222222222222222222222222222222222222222");
const NEW_TARGET: Address = address!("3333333333333333333333333333333333333333");
const ROUTER: Address = address!("4444444444444444444444444444444444444444");
const LIBRARY: Address = address!("5555555555555555555555555555555555555555");
const ECRECOVER: Address = address!("0000000000000000000000000000000000000001");
const RECOVERED: Address = address!("7e5f4552091a69125d5dfcb7b8c2659029395bdf");

fn insert_code(db: &mut CacheDB<EmptyDB>, address: Address, nonce: u64, code: Bytecode) {
    db.insert_account_info(
        address,
        AccountInfo {
            nonce,
            code_hash: code.hash_slow(),
            code: Some(code),
            ..Default::default()
        },
    );
}

fn call_tx(target: Address) -> TxEnv {
    TxEnv::builder()
        .caller(CALLER)
        .kind(TxKind::Call(target))
        .gas_limit(500_000)
        .gas_price(0)
        .build()
        .unwrap()
}

fn ecrecover_input() -> Bytes {
    bytes!(
        "1111111111111111111111111111111111111111111111111111111111111111\
            000000000000000000000000000000000000000000000000000000000000001c\
            e7c93726a865578504442b1a6827f676e0ed74bdff2be3960d1e253bbcfc4462\
            6aa772b878bc912bdbb33a0014ec507c4b3896ea85aa914b74dee9b7ac3e56da"
    )
}

fn recovered_output() -> Bytes {
    let mut output = [0; 32];
    output[12..].copy_from_slice(RECOVERED.as_slice());
    Bytes::copy_from_slice(&output)
}

fn storage_value(state: &crate::state::EvmState, address: Address, key: U256) -> Option<U256> {
    state
        .get(&address)
        .and_then(|account| account.storage.get(&key))
        .map(|slot| slot.present_value)
}

#[test]
fn eip7851_top_level_resolution_is_version_aware_and_gated() {
    let cases = [
        (
            "EIP-7702 strict default",
            Bytecode::new_eip7702(OLD_TARGET),
            SpecId::PRAGUE,
            false,
            true,
        ),
        (
            "EIP-7702 with EIP-7851",
            Bytecode::new_eip7702(OLD_TARGET),
            SpecId::PRAGUE,
            true,
            true,
        ),
        (
            "EIP-7851 enabled",
            Bytecode::new_eip7851(OLD_TARGET),
            SpecId::PRAGUE,
            true,
            true,
        ),
        (
            "EIP-7851 default off",
            Bytecode::new_eip7851(OLD_TARGET),
            SpecId::PRAGUE,
            false,
            false,
        ),
        (
            "EIP-7851 before Prague",
            Bytecode::new_eip7851(OLD_TARGET),
            SpecId::CANCUN,
            true,
            false,
        ),
        (
            "EIP-7702 before Prague",
            Bytecode::new_eip7702(OLD_TARGET),
            SpecId::CANCUN,
            true,
            false,
        ),
    ];

    for (case, designation, spec, enabled, should_resolve) in cases {
        let implementation = Bytecode::new_legacy([PUSH1, 0x2a, PUSH0, SSTORE, STOP].into());
        let mut db = CacheDB::<EmptyDB>::default();
        insert_code(&mut db, AUTHORITY, 7, designation);
        insert_code(&mut db, OLD_TARGET, 1, implementation);
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| {
                cfg.set_spec_and_mainnet_gas_params(spec);
                cfg.enable_eip7851 = enabled;
            })
            .with_db(db);
        let mut evm = ctx.build_mainnet();

        let output = evm
            .transact(call_tx(AUTHORITY))
            .unwrap_or_else(|err| panic!("{case}: {err:?}"));

        if should_resolve {
            assert!(
                matches!(output.result, ExecutionResult::Success { .. }),
                "{case}"
            );
            assert_eq!(
                storage_value(&output.state, AUTHORITY, U256::ZERO),
                Some(U256::from(0x2a))
            );
        } else {
            assert!(
                matches!(
                    output.result,
                    ExecutionResult::Halt {
                        reason: HaltReason::OpcodeNotFound,
                        ..
                    }
                ),
                "{case}"
            );
            assert_eq!(storage_value(&output.state, AUTHORITY, U256::ZERO), None);
        }
    }
}

#[test]
fn eip7851_nested_call_resolves_both_delegation_versions() {
    let mut router_code = vec![PUSH0, PUSH0, PUSH0, PUSH0, PUSH0, PUSH20];
    router_code.extend_from_slice(AUTHORITY.as_slice());
    router_code.extend_from_slice(&[GAS, CALL, POP, STOP]);
    let router_code = Bytecode::new_legacy(router_code.into());

    for designation in [
        Bytecode::new_eip7702(OLD_TARGET),
        Bytecode::new_eip7851(OLD_TARGET),
    ] {
        let implementation = Bytecode::new_legacy([PUSH1, 0x2a, PUSH0, SSTORE, STOP].into());
        let mut db = CacheDB::<EmptyDB>::default();
        insert_code(&mut db, ROUTER, 1, router_code.clone());
        insert_code(&mut db, AUTHORITY, 7, designation);
        insert_code(&mut db, OLD_TARGET, 1, implementation);
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| {
                cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE);
                cfg.enable_eip7851 = true;
            })
            .with_db(db);
        let mut evm = ctx.build_mainnet();

        let output = evm.transact(call_tx(ROUTER)).unwrap();

        assert!(matches!(output.result, ExecutionResult::Success { .. }));
        assert_eq!(
            storage_value(&output.state, AUTHORITY, U256::ZERO),
            Some(U256::from(0x2a))
        );
    }
}

#[test]
fn eip7851_setselfdelegate_keeps_loaded_code_and_reentry_uses_new_target() {
    let new_code = Bytecode::new_legacy([PUSH1, 0x22, PUSH1, 0x01, SSTORE, STOP].into());
    let mut old_code = vec![PUSH20];
    old_code.extend_from_slice(NEW_TARGET.as_slice());
    old_code.extend_from_slice(&[
        SETSELFDELEGATE,
        POP,
        PUSH1,
        0x11,
        PUSH0,
        SSTORE,
        PUSH0,
        PUSH0,
        PUSH0,
        PUSH0,
        PUSH0,
        PUSH20,
    ]);
    old_code.extend_from_slice(AUTHORITY.as_slice());
    old_code.extend_from_slice(&[GAS, CALL, POP, STOP]);
    let old_code = Bytecode::new_legacy(old_code.into());
    let designation = Bytecode::new_eip7702(OLD_TARGET);
    let mut db = CacheDB::<EmptyDB>::default();
    insert_code(&mut db, AUTHORITY, 7, designation);
    insert_code(&mut db, OLD_TARGET, 1, old_code);
    insert_code(&mut db, NEW_TARGET, 1, new_code);
    let ctx = Context::mainnet()
        .modify_cfg_chained(|cfg| {
            cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE);
            cfg.enable_eip7851 = true;
        })
        .with_db(db);
    let mut evm = ctx.build_mainnet();

    let output = evm.transact(call_tx(AUTHORITY)).unwrap();

    assert!(matches!(output.result, ExecutionResult::Success { .. }));
    let authority = &output.state[&AUTHORITY];
    assert_eq!(authority.info.nonce, 7);
    assert_eq!(
        authority.info.code.as_ref().unwrap().eip7851_address(),
        Some(NEW_TARGET)
    );
    assert_eq!(
        storage_value(&output.state, AUTHORITY, U256::ZERO),
        Some(U256::from(0x11))
    );
    assert_eq!(
        storage_value(&output.state, AUTHORITY, U256::from(1)),
        Some(U256::from(0x22))
    );
    assert_eq!(storage_value(&output.state, OLD_TARGET, U256::ZERO), None);
    assert_eq!(
        storage_value(&output.state, NEW_TARGET, U256::from(1)),
        None
    );
}

#[test]
fn eip7851_setselfdelegate_uses_delegatecall_execution_context() {
    let mut entry_code = vec![PUSH0, PUSH0, PUSH0, PUSH0, PUSH20];
    entry_code.extend_from_slice(LIBRARY.as_slice());
    entry_code.extend_from_slice(&[GAS, DELEGATECALL, POP, STOP]);
    let entry_code = Bytecode::new_legacy(entry_code.into());
    let mut library_code = vec![PUSH20];
    library_code.extend_from_slice(NEW_TARGET.as_slice());
    library_code.extend_from_slice(&[SETSELFDELEGATE, POP, STOP]);
    let library_code = Bytecode::new_legacy(library_code.into());
    let designation = Bytecode::new_eip7851(OLD_TARGET);
    let mut db = CacheDB::<EmptyDB>::default();
    insert_code(&mut db, AUTHORITY, 7, designation);
    insert_code(&mut db, OLD_TARGET, 1, entry_code);
    insert_code(&mut db, LIBRARY, 1, library_code.clone());
    insert_code(&mut db, NEW_TARGET, 1, Bytecode::new());
    let ctx = Context::mainnet()
        .modify_cfg_chained(|cfg| {
            cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE);
            cfg.enable_eip7851 = true;
        })
        .with_db(db);
    let mut evm = ctx.build_mainnet();

    let output = evm.transact(call_tx(AUTHORITY)).unwrap();

    assert!(matches!(output.result, ExecutionResult::Success { .. }));
    assert_eq!(
        output.state[&AUTHORITY]
            .info
            .code
            .as_ref()
            .unwrap()
            .eip7851_address(),
        Some(NEW_TARGET)
    );
    assert_eq!(output.state[&AUTHORITY].info.nonce, 7);
    assert_eq!(
        output.state[&LIBRARY].info.code.as_ref(),
        Some(&library_code)
    );
}

#[test]
fn eip7851_setselfdelegate_reverts_with_the_frame() {
    let mut implementation = vec![PUSH20];
    implementation.extend_from_slice(NEW_TARGET.as_slice());
    implementation.extend_from_slice(&[SETSELFDELEGATE, POP, PUSH0, PUSH0, REVERT]);
    let implementation = Bytecode::new_legacy(implementation.into());
    let original = Bytecode::new_eip7702(OLD_TARGET);
    let mut db = CacheDB::<EmptyDB>::default();
    insert_code(&mut db, AUTHORITY, 7, original.clone());
    insert_code(&mut db, OLD_TARGET, 1, implementation);
    let ctx = Context::mainnet()
        .modify_cfg_chained(|cfg| {
            cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE);
            cfg.enable_eip7851 = true;
        })
        .with_db(db);
    let mut evm = ctx.build_mainnet();

    let output = evm.transact(call_tx(AUTHORITY)).unwrap();

    assert!(matches!(output.result, ExecutionResult::Revert { .. }));
    assert_eq!(output.state[&AUTHORITY].info.nonce, 7);
    assert_eq!(output.state[&AUTHORITY].info.code.as_ref(), Some(&original));
}

/// Pinned to ethereum/EIPs@bf7a4067f263bf7ce01c1511de48473e281d885d:
/// EIP-8151 permits only version-zero delegation indicators even when EIP-7851 is enabled.
#[test]
fn eip7851_eip8151_accepts_ef0100_then_rejects_ef0101() {
    for (case, code, allowed) in [
        ("ef0100", Bytecode::new_eip7702(OLD_TARGET), true),
        ("ef0101", Bytecode::new_eip7851(OLD_TARGET), false),
    ] {
        let mut db = CacheDB::<EmptyDB>::default();
        insert_code(&mut db, RECOVERED, 1, code);
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| {
                cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE);
                cfg.enable_eip7851 = true;
                cfg.enable_eip8151 = true;
            })
            .with_db(db);
        let mut evm = ctx.build_mainnet();
        let tx = TxEnv::builder()
            .caller(CALLER)
            .kind(TxKind::Call(ECRECOVER))
            .gas_limit(100_000)
            .gas_price(0)
            .data(ecrecover_input())
            .build()
            .unwrap();

        let output = evm
            .transact(tx)
            .unwrap_or_else(|err| panic!("{case}: {err:?}"));

        assert!(
            matches!(output.result, ExecutionResult::Success { .. }),
            "{case}"
        );
        let expected = if allowed {
            recovered_output()
        } else {
            Bytes::from_static(&[0; 32])
        };
        assert_eq!(output.result.output(), Some(&expected), "{case}");
    }
}

fn recovered_authorization(delegate: Address, nonce: u64) -> RecoveredAuthorization {
    RecoveredAuthorization::new_unchecked(
        Authorization {
            chain_id: U256::ZERO,
            address: delegate,
            nonce,
        },
        RecoveredAuthority::Valid(AUTHORITY),
    )
}

fn disabled_authority_journal() -> (Journal<CacheDB<EmptyDB>>, Bytecode) {
    let disabled = Bytecode::new_eip7851(OLD_TARGET);
    let mut db = CacheDB::<EmptyDB>::default();
    insert_code(&mut db, AUTHORITY, 7, disabled.clone());
    let mut journal = Journal::new(db);
    journal.set_spec_id(SpecId::PRAGUE);
    let _ = journal.load_account_with_code(AUTHORITY).unwrap();
    (journal, disabled)
}

#[test]
fn eip7851_standard_authorization_skips_disabled_authority_without_mutation() {
    let (mut journal, disabled) = disabled_authority_journal();
    let authorization = recovered_authorization(NEW_TARGET, 7);
    let journal_entries = journal.inner.journal.len();

    let refund = apply_auth_list::<_, EVMError<Infallible>>(
        1,
        core::iter::once(&authorization),
        &mut journal,
    )
    .unwrap();

    assert_eq!(refund, 0);
    assert_eq!(journal.inner.journal.len(), journal_entries);
    let account = journal.load_account_with_code(AUTHORITY).unwrap();
    assert_eq!(account.info.nonce, 7);
    assert_eq!(account.info.code.as_ref(), Some(&disabled));
}

#[test]
fn eip7851_eip2780_authorization_skips_disabled_authority_without_mutation() {
    let (mut journal, disabled) = disabled_authority_journal();
    let authorization = recovered_authorization(NEW_TARGET, 7);
    let mut written_accounts = HashSet::default();
    let mut gas = GasTracker::new(1_000_000, 1_000_000, 0);
    let original_gas = gas;
    let journal_entries = journal.inner.journal.len();

    let out_of_gas = apply_auth_list_eip2780::<_, EVMError<Infallible>>(
        1,
        core::iter::once(&authorization),
        &mut journal,
        101,
        202,
        303,
        &mut written_accounts,
        &mut gas,
    )
    .unwrap();

    assert!(!out_of_gas);
    assert_eq!(gas, original_gas);
    assert!(written_accounts.is_empty());
    assert_eq!(journal.inner.journal.len(), journal_entries);
    let account = journal.load_account_with_code(AUTHORITY).unwrap();
    assert_eq!(account.info.nonce, 7);
    assert_eq!(account.info.code.as_ref(), Some(&disabled));
}

#[cfg(feature = "optional_eip3607")]
#[test]
fn eip7851_sender_rejection_applies_only_to_protocol_ecdsa_transactions() {
    let authenticated = TxEnv::builder()
        .caller(AUTHORITY)
        .kind(TxKind::Call(ROUTER))
        .gas_limit(100_000)
        .gas_price(0)
        .build()
        .unwrap();
    let mut unauthenticated = authenticated.clone();
    unauthenticated.set_eip7851_sender_ecdsa_authenticated(false);

    for (tx, should_reject) in [(authenticated, true), (unauthenticated, false)] {
        let disabled = Bytecode::new_eip7851(OLD_TARGET);
        let mut db = CacheDB::<EmptyDB>::default();
        insert_code(&mut db, AUTHORITY, 0, disabled);
        let ctx = Context::mainnet()
            .modify_cfg_chained(|cfg| {
                cfg.set_spec_and_mainnet_gas_params(SpecId::PRAGUE);
                cfg.enable_eip7851 = true;
                cfg.disable_eip3607 = true;
            })
            .with_db(db);
        let mut evm = ctx.build_mainnet();

        let result = evm.transact(tx);

        if should_reject {
            assert!(matches!(
                result,
                Err(EVMError::Transaction(
                    InvalidTransaction::RejectCallerWithCode
                ))
            ));
        } else {
            assert!(matches!(
                result.unwrap().result,
                ExecutionResult::Success { .. }
            ));
        }
    }
}

#[cfg(feature = "serde")]
#[test]
fn eip7851_sender_authentication_serde_roundtrips_and_defaults_true() {
    let tx = TxEnv::builder()
        .eip7851_sender_ecdsa_authenticated(false)
        .build()
        .unwrap();
    let mut json = serde_json::to_value(&tx).unwrap();
    assert_eq!(
        json["eip7851_sender_ecdsa_authenticated"],
        serde_json::Value::Bool(false)
    );

    let decoded: TxEnv = serde_json::from_value(json.clone()).unwrap();
    assert!(!decoded.eip7851_sender_ecdsa_authenticated);

    json.as_object_mut()
        .unwrap()
        .remove("eip7851_sender_ecdsa_authenticated");
    let decoded_legacy: TxEnv = serde_json::from_value(json).unwrap();
    assert!(decoded_legacy.eip7851_sender_ecdsa_authenticated);
}
