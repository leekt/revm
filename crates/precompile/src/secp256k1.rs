//! `ecrecover` precompile.
//!
//! Depending on enabled features, it will use different implementations of `ecrecover`.
//! * [`k256`](https://crates.io/crates/k256) - uses maintained pure rust lib `k256`, it is perfect use for no_std environments.
//! * [`secp256k1`](https://crates.io/crates/secp256k1) - uses `bitcoin_secp256k1` lib, it is a C implementation of secp256k1 used in bitcoin core.
//!   It is faster than k256 and enabled by default and in std environment.

//!   Order of preference is `secp256k1` -> `k256`. Where if no features are enabled, it will use `k256`.
//!
//! Input format:
//! [32 bytes for message][64 bytes for signature][1 byte for recovery id]
//!
//! Output format:
//! [32 bytes for recovered address]
#[cfg(feature = "secp256k1")]
pub mod bitcoin_secp256k1;
pub mod k256;

use crate::{
    crypto, eth_precompile_fn, utilities::right_pad, EthPrecompileOutput, EthPrecompileResult,
    Precompile, PrecompileHalt, PrecompileId,
};
use primitives::{alloy_primitives::B512, Bytes, B256};

eth_precompile_fn!(ecrecover_precompile, ec_recover_run);

/// `ecrecover` precompile, containing address and function to run.
pub const ECRECOVER: Precompile = Precompile::new(
    PrecompileId::EcRec,
    crate::u64_to_address(1),
    ecrecover_precompile,
);

/// Returns whether raw account code may retain `ecrecover` authority under EIP-8151.
///
/// The only permitted non-empty form is an exact EIP-7702 delegation indicator,
/// `0xef0100 || address`. The delegate address itself is intentionally unrestricted.
#[inline]
pub fn is_ecrecover_code_eligible(code: &[u8]) -> bool {
    code.is_empty() || (code.len() == 23 && code.starts_with(&[0xef, 0x01, 0x00]))
}

/// `ecrecover` precompile function. Read more about input and output format in [this module docs](self).
pub fn ec_recover_run(input: &[u8], gas_limit: u64) -> EthPrecompileResult {
    const ECRECOVER_BASE: u64 = 3_000;

    if ECRECOVER_BASE > gas_limit {
        return Err(PrecompileHalt::OutOfGas);
    }

    let input = right_pad::<128>(input);

    // `v` must be a 32-byte big-endian integer equal to 27 or 28.
    if !(input[32..63].iter().all(|&b| b == 0) && matches!(input[63], 27 | 28)) {
        return Ok(EthPrecompileOutput::new(ECRECOVER_BASE, Bytes::new()));
    }

    let msg = <&B256>::try_from(&input[0..32]).unwrap();
    let recid = input[63] - 27;
    let sig = <&B512>::try_from(&input[64..128]).unwrap();

    let res = crypto().secp256k1_ecrecover(&sig.0, recid, &msg.0).ok();
    let out = res.map(|o| o.to_vec().into()).unwrap_or_default();
    Ok(EthPrecompileOutput::new(ECRECOVER_BASE, out))
}

pub(crate) fn ecrecover_bytes(sig: &[u8; 64], recid: u8, msg: &[u8; 32]) -> Option<[u8; 32]> {
    match ecrecover(sig.into(), recid, msg.into()) {
        Ok(address) => Some(address.0),
        Err(_) => None,
    }
}

// Select the correct implementation based on the enabled features.
cfg_if::cfg_if! {
    if #[cfg(feature = "secp256k1")] {
        pub use bitcoin_secp256k1::ecrecover;
    } else {
        pub use k256::ecrecover;
    }
}

#[cfg(test)]
mod tests {
    use super::{ec_recover_run, is_ecrecover_code_eligible};

    /// Pinned to ethereum/EIPs@bf7a4067f263bf7ce01c1511de48473e281d885d.
    #[test]
    fn ecrecover_code_restriction() {
        let mut invalid_input = [0; 128];
        invalid_input[63] = 27;
        assert!(ec_recover_run(&invalid_input, 3_000)
            .unwrap()
            .bytes
            .is_empty());

        let mut eip7702_zero_delegate = vec![0xef, 0x01, 0x00];
        eip7702_zero_delegate.extend_from_slice(&[0; 20]);
        let mut eip7702_nonzero_delegate = vec![0xef, 0x01, 0x00];
        eip7702_nonzero_delegate.extend_from_slice(&[0x42; 20]);

        assert!(is_ecrecover_code_eligible(&[]));
        assert!(is_ecrecover_code_eligible(&eip7702_zero_delegate));
        assert!(is_ecrecover_code_eligible(&eip7702_nonzero_delegate));

        let mut eip7851 = eip7702_nonzero_delegate.clone();
        eip7851[2] = 0x01;
        let mut trailing = eip7702_nonzero_delegate.clone();
        trailing.push(0);
        let mut wrong_version = vec![0; 23];
        wrong_version[..3].copy_from_slice(&[0xef, 0x01, 0x02]);

        for code in [
            vec![0x00],
            vec![0xef, 0x01, 0x00],
            eip7702_nonzero_delegate[..22].to_vec(),
            trailing,
            eip7851,
            wrong_version,
        ] {
            assert!(!is_ecrecover_code_eligible(&code), "accepted {code:02x?}");
        }
    }
}
