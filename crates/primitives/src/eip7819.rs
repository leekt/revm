//! EIP-7819: SETDELEGATE instruction.

use crate::{keccak256, Address, U256};

/// Prefix used both in the deterministic-address preimage and EIP-7702 delegation code.
pub const DELEGATION_PREFIX: [u8; 3] = [0xef, 0x01, 0x00];

/// Gross gas charged by `SETDELEGATE`.
pub const EMPTY_ACCOUNT_COST: u64 = 25_000;

/// Effective cost when the destination account already exists.
pub const BASE_COST: u64 = 12_500;

/// Refund recorded when the destination account already exists.
pub const EXISTING_ACCOUNT_REFUND: i64 = (EMPTY_ACCOUNT_COST - BASE_COST) as i64;

/// Derives the deterministic delegation location for the current execution address and salt.
#[inline]
pub fn setdelegate_address(address: Address, salt: U256) -> Address {
    let mut preimage = [0u8; 55];
    preimage[..3].copy_from_slice(&DELEGATION_PREFIX);
    preimage[3..23].copy_from_slice(address.as_slice());
    preimage[23..].copy_from_slice(&salt.to_be_bytes::<32>());
    Address::from_slice(&keccak256(preimage)[12..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{address, uint};

    #[test]
    fn derives_reference_location() {
        assert_eq!(
            setdelegate_address(
                address!("1111111111111111111111111111111111111111"),
                U256::ZERO
            ),
            address!("7a41c03bf3062738d4ad052749101d8ec0f5639d")
        );
    }

    #[test]
    fn salt_uses_all_256_bits() {
        let address = Address::repeat_byte(0x11);
        assert_ne!(
            setdelegate_address(address, U256::ZERO),
            setdelegate_address(
                address,
                uint!(0x1000000000000000000000000000000000000000000000000000000000000000_U256)
            )
        );
    }
}
