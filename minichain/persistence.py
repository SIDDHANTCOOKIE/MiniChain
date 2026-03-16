"""
Chain persistence: save and load the blockchain and state to/from JSON.

Design:
  - blockchain.json  holds the full list of serialised blocks
  - state.json       holds the accounts dict (includes off-chain credits)

Both files are written atomically (temp → rename) to prevent corruption
on crash.  On load, chain integrity is verified before the data is trusted.

Usage:
    from minichain.persistence import save, load

    save(blockchain, path="data/")
    blockchain = load(path="data/")
"""

import json
import os
import tempfile
import logging

from .block import Block
from .transaction import Transaction
from .chain import Blockchain
from .state import State
from .pow import calculate_hash

logger = logging.getLogger(__name__)

_CHAIN_FILE = "blockchain.json"
_STATE_FILE = "state.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save(blockchain: Blockchain, path: str = ".") -> None:
    """
    Persist the blockchain and account state to JSON files inside *path*.

    Uses atomic write (write-to-temp → rename) so a crash mid-save
    never corrupts the existing file.
    """
    os.makedirs(path, exist_ok=True)

    with blockchain._lock:  # Thread-safe: hold lock while serialising
        chain_data = [block.to_dict() for block in blockchain.chain]
        state_data = blockchain.state.accounts.copy()

    _atomic_write_json(os.path.join(path, _CHAIN_FILE), chain_data)
    _atomic_write_json(os.path.join(path, _STATE_FILE), state_data)

    logger.info(
        "Saved %d blocks and %d accounts to '%s'",
        len(chain_data),
        len(state_data),
        path,
    )


def load(path: str = ".") -> Blockchain:
    """
    Restore a Blockchain from JSON files inside *path*.

    Steps:
      1. Load and deserialise blocks from blockchain.json
      2. Verify chain integrity (genesis, linkage, hashes)
      3. Load account state from state.json

    Raises:
        FileNotFoundError: if blockchain.json or state.json is missing.
        ValueError:        if data is invalid or integrity checks fail.
    """
    chain_path = os.path.join(path, _CHAIN_FILE)
    state_path = os.path.join(path, _STATE_FILE)

    raw_blocks = _read_json(chain_path)
    raw_accounts = _read_json(state_path)

    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError(f"Invalid or empty chain data in '{chain_path}'")
    if not isinstance(raw_accounts, dict):
        raise ValueError(f"Invalid accounts data in '{state_path}'")

    blocks = [_deserialize_block(b) for b in raw_blocks]

    # --- Integrity verification ---
    _verify_chain_integrity(blocks)

    # --- Rebuild blockchain properly (no __new__ hack) ---
    blockchain = Blockchain()           # creates genesis + fresh state
    blockchain.chain = blocks           # replace with loaded chain

    # Restore state
    blockchain.state.accounts = raw_accounts

    logger.info(
        "Loaded %d blocks and %d accounts from '%s'",
        len(blockchain.chain),
        len(blockchain.state.accounts),
        path,
    )
    return blockchain


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------

def _verify_chain_integrity(blocks: list) -> None:
    """Verify genesis, hash linkage, and block hashes."""
    # Check genesis
    genesis = blocks[0]
    if genesis.index != 0 or genesis.hash != "0" * 64:
        raise ValueError("Invalid genesis block")

    # Check linkage and hashes for every subsequent block
    for i in range(1, len(blocks)):
        block = blocks[i]
        prev = blocks[i - 1]

        if block.index != prev.index + 1:
            raise ValueError(
                f"Block #{block.index}: index gap (expected {prev.index + 1})"
            )

        if block.previous_hash != prev.hash:
            raise ValueError(
                f"Block #{block.index}: previous_hash mismatch"
            )

        expected_hash = calculate_hash(block.to_header_dict())
        if block.hash != expected_hash:
            raise ValueError(
                f"Block #{block.index}: hash mismatch "
                f"(stored={block.hash[:16]}..., computed={expected_hash[:16]}...)"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(filepath: str, data) -> None:
    """Write JSON atomically: temp file -> os.replace (crash-safe)."""
    dir_name = os.path.dirname(filepath) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, filepath)   # atomic on all platforms
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _read_json(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Persistence file not found: '{filepath}'")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _deserialize_block(data: dict) -> Block:
    """Reconstruct a Block (including its transactions) from a plain dict."""
    transactions = [
        Transaction(
            sender=tx["sender"],
            receiver=tx["receiver"],
            amount=tx["amount"],
            nonce=tx["nonce"],
            data=tx.get("data"),
            signature=tx.get("signature"),
            timestamp=tx["timestamp"],
        )
        for tx in data.get("transactions", [])
    ]

    block = Block(
        index=data["index"],
        previous_hash=data["previous_hash"],
        transactions=transactions,
        timestamp=data["timestamp"],
        difficulty=data.get("difficulty"),
    )
    block.nonce = data["nonce"]
    block.hash = data["hash"]
    block.merkle_root = data.get("merkle_root")
    return block
