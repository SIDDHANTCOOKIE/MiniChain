import logging
import threading

logger = logging.getLogger(__name__)

class Mempool:
    def __init__(self, max_size=1000, transactions_per_block=100):
        self._pool = {}
        self._size = 0
        self._lock = threading.Lock()
        self.max_size = max_size
        self.transactions_per_block = transactions_per_block

    def add_transaction(self, tx):
        if not tx.verify():
            logger.warning("Mempool: Invalid signature rejected")
            return False

        with self._lock:
            existing = self._pool.get(tx.sender, {}).get(tx.nonce)

            if existing and existing.tx_id == tx.tx_id:
                logger.warning("Mempool: Duplicate transaction rejected %s", tx.tx_id)
                return False
            if not existing and self._size >= self.max_size:
                logger.warning("Mempool: Full, rejecting transaction")
                return False

            self._size += 0 if existing else 1
            pool = self._pool.setdefault(tx.sender, {})
            pool[tx.nonce] = tx
            return True

    def get_transactions_for_block(self):
        with self._lock:
            txs = [t for pool in self._pool.values() for t in pool.values()]
            
        txs.sort(key=lambda t: (t.timestamp, t.sender, t.nonce))
        return txs[: self.transactions_per_block]

    def remove_transactions(self, transactions):
        with self._lock:
            for tx in transactions:
                pool = self._pool.get(tx.sender)
                if pool and tx.nonce in pool:
                    del pool[tx.nonce]
                    self._size -= 1
                    if not pool:
                        del self._pool[tx.sender]

    def __len__(self):
        with self._lock:
            return self._size
