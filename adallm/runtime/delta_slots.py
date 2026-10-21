"""Free-list allocator for delta-net (mamba-style) state pool slots.

Slot 0 is reserved as the always-zero NULL slot (kernels skip state_idx <= 0);
real sequences get slots >= 1. GPU-side state lives in each delta layer's
`mamba_state`/`conv_state` pools (see qwen3_5.allocate_state_pool); this class
only tracks CPU-side ownership.
"""


class DeltaSlotManager:

    def __init__(self, num_slots: int):
        assert num_slots >= 2
        self.num_slots = num_slots
        self._free = list(range(1, num_slots))
        self._used = set()

    def alloc(self) -> int:
        if not self._free:
            raise RuntimeError("delta state pool exhausted (too many concurrent seqs)")
        slot = self._free.pop(0)
        self._used.add(slot)
        return slot

    def free(self, slot: int):
        if slot in self._used:
            self._used.remove(slot)
            self._free.append(slot)

    @property
    def num_used(self) -> int:
        return len(self._used)
