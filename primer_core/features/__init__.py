"""Pure, deterministic feature functions for primer / template sequences.

Every function in this subpackage is side-effect free (no I/O, no network, no
filesystem access) so it can be unit-tested in isolation with hand-checked
expected values, by project convention. Functions that require external I/O (BLAST
databases, ViennaRNA, primer3 bindings) still live here because they are
"features" conceptually, but they are documented as such and covered by
stub-level tests only until wired up.
"""
