"""I/O layer: sequence loaders, dataset loaders, and schema validation.

Nothing in ``primer_core/features/`` should import from here — features stay
pure and I/O-free. This subpackage is where filesystem/network access is
allowed.
"""
