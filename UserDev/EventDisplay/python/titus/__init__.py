"""
TITUS event display.
"""

import importlib


# TITUS's GUI, modules and gallery interface need ROOT, so these modules are
# imported lazily.
_LAZY_SUBPACKAGES = ('gui', 'modules', 'gallery_interface')


def __getattr__(name):
    # tools probe for dunder/private names; don't import ROOT for those
    if name.startswith('_'):
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    for sub in _LAZY_SUBPACKAGES:
        module = importlib.import_module(f'{__name__}.{sub}')
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
