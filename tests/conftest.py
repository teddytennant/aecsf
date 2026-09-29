"""Load host C++/zlib before test modules import jax."""

from aecsf._host import preload

preload()

import jax

jax.config.update("jax_enable_x64", True)
