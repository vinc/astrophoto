import os
import tempfile

# Tests must not depend on the user's config.toml or downloaded data
for variable in ("XDG_CONFIG_HOME", "XDG_DATA_HOME"):
    os.environ[variable] = tempfile.mkdtemp(prefix="astrophoto-tests-")
