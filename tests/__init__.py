import os
import tempfile

# Tests must not pick up the observer from the user's ~/.config/astrophoto/config.toml
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="astrophoto-tests-")
