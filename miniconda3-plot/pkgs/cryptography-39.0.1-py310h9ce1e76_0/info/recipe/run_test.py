import subprocess
import time
import cryptography
import cryptography.fernet
import cryptography.hazmat
import cryptography.hazmat.backends
import cryptography.hazmat.backends.openssl
import cryptography.hazmat.bindings
import cryptography.hazmat.bindings.openssl
import cryptography.hazmat.primitives
import cryptography.hazmat.primitives.asymmetric
import cryptography.hazmat.primitives.ciphers
import cryptography.hazmat.primitives.kdf
import cryptography.hazmat.primitives.twofactor
import cryptography.x509
from cryptography.hazmat.backends.openssl import backend

# the version that cryptography uses
linked_version = backend.openssl_version_text()
# the version present in the conda environment
env_version = subprocess.check_output('openssl version', shell=True).decode('utf8').strip()

print('Version used by cryptography:\n{linked_version}'.format(linked_version=linked_version))
print('Version in conda environment:\n{env_version}'.format(env_version=env_version))

# avoid race condition between print and (possible) AssertionError
time.sleep(1)

# linking problems have appeared on windows before (see issue #38),
# and were only caught by lucky accident through the test suite.
# This is intended to ensure it does not happen again.
assert linked_version == env_version
