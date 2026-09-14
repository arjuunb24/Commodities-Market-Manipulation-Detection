import pytest
import sys
with open('pytest_out.txt', 'w') as f:
    sys.stdout = f
    sys.stderr = f
    pytest.main(['tests/', '-v'])
