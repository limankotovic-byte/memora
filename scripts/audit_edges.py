#!/usr/bin/env python3
"""CLI wrapper for the memora edge auditor.

The implementation lives in :mod:`memora.audit` so the tool is also
available as the ``memora-audit-edges`` entry point after install.
"""

import sys

from memora.audit import main

if __name__ == "__main__":
    sys.exit(main())
