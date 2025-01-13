# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Configuration file for nox."""

from frequenz.repo.config import RepositoryType, nox

nox.configure(RepositoryType.LIB)
