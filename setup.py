#!/usr/bin/env python

import re

from setuptools import setup


def custom_local_scheme(version):
    """
    Returns the local part of the version string.
    If on a branch other than 'main' or 'master', it includes the branch name.
    """
    if version.exact:
        return ""
    # Clean branch name for PEP 440 compatibility
    branch_name = re.sub(r"[^a-zA-Z0-9]", ".", version.branch)
    # Include branch name and node (commit hash)
    local_version = f"+{branch_name}.{version.short_node}"
    if version.dirty:
        local_version += ".dirty"
    return local_version


# The metadata is all in pyproject.toml.
# This step is just to support editable installs with setuptools < 64.
setup(use_scm_version={"local_scheme": custom_local_scheme})
