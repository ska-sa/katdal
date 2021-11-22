################################################################################
# Copyright (c) 2017-2021, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Test utilities for code that interacts with the S3 API.

It provides a class for managing running an external S3 server (currently
`MinIO`_).

Versions of minio prior to 2018-08-25T01:56:38Z contain a `race condition`_
that can cause it to crash when queried at the wrong point during startup, so
should not be used.

.. _minio: https://github.com/minio/minio
.. _race condition: https://github.com/minio/minio/issues/6324
"""

import contextlib
import os
import pathlib
import subprocess
import time
import urllib.parse

import requests


class MissingProgram(RuntimeError):
    """An required executable program was not found."""


class ProgramFailed(RuntimeError):
    """An external program did not run successfully."""


class S3User:
    """Credentials for an S3 user."""

    def __init__(self, access_key: str, secret_key: str) -> None:
        self.access_key = access_key
        self.secret_key = secret_key


class S3Server:
    """Run and manage an external program to run an S3 server.

    This can be used as a context manager, to shut down the server when
    finished.

    Parameters
    ----------
    host
        Host to bind to
    port
        Port to bind to
    path
        Directory in which objects and config will be stored.
    user
        Credentials for the default admin user.

    Attributes
    ----------
    host
        Hostname for connecting to the server
    port
        Port for connecting to the server
    url
        Base URL for the server
    auth_url
        URL with the access_key and secret_key baked in
    path
        Path given to the constructor
    user
        User given to the constructor

    Raises
    ------
    MissingProgram
        if the ``minio`` binary was not found.
    ProgramFailed
        if minio started but failed before it became healthy
    """

    def __init__(self, host: str, port: int, path: pathlib.Path, user: S3User) -> None:
        self.host = host
        self.port = port
        self.path = path
        self.user = user
        self.url = f'http://{self.host}:{self.port}'
        self.auth_url = f'http://{user.access_key}:{user.secret_key}@{self.host}:{self.port}'
        self._process = None

        env = os.environ.copy()
        env['MINIO_BROWSER'] = 'off'
        env['MINIO_ACCESS_KEY'] = self.user.access_key
        env['MINIO_SECRET_KEY'] = self.user.secret_key
        try:
            self._process = subprocess.Popen(
                [
                    'minio', 'server', '--quiet',
                    '--address', f'{self.host}:{self.port}',
                    '-C', str(self.path / 'config'),
                    str(self.path / 'data'),
                ],
                stdout=subprocess.DEVNULL,
                env=env
            )
        except OSError as exc:
            raise MissingProgram(f'Could not run minio: {exc}') from exc

        with contextlib.ExitStack() as exit_stack:
            exit_stack.callback(self._process.terminate)
            health_url = urllib.parse.urljoin(self.url, '/minio/health/live')
            for i in range(100):
                try:
                    with requests.get(health_url) as resp:
                        if resp.ok:
                            break
                except requests.ConnectionError:
                    pass
                if self._process.poll() is not None:
                    raise ProgramFailed('Minio died before it became healthy')
                time.sleep(0.1)
            else:
                raise ProgramFailed('Timed out waiting for minio to be ready')
            exit_stack.pop_all()

    def wipe(self) -> None:
        """Remove all buckets and objects, but leave the server running.

        See :meth:`mc` for information about exceptions.
        """
        self.mc('rb', '--force', '--dangerous', 'minio')

    def close(self) -> None:
        """Shut down the server."""
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None

    def __enter__(self) -> 'S3Server':
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        self.close()

    def mc(self, *args) -> None:
        """Run a (minio) mc subcommand against the running server.

        The running server has the alias ``minio``.

        .. note::

           The credentials will be exposed in the environment. This is only
           intended for unit testing, and hence not with sensitive
           credentials.

        Raises
        ------
        MissingProgram
            if the ``mc`` command is not found on the path
        ProgramFailed
            if the command returned a non-zero exit status. The exception
            message will include the stderr output.
        """
        env = os.environ.copy()
        env['MC_HOST_minio'] = self.auth_url
        # --config-dir is set just to prevent any config set by the user
        # from interfering with the test.
        try:
            subprocess.run(
                [
                    'mc', '--quiet', '--no-color', f'--config-dir={self.path}',
                    *args
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=env,
                encoding='utf-8',
                errors='replace',
                check=True
            )
        except OSError as exc:
            raise MissingProgram(f'mc could not be run: {exc}') from exc
        except subprocess.CalledProcessError as exc:
            raise ProgramFailed(exc.stderr) from exc
