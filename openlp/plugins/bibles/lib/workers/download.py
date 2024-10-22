# -*- coding: utf-8 -*-

##########################################################################
# OpenLP - Open Source Lyrics Projection                                 #
# ---------------------------------------------------------------------- #
# Copyright (c) 2008-2024 OpenLP Developers                              #
# ---------------------------------------------------------------------- #
# This program is free software: you can redistribute it and/or modify   #
# it under the terms of the GNU General Public License as published by   #
# the Free Software Foundation, either version 3 of the License, or      #
# (at your option) any later version.                                    #
#                                                                        #
# This program is distributed in the hope that it will be useful,        #
# but WITHOUT ANY WARRANTY; without even the implied warranty of         #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          #
# GNU General Public License for more details.                           #
#                                                                        #
# You should have received a copy of the GNU General Public License      #
# along with this program.  If not, see <https://www.gnu.org/licenses/>. #
##########################################################################

import logging
from pathlib import Path
import tarfile
from typing import List
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import urlopen

from PyQt5 import QtCore
import requests

from openlp.core.common.httputils import get_url_file_size
from openlp.core.common.i18n import translate
from openlp.core.threading import ThreadWorker


log = logging.getLogger(__name__)


class ModelDownloadWorker(ThreadWorker):
    """
    This worker allows a file to be downloaded in a thread
    """
    download_finished = QtCore.pyqtSignal(list, dict)
    download_progress = QtCore.pyqtSignal(str, float)

    def __init__(self, base_url: str, files: List[str], download_dir: Path):
        """
        Set up the worker object
        """
        super().__init__()
        log.debug('ModelDownloadWorker - Initialise')
        self._base_url = base_url
        self._files = files if files else ['']
        self._download_dir = download_dir
        self._downloaded_size = 0
        self._total_size = 0
        self._downloaded_files = []
        self._failed_files = {}
        self._file_sizes = {}
        self.is_cancelled = False
        self.current_file = None

    def start(self):
        """
        Download the files from the base URL to the download directory
        """
        log.debug('ModelDownloadWorker - Start')
        if self.is_cancelled:
            self.quit.emit()
            return
        self._file_sizes = {file: get_url_file_size(urljoin(self._base_url, file)) for file in self._files}
        self._total_size = sum(self._file_sizes.values())

        for file in self._files:
            if self.is_cancelled:
                for downloaded_file in self._downloaded_files:
                    if downloaded_file.exists():
                        downloaded_file.unlink()
                self.quit.emit()
                return
            try:
                self.download(file)
            except (requests.RequestException, tarfile.TarError, URLError) as e:
                log.exception('Unable to download %s', self.current_file)
                self._failed_files[file] = e
                return
        self.download_finished.emit(self._downloaded_files, self._failed_files)
        self.quit.emit()
        return self._downloaded_files, self._failed_files

    def download(self, file: str):
        self.current_file = file if file else 'model'
        tar_size = self._get_remote_tarfile_size(file)

        if tar_size:
            self._file_sizes[file] = tar_size
            self._total_size = sum(self._file_sizes.values())
            self._download_tarfile(file)
        else:
            self._download_file(file)
        log.debug('Downloaded %s', self.current_file)

    def _get_remote_tarfile_size(self, file: str):
        """
        Get the size of a remote tar file
        """
        try:
            with urlopen(urljoin(self._base_url, file)) as f_stream:
                with tarfile.open(mode="r|*", fileobj=f_stream) as tgz:
                    return sum([member.size for member in tgz])
        except tarfile.TarError:
            return 0

    def _download_tarfile(self, file: str):
        """
        Download a tar file
        """
        with urlopen(urljoin(self._base_url, file)) as f_stream:
            with tarfile.open(mode="r|*", fileobj=f_stream) as tgz:
                self.download_progress.emit(
                    translate("BiblesPlugin", "Downloading {0}").format(
                        self.current_file
                    ),
                    self._downloaded_size / self._total_size,
                )
                for tarinfo in tgz:
                    tgz.extract(tarinfo, self._download_dir, filter="data")
                    self._downloaded_files.append(tarinfo.name)
                    self._downloaded_size += tarinfo.size
                    self.download_progress.emit(
                        translate("BiblesPlugin", "Unpacking {0}").format(
                            tarinfo.name
                        ),
                        self._downloaded_size / self._total_size,
                    )

    def _download_file(self, file: str):
        """
        Download a file
        """
        dest_path = self._download_dir / Path(self.current_file)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_size = 1024 * 1024
        resp = requests.get(urljoin(self._base_url, file), stream=True, timeout=10)
        resp.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                    self._downloaded_size += len(chunk)
                    self.download_progress.emit(
                        translate("BiblesPlugin", "Downloading {0}").format(
                            self.current_file
                        ),
                        self._downloaded_size / self._total_size,
                    )
            self._downloaded_files.append(self.current_file)

    @QtCore.pyqtSlot()
    def cancel_download(self):
        """
        A slot to allow the download to be cancelled from outside of the thread
        """
        self.is_cancelled = True
