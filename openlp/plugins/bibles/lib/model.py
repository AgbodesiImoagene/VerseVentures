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
import subprocess
from typing import Dict, List

from PyQt5 import QtCore
import numpy as np

from openlp.core.common.i18n import translate
from openlp.core.common.mixins import LogMixin, RegistryProperties
from openlp.core.common.registry import Registry
from openlp.core.db.manager import DBManager
from openlp.core.threading import get_thread_worker, run_thread
from openlp.plugins.bibles.lib.db import Model, init_schema
from openlp.plugins.bibles.lib.workers.download import ModelDownloadWorker


log = logging.getLogger(__name__)


class ModelBase(QtCore.QObject, LogMixin, RegistryProperties):
    """
    This class is used as a base class for models.
    """
    download_completed = QtCore.pyqtSignal(bool)
    model_manager = DBManager('models', init_schema)

    def __init__(self, name: str, *args, **kwargs):
        """
        Constructor

        :param name: The name of the model.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        """
        super().__init__()
        self.name = name
        self.wizard = None
        self.model = None
        self.db_model = self.model_manager.get_object_filtered(Model, Model.name == self.name)
        self.model_info = {}
        metadata = {}
        if self.db_model:
            self.path = Path(self.db_model.path)
            self.url = self.db_model.download_source
            self.type = self.db_model.type
            self.library = self.db_model.library
            metadata = self.db_model.meta
        else:
            self.path = Path(kwargs['path'])
            self.url = kwargs['url']
            self.type = kwargs['type']
            self.library = kwargs['library']
        metadata.update(kwargs)
        self.model_info['file_list'] = metadata.get('file_list', [])
        self.model_info['display_name'] = metadata.get('display_name', '')
        self.model_info['description'] = metadata.get('description', '')
        self.model_info['version'] = metadata.get('version', '')
        self.model_info['author'] = metadata.get('author', '')
        self.model_info['size'] = metadata.get('size')
        self.stop_import_flag = False
        self.threading = False
        self.download_worker = None
        self.download_thread_name = None
        Registry().register_function('openlp_stop_wizard', self.stop_import)

    def download(self):
        """
        Download the model. This method must be overridden by descendant classes.
        """
        pass

    def _download(self, url: str, file_list: List[str], path: Path):
        """
        Download the model. This method must be overridden by descendant classes.
        """
        if not self.is_downloaded():
            download_worker = ModelDownloadWorker(url, file_list, path)
            download_worker.download_progress.connect(self.on_download_progress)
            download_worker.download_finished.connect(self.on_download_finished)
            if self.threading:
                self.download_thread_name = "download-worker-{id}".format(id=id(download_worker))
                run_thread(download_worker, self.download_thread_name)
            else:
                self.download_worker = download_worker
                self.download_worker.start()
                del download_worker
                self.download_worker = None

    def is_downloaded(self):
        """
        Check if the model is already downloaded. This method must be overridden by descendant classes.

        :return: True if the model is downloaded, otherwise False.
        """
        if not (self.path.exists() and self.path.is_dir() and self.model_info['file_list']):
            return False
        return all((self.path / file).exists() for file in self.model_info['file_list'])

    def log_download_error(self, file_path, reason):
        """
        This should be called, when a model could not be downloaded. It will display the error in the wizard.

        :param file_path: The path to the file that could not be downloaded.
        :param reason: The reason why the download failed. The string should be as informative as possible.
        """
        if self.wizard is None:
            return
        if self.wizard.error_report_text_edit.isHidden():
            self.wizard.error_report_text_edit.setText(
                translate('BibleManager.UI', 'The model could not be downloaded. The following error(s) occurred:'))
            self.wizard.error_report_text_edit.setVisible(True)
            self.wizard.error_copy_to_button.setVisible(True)
            self.wizard.error_save_to_button.setVisible(True)
        self.wizard.error_report_text_edit.append('- {path} ({error})'.format(path=file_path, error=reason))

    def log_error_to_wizard(self, message):
        """
        Log an error to the wizard.

        :param message: The error message.
        """
        if self.wizard is None:
            return
        if self.wizard.error_report_text_edit.isHidden():
            self.wizard.error_report_text_edit.setText(
                translate('BibleManager.UI', 'The model could not be loaded. The following error(s) occurred:'))
            self.wizard.error_report_text_edit.setVisible(True)
            self.wizard.error_copy_to_button.setVisible(True)
            self.wizard.error_save_to_button.setVisible(True)
        self.wizard.error_report_text_edit.append(message)

    def register(self, wizard):
        """
        This method basically just initialises the database. It is called from the Bible Manager when a Model is
        imported. Descendant classes may want to override this method to supply their own custom
        initialisation as well.

        :param wizard: The actual Qt wizard form.
        """
        self.wizard = wizard
        self.wizard.progress_bar.reset()
        self.wizard.progress_bar.setRange(0, 1000)
        return self.name

    def stop_import(self):
        """
        Stops the import of the model.
        """
        self.log_debug('Stopping import')
        self.stop_import_flag = True
        if self.threading and self.download_thread_name:
            download_worker = get_thread_worker(self.download_thread_name)
            if download_worker:
                download_worker.cancel_download()
        elif self.download_worker:
            self.download_worker.cancel_download()

    def load(self):
        """
        Load the model. This method must be overridden by descendant classes.
        """
        pass

    def save_to_db(self):
        """
        Save the model to the database.
        """
        for key, value in self.model_info.items():
            if not isinstance(value, (str, int, float, bool, list, dict)):
                self.model_info[key] = str(value)
        if not self.db_model:
            self.db_model = self.model_manager.get_object_filtered(Model, Model.name == self.name)
            if not self.db_model:
                self.db_model = Model(
                    name=self.name,
                    description=self.model_info.get('description', ''),
                    path=str(self.path),
                    download_source=self.url,
                    meta=self.model_info,
                    type=self.type,
                    library=self.library
                )
        else:
            self.db_model.path = str(self.path)
            self.db_model.meta = self.model_info
        self.db_model.downloaded = self.is_downloaded()
        self.model_manager.save_object(self.db_model)

    def has_gpu(self):
        """
        Check if the model has GPU support.

        :return: True if the model has GPU support, otherwise False.
        """
        try:
            subprocess.check_output('nvidia-smi')
            return True
        except Exception:
            # this command not being found can raise quite a few different errors depending on the configuration
            return False

    @QtCore.pyqtSlot(str, float)
    def on_download_progress(self, status: str, progress: float):
        """
        Update the progress.

        :param status: The status.
        :param progress: The progress.
        """
        if self.wizard:
            self.wizard.update_progress_bar(status, progress)

    @QtCore.pyqtSlot(list, dict)
    def on_download_finished(self, file_list: List[Path], errors: Dict[str, Exception]):
        """
        The download has finished.

        :param file_list: The list of files.
        :param errors: The errors.
        """
        for file_path, error in errors.items():
            self.log_download_error(file_path, str(error))
        all_files = list(map(str, file_list)) + list(errors.keys())
        self.model_info['file_list'] = all_files
        self.save_to_db()
        self.download_completed.emit(all_files == file_list)
        return file_list

    def destroy(self):
        """
        Destroy the model.
        """
        del self.model


class EncoderModel(ModelBase):
    """
    This class is used for encoder models.
    """
    def __init__(self, name: str, *args, **kwargs):
        """
        Constructor

        :param name: The name of the model.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        """
        super().__init__(name, *args, **kwargs)

    def encode(self, text: str | List[str], *args, **kwargs) -> np.ndarray:
        """
        Encode the text.

        :param text: The text(s) to encode.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        :return: The encoded text.
        """
        pass

    def similarity(self, text: str, embeddings: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Calculate the similarity between two texts or lists of texts.

        :param text: The text.
        :param embeddings: The embeddings to compare to.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        :return: The similarities of the text to the embeddings.
        """
        pass


class TranscriberModel(ModelBase):
    """
    This class is used for transcriber models.
    """
    def __init__(self, name: str, *args, **kwargs):
        """
        Constructor

        :param name: The name of the model.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        """
        super().__init__(name, *args, **kwargs)

    def transcribe(self, audio: np.ndarray, *args, **kwargs) -> str:
        """
        Transcribe the audio.

        :param audio: The audio to transcribe.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        :return: The transcription.
        """
        pass
