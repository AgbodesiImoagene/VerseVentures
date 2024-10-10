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
import subprocess
from typing import List

from PySide6 import QtCore
import numpy as np

from openlp.core.common.i18n import translate
from openlp.core.common.mixins import LogMixin, RegistryProperties
from openlp.core.common.registry import Registry


log = logging.getLogger(__name__)


class ModelBase(QtCore.QObject, LogMixin, RegistryProperties):
    """
    This class is used as a base class for models.
    """

    def __init__(self, name: str, *args, **kwargs):
        """
        Constructor

        :param name: The name of the model.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        """
        super().__init__()
        self.name = name
        self.model = None
        self.wizard = None
        self.stop_import_flag = False
        self.model_info = {}
        self.path = kwargs['path']
        self.url = kwargs['url']
        if 'display_name' in kwargs:
            self.model_info['display_name'] = kwargs['display_name']
        if 'description' in kwargs:
            self.model_info['description'] = kwargs['description']
        if 'version' in kwargs:
            self.model_info['version'] = kwargs['version']
        if 'author' in kwargs:
            self.model_info['author'] = kwargs['author']
        if 'size' in kwargs:
            self.model_info['size'] = kwargs['size']
        Registry().register_function('openlp_stop_wizard', self.stop_import)

    def download(self):
        """
        Download the model. This method must be overridden by descendant classes.
        """
        pass

    def is_downloaded(self):
        """
        Check if the model is already downloaded. This method must be overridden by descendant classes.

        :return: True if the model is downloaded, otherwise False.
        """
        pass

    def log_download_error(self, file_path, reason):
        """
        This should be called, when a model could not be downloaded. It will display the error in the wizard.

        :param file_path: The path to the file that could not be downloaded.
        :param reason: The reason why the download failed. The string should be as informative as possible.
        """
        self.set_defaults()
        if self.wizard is None:
            return
        if self.wizard.error_report_text_edit.isHidden():
            self.wizard.error_report_text_edit.setText(
                translate('BibleManager.UI', 'The model could not be downloaded. The following error(s) occurred:'))
            self.wizard.error_report_text_edit.setVisible(True)
            self.wizard.error_copy_to_button.setVisible(True)
            self.wizard.error_save_to_button.setVisible(True)
        self.wizard.error_report_text_edit.append('- {path} ({error})'.format(path=file_path, error=reason))

    def register(self, wizard):
        """
        This method basically just initialises the database. It is called from the Bible Manager when a Model is
        imported. Descendant classes may want to override this method to supply their own custom
        initialisation as well.

        :param wizard: The actual Qt wizard form.
        """
        self.wizard = wizard
        return self.name

    def stop_import(self):
        """
        Stops the import of the model.
        """
        self.log_debug('Stopping import')
        self.stop_import_flag = True

    def load(self):
        """
        Load the model. This method must be overridden by descendant classes.
        """
        pass

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
