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
from urllib.request import urlretrieve

import numpy as np
import torch
from whisper import load_model

from openlp.plugins.bibles.lib.model import TranscriberModel

log = logging.getLogger(__name__)


class WhisperTranscriberModel(TranscriberModel):
    def __init__(self, name: str, manager, *args, **kwargs):
        """
        Constructor

        :param name: The name of the model.
        :param manager: The manager.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        """
        log.debug("Loading WhisperTranscriberModel: %s", name)
        super().__init__(name, manager, *args, **kwargs)
        self.filename = self.path / self.url.split('/')[-1]
        self.gpu = self.has_gpu()

    def download(self):
        """
        Download the model.
        """
        if self.is_downloaded():
            return self.path
        log.debug("Downloading WhisperTranscriberModel: %s", self.name)
        urlretrieve(self.url, self.filename)
        return self.path

    def is_downloaded(self):
        """
        Check if the model is downloaded.

        :return: True if the model is downloaded, False otherwise.
        """
        return self.filename.exists()

    def load(self):
        """
        Load the model.

        :return: The model.
        """
        if not self.model:
            if not self.is_downloaded():
                self.download()
            log.debug("Loading WhisperTranscriberModel: %s", self.name)
            self.model = load_model(self.filename)

    def transcribe(self, audio: np.ndarray, *args, **kwargs) -> str:
        """
        Transcribe the audio.

        :param audio: The audio.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        :return: The transcription.
        """
        if not self.model:
            self.load()
        return self.model.transcribe(audio, *args, fp16=self.gpu, **kwargs)['text']

    def has_gpu(self):
        return torch.cuda.is_available()
