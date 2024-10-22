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
        super(WhisperTranscriberModel, self).__init__(name, manager, *args, **kwargs)
        self.gpu = self.has_gpu()

    def download(self):
        """
        Download the model.
        """
        log.debug("Downloading WhisperTranscriberModel: %s", self.name)
        self._download(self.url, self.model_info['file_list'], self.path)

    def load(self):
        """
        Load the model.

        :return: The model.
        """
        if not self.model:
            log.debug("Loading WhisperTranscriberModel: %s", self.name)
            device = torch.device("cuda" if self.gpu else "cpu")
            self.model = load_model(str(self.path / self.model_info['file_list'][0]), device=device)

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
