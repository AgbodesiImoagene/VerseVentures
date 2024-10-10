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
from speechbrain.inference.ASR import EncoderDecoderASR
from huggingface_hub import snapshot_download
import torch

from openlp.plugins.bibles.lib.model import TranscriberModel

log = logging.getLogger(__name__)


class SpeechBrainTranscriberModel(TranscriberModel):
    def __init__(self, name: str, manager, *args, **kwargs):
        """
        Constructor

        :param name: The name of the model.
        :param manager: The manager.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        """
        log.debug("Loading SpeechBrainTranscriberModel: %s", name)
        super().__init__(name, manager, *args, **kwargs)

    def _get_repo_id(self):
        """
        Get the repository ID.

        :return: The repository ID.
        """
        return self.url.replace("https://huggingface.co/", "")

    def download(self):
        """
        Download the model.
        """
        if self.is_downloaded():
            return self.path
        log.debug("Downloading SpeechBrainTranscriberModel: %s", self.name)
        return snapshot_download(self._get_repo_id(), local_dir=self.path)

    def is_downloaded(self):
        """
        Check if the model is downloaded.

        :return: True if the model is downloaded, False otherwise.
        """
        # check if repo has hyperparams.yaml and custom.py
        if not self.path.exists():
            return False
        hyperparams = self.path / "hyperparams.yaml"
        custom = self.path / "custom.py"
        return hyperparams.exists() and custom.exists()

    def load(self):
        """
        Load the model.

        :return: The model.
        """
        if not self.model:
            if not self.is_downloaded():
                self.download()
            log.debug("Loading SpeechBrainTranscriberModel: %s", self.name)
            self.model = EncoderDecoderASR.from_hparams(source=self.path)

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
        # convert audio to tensor
        audio = torch.tensor(audio).unsqueeze(0)
        lengths = torch.tensor([1.0])
        texts, _ = self.model.transcribe_batch(audio, lengths)
        return texts[0]
