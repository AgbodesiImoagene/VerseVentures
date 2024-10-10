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
from typing import List

from huggingface_hub import snapshot_download
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import (
    cos_sim,
    get_device_name,
    is_sentence_transformer_model,
)

from openlp.plugins.bibles.lib.model import EncoderModel


log = logging.getLogger(__name__)


class SentenceTransformerEncoderModel(EncoderModel):
    def __init__(self, name: str, manager, *args, **kwargs):
        """
        Constructor

        :param name: The name of the model.
        :param manager: The manager.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        """
        log.debug("Loading SentenceTransformerEncoderModel: %s", name)
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
        log.debug("Downloading SentenceTransformerEncoderModel: %s", self.name)
        return snapshot_download(self._get_repo_id(), local_dir=self.path)

    def is_downloaded(self):
        """
        Check if the model is downloaded.
        """
        return is_sentence_transformer_model(str(self.path))

    def load(self):
        """
        Load the model.
        """
        if not self.model:
            if not self.is_downloaded():
                self.download()
            self.model = SentenceTransformer(
                self.path, device=get_device_name(), local_files_only=True
            )

    def encode(self, text: str | List[str], *args, **kwargs) -> np.ndarray:
        """
        Encode the text.

        :param text: The text(s) to encode.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        :return: The encoded text.
        """
        if not self.model:
            self.load()
        return self.model.encode(text)

    def similarity(self, text: str, embeddings: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Calculate the similarity between two texts or lists of texts.

        :param text: The text.
        :param embeddings: The embeddings to compare to.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        :return: The similarities of the text to the embeddings.
        """
        if not self.model:
            self.load()
        return cos_sim(self.encode(text), embeddings).numpy()
