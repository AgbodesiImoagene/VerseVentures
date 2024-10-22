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
from urllib.parse import urljoin

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, get_device_name

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
        super(SentenceTransformerEncoderModel, self).__init__(name, manager, *args, **kwargs)

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
        log.debug("Downloading SentenceTransformerEncoderModel: %s", self.name)
        repo_id = self._get_repo_id()
        if not self.model_info["file_list"]:
            try:
                repo_data = requests.get(
                    urljoin(
                        "https://huggingface.co/api/models/",
                        repo_id,
                        allow_fragments=False,
                    ),
                    timeout=10,
                ).json()
                self.model_info["file_list"] = [sibling["rfilename"] for sibling in repo_data["siblings"]]
            except requests.RequestException as e:
                log.error("Failed to get model data: %s", e)
        url = urljoin("https://huggingface.co/", repo_id, allow_fragments=False)
        url = urljoin(url, "resolve/main/", allow_fragments=False)
        self._download(url, self.model_info["file_list"], self.path)

    def load(self):
        """
        Load the model.
        """
        if not self.model:
            if not self.is_downloaded():
                self.download()
            self.model = SentenceTransformer(
                str(self.path), device=get_device_name(), local_files_only=True
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
        return cos_sim(self.encode(text), embeddings).numpy().squeeze()
