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
from urllib.request import Request, urlopen

import numpy as np
import tensorflow as tf
import tensorflow_text  # noqa
from tensorflow_hub.file_utils import extract_tarfile_to_destination

from openlp.plugins.bibles.lib.model import EncoderModel


log = logging.getLogger(__name__)

COMPRESSION_QUERY = "?tf-hub-format=compressed"


class TensorFlowEncoderModel(EncoderModel):
    def __init__(self, name: str, manager, *args, **kwargs):
        """
        Constructor

        :param name: The name of the model.
        :param manager: The manager.
        :param args: The arguments.
        :param kwargs: The keyword arguments.
        """
        log.debug("Loading TensorFlowEncoderModel: %s", name)
        super().__init__(name, manager, *args, **kwargs)

    def download(self):
        """
        Download the model.
        """
        if self.is_downloaded():
            return self.path
        log.debug("Downloading TensorFlowEncoderModel: %s", self.name)
        request = Request(self.url + COMPRESSION_QUERY)
        response = urlopen(request)
        extract_tarfile_to_destination(response, str(self.path))
        return self.path

    def is_downloaded(self):
        """
        Check if the model is downloaded.

        :return: True if the model is downloaded, False otherwise.
        """
        return any(file.suffix == ".pb" for file in self.path.iterdir())

    def load(self):
        """
        Load the model.

        :return: The model.
        """
        if not self.model:
            if not self.is_downloaded():
                self.download()
            self.model = tf.saved_model.load(str(self.path))

    def encode(self, text: str | List[str], *args, **kwargs) -> np.ndarray:
        """
        Encode the text.

        :param text: The text to encode.
        :return: The encoded text.
        """
        if not self.model:
            self.load()
        scalar = False
        if isinstance(text, str):
            scalar = True
            text = [text]
        return self.model(text).numpy()[0] if scalar else self.model(text).numpy()

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
        text_embedding = self.encode(text)
        if text_embedding.ndim == 1:
            text_embedding = text_embedding.reshape(1, -1)
        # normalize the embeddings
        text_embedding = text_embedding / np.linalg.norm(text_embedding, ord=2, axis=1, keepdims=True)
        embeddings = embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        return np.dot(text_embedding, embeddings.T).squeeze()

    def has_gpu(self):
        """
        Check if the model has GPU support.

        :return: True if the model has GPU support, False otherwise.
        """
        return tf.config.list_physical_devices("GPU") != []
