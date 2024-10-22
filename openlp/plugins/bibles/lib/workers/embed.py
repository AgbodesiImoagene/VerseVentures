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
from pickle import dumps
import re

from PyQt5 import QtCore

from openlp.core.threading import ThreadWorker


log = logging.getLogger(__name__)


class EmbeddingWorker(ThreadWorker):
    """
    This worker allows bible verse embeddings to be created in a thread
    """
    embedding_finished = QtCore.pyqtSignal(str, str)
    embedding_progress = QtCore.pyqtSignal(str, float)

    current_processes = []

    def __init__(self, model, bible):
        """
        Set up the worker object
        """
        super().__init__()
        log.debug('EmbeddingWorker - Initialise')
        self.model = model
        self.bible = bible
        self.is_cancelled = False

    def start(self):
        """
        Start the worker
        """
        log.debug('EmbeddingWorker - Start')
        key = f'{self.model.name} - {self.bible.name}'
        if not self.bible or not self.model or self.is_cancelled or key in EmbeddingWorker.current_processes:
            return
        EmbeddingWorker.current_processes.append(key)
        all_verses = self.bible.get_all_objects(self.bible.Verse)
        verse_texts = [(verse.id, text) for verse in all_verses for text in self._prepare_verse(verse.text)]
        # split list into ids and texts
        verse_ids, verse_texts = zip(*verse_texts)
        encodings = self.model.encode(verse_texts)
        encodings = [self.bible.Encoding(verse_id=verse_id, model_name=self.model.name, encoding=dumps(encoding))
                     for verse_id, encoding in zip(verse_ids, encodings)]
        self.bible.save_objects(encodings)
        self.embedding_finished.emit(self.model.name, self.bible.name)
        EmbeddingWorker.current_processes.remove(key)

    def _prepare_verse(self, verse):
        """
        Prepare a verse for encoding.
        """
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', verse)
        if len(sentences) > 1:
            sentences.append(verse)

        def clean_text(text):
            return text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()

        sentences = [clean_text(sentence) for sentence in sentences]
        return list(filter(lambda x: x, sentences))
