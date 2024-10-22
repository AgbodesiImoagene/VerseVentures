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
from functools import lru_cache
import gc
import logging
from pathlib import Path
from pickle import loads

from PyQt5 import QtWidgets as QWidgets

from openlp.core.common import delete_file
from openlp.core.common.enum import LanguageSelection
from openlp.core.common.applocation import AppLocation
from openlp.core.common.i18n import UiStrings, translate
from openlp.core.common.mixins import LogMixin, RegistryProperties
from openlp.core.common.registry import Registry
from openlp.core.db.manager import DBManager
from openlp.core.threading import run_thread
from openlp.plugins.bibles.lib import ModelInfo, ModelLibrary, ModelType, parse_reference
from openlp.plugins.bibles.lib.db import BibleDB, Model, init_schema
from openlp.plugins.bibles.lib.workers.embed import EmbeddingWorker

from .importers.csvbible import CSVBible
from .importers.http import HTTPBible
from .importers.opensong import OpenSongBible
from .importers.osis import OSISBible
from .importers.wordproject import WordProjectBible
from .importers.zefania import ZefaniaBible


try:
    from .importers.sword import SwordBible
except ImportError:
    pass

log = logging.getLogger(__name__)


class BibleFormat(object):
    """
    This is a special enumeration class that holds the various types of Bibles.
    """
    Unknown = -1
    OSIS = 0
    CSV = 1
    OpenSong = 2
    WebDownload = 3
    Zefania = 4
    SWORD = 5
    WordProject = 6

    @staticmethod
    def get_class(bible_format):
        """
        Return the appropriate implementation class.

        :param bible_format: The Bible format.
        """
        if bible_format == BibleFormat.OSIS:
            return OSISBible
        elif bible_format == BibleFormat.CSV:
            return CSVBible
        elif bible_format == BibleFormat.OpenSong:
            return OpenSongBible
        elif bible_format == BibleFormat.WebDownload:
            return HTTPBible
        elif bible_format == BibleFormat.Zefania:
            return ZefaniaBible
        elif bible_format == BibleFormat.SWORD:
            return SwordBible
        elif bible_format == BibleFormat.WordProject:
            return WordProjectBible
        else:
            return None

    @staticmethod
    def get_formats_list():
        """
        Return a list of the supported Bible formats.
        """
        return [
            BibleFormat.OSIS,
            BibleFormat.CSV,
            BibleFormat.OpenSong,
            BibleFormat.WebDownload,
            BibleFormat.Zefania,
            BibleFormat.SWORD,
            BibleFormat.WordProject
        ]


class BibleManager(LogMixin, RegistryProperties):
    """
    The Bible manager which holds and manages all the Bibles.
    """
    log.info('Bible manager loaded')

    def __init__(self, parent):
        """
        Finds all the bibles defined for the system and creates an interface object for each bible containing
        connection information. Throws Exception if no Bibles are found.

        Init confirms the bible exists and stores the database path.
        """
        log.debug('Bible Initialising')
        self.parent = parent

        self.web = 'Web'
        self.db_cache = None
        self.path = AppLocation.get_section_data_path('bibles')
        self.suffix = '.sqlite'
        self.model_path = AppLocation.get_section_data_path('models')
        self.model_manager = DBManager('models', init_schema)
        self.encoder_model = None
        self.models_cache = None
        self.suffix = '.sqlite'
        self.import_wizard = None
        self.reload_bibles()
        self.reload_models()
        self.media = None

    def reload_bibles(self):
        """
        Reloads the Bibles from the available Bible databases on disk. If a web Bible is encountered, an instance
        of HTTPBible is loaded instead of the BibleDB class.
        """
        log.debug('Reload bibles')
        file_paths = AppLocation.get_files('bibles', self.suffix)
        if Path('alternative_book_names.sqlite') in file_paths:
            file_paths.remove(Path('alternative_book_names.sqlite'))
        log.debug('Bible Files {text}'.format(text=file_paths))
        self.db_cache = {}
        for file_path in file_paths:
            bible = BibleDB(self.parent, path=self.path, file=file_path)
            if not bible.session:
                continue
            name = bible.get_name()
            # Remove corrupted files.
            if name is None:
                bible.session.close()
                bible.session = None
                gc.collect()
                delete_file(self.path / file_path)
                continue
            log.debug('Bible Name: "{name}"'.format(name=name))
            self.db_cache[name] = bible
            # Look to see if lazy load bible exists and get create getter.
            if self.db_cache[name].is_web_bible:
                source = self.db_cache[name].get_object(bible.BibleMeta, 'download_source')
                download_name = self.db_cache[name].get_object(bible.BibleMeta, 'download_name').value
                web_bible = HTTPBible(self.parent, path=self.path, file=file_path, download_source=source.value,
                                      download_name=download_name)
                self.db_cache[name] = web_bible
        log.debug('Bibles reloaded')

    def reload_models(self):
        """
        Reloads the models from the available model databases on disk.
        """
        log.debug('Reload models')
        all_models = self.model_manager.get_all_objects(Model, Model.downloaded == True)
        self.models_cache = {model.name: model for model in all_models}

    def set_process_dialog(self, wizard):
        """
        Sets the reference to the dialog with the progress bar on it.

        :param wizard: The reference to the import wizard.
        """
        self.import_wizard = wizard

    def import_bible(self, type, **kwargs):
        """
        Register a bible in the bible cache, and then import the verses.

        :param type: What type of Bible, one of the ``BibleFormat`` values.
        :param kwargs: Keyword arguments to send to the actual importer class.
        """
        class_ = BibleFormat.get_class(type)
        kwargs['path'] = self.path
        importer = class_(self.parent, **kwargs)
        name = importer.register(self.import_wizard)
        self.db_cache[name] = importer
        return importer

    def import_model(self, model, **kwargs):
        model.register(self.import_wizard)
        model.download()
        self.save_model(model)

    def delete_bible(self, name):
        """
        Delete a bible completely.

        :param name: The name of the bible.
        """
        log.debug('BibleManager.delete_bible("{name}")'.format(name=name))
        bible = self.db_cache[name]
        bible.session.close()
        bible.session = None
        gc.collect()
        return delete_file(bible.path / '{name}{suffix}'.format(name=name, suffix=self.suffix))

    def on_bible_encoding_finished(self, model_name, bible_name):
        key = 'bible_embedding_{model}'.format(model=model_name)
        self.db_cache[bible_name].save_meta(key, True)
        QWidgets.QMessageBox.information(
            self.application.main_window,
            translate("BiblesPlugin.BibleManager", "Bible Encoding Finished"),
            translate(
                "BiblesPlugin.BibleManager",
                "Bible {bible} has been encoded with model {model}. You can now "
                "use the semantic search feature of this Bible with this model.",
            ).format(bible=bible_name, model=model_name),
        )
        self.application.process_events()

    def _encode_bible(self, bible, model):
        log.debug('Encoding Bible {bible} with {model}'.format(bible=bible.name, model=model.name))
        if not bible or not model:
            return
        encode_worker = EmbeddingWorker(model, bible)
        encode_worker.embedding_finished.connect(self.on_bible_encoding_finished)
        thread_name = "encode-worker-{bible}-{model}-{id}".format(
            bible=bible.name, model=model.name, id=id(encode_worker)
        )
        run_thread(encode_worker, thread_name)

    def encode_bibles(self):
        for db_model in self.get_models(type=ModelType.ENCODER).values():
            model = self.load_model(db_model)
            for bible in self.db_cache.values():
                key = 'bible_embedding_{model}'.format(model=model.name)
                if not bible.get_object(bible.BibleMeta, key) and not bible.is_web_bible:
                    self._encode_bible(bible, model)

    def get_bibles(self):
        """
        Returns a dict with all available Bibles.
        """
        log.debug('get_bibles')
        return self.db_cache

    def get_books(self, bible):
        """
        Returns a list of Bible books, and the number of chapters in that book.

        :param bible: Unicode. The Bible to get the list of books from.
        """
        log.debug('BibleManager.get_books("{bible}")'.format(bible=bible))
        return [
            {
                'name': book.name,
                'book_reference_id': book.book_reference_id,
                'chapters': self.db_cache[bible].get_chapter_count(book)
            }
            for book in self.db_cache[bible].get_books()
        ]

    def get_book_by_id(self, bible, id):
        """
        Returns a book object by given id.

        :param bible: Unicode. The Bible to get the list of books from.
        :param id: Unicode. The book_reference_id to get the book for.
        """
        log.debug('BibleManager.get_book_by_id("{bible}", "{ref}")'.format(bible=bible, ref=id))
        return self.db_cache[bible].get_book_by_book_ref_id(id)

    def get_chapter_count(self, bible, book):
        """
        Returns the number of Chapters for a given book.

        :param bible: Unicode. The Bible to get the list of books from.
        :param book: The book object to get the chapter count for.
        """
        log.debug('BibleManager.get_book_chapter_count ("{bible}", "{name}")'.format(bible=bible, name=book.name))
        return self.db_cache[bible].get_chapter_count(book)

    def get_verse_count(self, bible, book, chapter):
        """
        Returns all the number of verses for a given book and chapterMaxBibleBookVerses.
        """
        log.debug('BibleManager.get_verse_count("{bible}", "{book}", {chapter})'.format(bible=bible,
                                                                                        book=book,
                                                                                        chapter=chapter))
        language_selection = self.get_language_selection(bible)
        book_ref_ids = self.db_cache[bible].get_book_ref_id_by_localised_name(book, language_selection)
        if book_ref_ids:
            return self.db_cache[bible].get_verse_count(book_ref_ids[0], chapter)
        return 0

    def get_verse_count_by_book_ref_id(self, bible, book_ref_id, chapter):
        """
        Returns all the number of verses for a given
        book_ref_id and chapterMaxBibleBookVerses.
        """
        log.debug('BibleManager.get_verse_count_by_book_ref_id("{bible}", '
                  '"{book}", "{chapter}")'.format(bible=bible, book=book_ref_id, chapter=chapter))
        return self.db_cache[bible].get_verse_count(book_ref_id, chapter)

    def parse_ref(self, bible, reference_text, book_ref_id=False):
        if not bible:
            return
        language_selection = self.get_language_selection(bible)
        return parse_reference(reference_text, self.db_cache[bible], language_selection, book_ref_id)

    def get_verses(self, bible, ref_list, show_error=True):
        """
        Parses a scripture reference, fetches the verses from the Bible
        specified, and returns a list of ``Verse`` objects.

        :param bible: Unicode. The Bible to use.
        :param verse_text:
             String. The scripture reference. Valid scripture references are:

                - Genesis 1
                - Genesis 1-2
                - Genesis 1:1
                - Genesis 1:1-10
                - Genesis 1:1-10,15-20
                - Genesis 1:1-2:10
                - Genesis 1:1-10,2:1-10

        :param book_ref_id:  Unicode. The book reference id from the book in verse_text.
            For second bible this is necessary.
        :param show_error:
        """
        if not bible or not ref_list:
            return []
        return self.db_cache[bible].get_verses(ref_list, show_error)

    def get_language_selection(self, bible):
        """
        Returns the language selection of a bible.

        :param bible:  Unicode. The Bible to get the language selection from.
        """
        log.debug('BibleManager.get_language_selection("{bible}")'.format(bible=bible))
        language_selection = self.get_meta_data(bible, 'book_name_language')
        if not language_selection or language_selection.value == "None" or language_selection.value == "-1":
            # If None is returned, it's not the singleton object but a
            # BibleMeta object with the value "None"
            language_selection = Registry().get('settings').value('bibles/book name language')
        else:
            language_selection = language_selection.value
        try:
            language_selection = int(language_selection)
        except (ValueError, TypeError):
            language_selection = LanguageSelection.Application
        return language_selection

    def verse_search(self, bible, text):
        """
        Does a verse search for the given bible and text.

        :param str bible: The bible to search
        :param str text: The text to search for
        :return: The search results if valid, or None if the search is invalid.
        :rtype: None | list
        """
        log.debug('BibleManager.verse_search("{bible}", "{text}")'.format(bible=bible, text=text))
        if not text:
            return None
        # If no bibles are installed, message is given.
        if not bible:
            self.main_window.information_message(
                UiStrings().BibleNoBiblesTitle,
                UiStrings().BibleNoBibles)
            return None
        # Check if the bible is a web bible.
        if self.db_cache[bible].is_web_bible:
            # If Bible is Web, cursor is reset to normal and message is given.
            self.application.set_normal_cursor()
            self.main_window.information_message(
                translate('BiblesPlugin.BibleManager', 'Web Bible cannot be used in Text Search'),
                translate('BiblesPlugin.BibleManager', 'Text Search is not available with Web Bibles.\n'
                                                       'Please use the Scripture Reference Search instead.\n\n'
                                                       'This means that the currently selected Bible is a Web Bible.')
            )
            return None
        # Fetch the results from db. If no results are found, return None, no message is given for this.
        return self.db_cache[bible].verse_search(text)

    def similarity_search(self, bible, text, similarity_threshold=0.5, max_results=10):
        """
        Does a similarity search for the given bible and text.

        :param str bible: The bible to search
        :param str text: The text to search for
        :return: The search results if valid, or an empty list if the search is invalid.
        :rtype: list
        """
        log.debug('BibleManager.similarity_search("{bible}", "{text}")'.format(bible=bible, text=text))
        if not text:
            return []
        # If no bibles are installed, message is given.
        if not bible:
            self.main_window.information_message(
                UiStrings().BibleNoBiblesTitle,
                UiStrings().BibleNoBibles)
            return []
        # Check if the bible is a web bible.
        if self.db_cache[bible].is_web_bible:
            # If Bible is Web, cursor is reset to normal and message is given.
            self.application.set_normal_cursor()
            self.main_window.information_message(
                translate('BiblesPlugin.BibleManager', 'Web Bible cannot be used in Semantic Search'),
                translate('BiblesPlugin.BibleManager', 'Semantic Search is not available with Web Bibles.\n'
                                                       'Please use the Scripture Reference Search instead.\n\n'
                                                       'This means that the currently selected Bible is a Web Bible.')
            )
            return []
        if self.encoder_model is None:
            self.main_window.information_message(
                translate('BiblesPlugin.BibleManager', 'No Encoder Model Selected'),
                translate('BiblesPlugin.BibleManager', 'Please select an encoder model to use for semantic search.')
            )
            return []
        # Fetch the embeddings from db. If no results are found, return None, no message is given for this.
        verse_ids, encodings = zip(*self.get_encodings(bible, self.encoder_model.name))
        similarities = self.encoder_model.similarity(text, encodings)
        verse_similarity = zip(verse_ids, similarities)
        verse_similarity = sorted(verse_similarity, key=lambda x: x[1], reverse=True)
        # Filter out duplicate verses, keeping the highest similarity.
        results = []
        for verse_id, similarity in verse_similarity:
            if verse_id not in results and similarity >= similarity_threshold:
                results.append(verse_id)
                if len(results) >= max_results:
                    break
        return self.db_cache[bible].get_verses_by_id(results)

    def process_verse_range(self, book_ref_id, chapter_from, verse_from, chapter_to, verse_to):
        verse_ranges = []
        for chapter in range(chapter_from, chapter_to + 1):
            if chapter == chapter_from:
                start_verse = verse_from
            else:
                start_verse = 1
            if chapter == chapter_to:
                end_verse = verse_to
            else:
                end_verse = -1
            verse_ranges.append((book_ref_id, chapter, start_verse, end_verse))
        return verse_ranges

    def save_meta_data(self, bible, version, copyright, permissions, full_license, book_name_language=None):
        """
        Saves the bibles meta data.
        """
        log.debug('save_meta data {bible}, {version}, {copyright},'
                  ' {perms}, {full_license}'.format(bible=bible, version=version, copyright=copyright,
                                                    perms=permissions, full_license=full_license))
        self.db_cache[bible].save_meta('name', version)
        self.db_cache[bible].save_meta('copyright', copyright)
        self.db_cache[bible].save_meta('permissions', permissions)
        self.db_cache[bible].save_meta('full_license', full_license)
        self.db_cache[bible].save_meta('book_name_language', book_name_language)

    def get_meta_data(self, bible, key):
        """
        Returns the meta data for a given key.
        """
        log.debug('get_meta {bible},{key}'.format(bible=bible, key=key))
        bible_db = self.db_cache[bible]
        return bible_db.get_object(bible_db.BibleMeta, key)

    def update_book(self, bible, book):
        """
        Update a book of the bible.
        """
        log.debug('BibleManager.update_book("{bible}", "{name}")'.format(bible=bible, name=book.name))
        self.db_cache[bible].update_book(book)

    def save_model(self, model):
        """
        Save the model to the database.

        :param model: The model to save.
        """
        log.debug('Committing %s model to database', model.name)
        model_info = ModelInfo.get_model_info(model.name)
        db_model = self.model_manager.get_object_filtered(Model, Model.name == model.name)
        if db_model is None:
            db_model = Model()
            db_model.name = model.name
            db_model.type = model_info.get('type')
            db_model.library = model_info.get('library')
            db_model.description = model_info.get('description')
        db_model.path = str(model.path)
        if model.is_downloaded() and not db_model.downloaded:
            db_model.downloaded = True
            db_model.download_source = model.url
        db_model.meta = model.model_info
        self.model_manager.save_object(db_model)

    def get_models(self, type: ModelType | None = None):
        """
        Get all the models from the cache.

        :param type: The type of model to get.
        :param downloaded: Whether to get only downloaded models.
        :return: Available models.
        """
        if type is None or not self.models_cache:
            return {}
        return {name: model for name, model in self.models_cache.items() if model.type == type}

    def load_model(self, db_model):
        """
        Load a requested model.

        :param model_name: The name of the model to load.
        :return: The model object.
        """
        if isinstance(db_model, str):
            db_model = self.models_cache.get(db_model)
        if db_model is None or not db_model.downloaded:
            return None
        model_class = None
        if db_model.library == ModelLibrary.WHISPER:
            from openlp.plugins.bibles.lib.models.whispertranscriber import WhisperTranscriberModel
            model_class = WhisperTranscriberModel
        elif db_model.library == ModelLibrary.SPEECHBRAIN:
            from openlp.plugins.bibles.lib.models.sptranscriber import SpeechBrainTranscriberModel
            model_class = SpeechBrainTranscriberModel
        elif db_model.library == ModelLibrary.SENTENCE_TRANSFORMERS:
            from openlp.plugins.bibles.lib.models.stencoder import SentenceTransformerEncoderModel
            model_class = SentenceTransformerEncoderModel
        elif db_model.library == ModelLibrary.TENSORFLOW:
            from openlp.plugins.bibles.lib.models.tfencoder import TensorFlowEncoderModel
            model_class = TensorFlowEncoderModel

        model = model_class(db_model.name, self)
        model.load()
        return model

    @lru_cache(maxsize=8)
    def get_encodings(self, bible, model_name):
        """
        Load the encodings for a bible.

        :param str bible: The bible to load the encodings for.
        :param str model_name: The name of the model to load.
        """
        if not bible or not model_name:
            return None
        bible_model = self.db_cache[bible]
        if bible_model.is_web_bible:
            return None
        key = 'bible_embedding_{model}'.format(model=model_name)
        if not bible_model.get_object(bible_model.BibleMeta, key):
            return []
        results = bible_model.get_all_objects(bible_model.Encoding, bible_model.Encoding.model_name == model_name)
        return [(result.verse_id, loads(result.encoding)) for result in results]

    def set_encoder_model(self, model_name):
        """
        Set the encoder model.

        :param model_name: The name of the model to set.
        """
        if model_name in self.get_models(type=ModelType.ENCODER).keys():
            self.encoder_model = self.load_model(model_name)

    def exists(self, name):
        """
        Check cache to see if new bible.
        """
        if not isinstance(name, str):
            name = str(name)
        for bible in list(self.db_cache.keys()):
            log.debug('Bible from cache in is_new_bible {bible}'.format(bible=bible))
            if not isinstance(bible, str):
                bible = str(bible)
            if bible == name:
                return True
        return False

    def finalise(self):
        """
        Loop through the databases to VACUUM them.
        """
        for bible in self.db_cache:
            self.db_cache[bible].finalise()


__all__ = ['BibleFormat', 'BibleManager']
