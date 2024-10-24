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
"""
The bible import functions for OpenLP
"""
import logging

from PyQt5 import QtCore, QtWidgets

from openlp.core.common import trace_error_handler
from openlp.core.common.applocation import AppLocation
from openlp.core.common.i18n import UiStrings, get_locale_key, translate
from openlp.core.common.path import create_paths
from openlp.core.lib.ui import critical_error_message_box
from openlp.core.widgets.enums import PathEditType
from openlp.core.widgets.edits import PathEdit
from openlp.core.widgets.wizard import OpenLPWizard, WizardStrings
from openlp.plugins.bibles.lib.db import clean_filename
from openlp.plugins.bibles.lib import ModelInfo, ModelLibrary, ModelType, get_size_from_string


log = logging.getLogger(__name__)


class ModelDownloadForm(OpenLPWizard):
    """
    This is the Model Download Wizard, which allows easy downloading of text encoding and audio transcription models.
    """
    log.info('ModelDownloadForm loaded')

    def __init__(self, parent, manager, bible_plugin):
        """
        Instantiate the wizard, and run any extra setup we need to.

        :param parent: The QWidget-derived parent of the wizard.
        :param manager: The Bible manager.
        :param bible_plugin: The Bible plugin.
        """
        self.manager = manager
        self.web_bible_list = {}
        super(ModelDownloadForm, self).__init__(parent, bible_plugin,
                                                'modelDownloadWizard', ':/wizards/wizard_downloadmodel.bmp')

    def setup_ui(self, image):
        """
        Set up the UI for the model wizard.
        """
        super(ModelDownloadForm, self).setup_ui(image)
        self.model_type_combo_box.currentIndexChanged.connect(self.on_current_index_changed)

    def on_current_index_changed(self, index):
        """
        Called when the format combo box's index changed. We have to check if
        the import is available and accordingly to disable or enable the next
        button.
        """
        self.select_stack.setCurrentIndex(index)

    def custom_init(self):
        """
        Perform any custom initialisation for bible importing.
        """
        self.manager.set_process_dialog(self)
        self.restart()
        self.select_stack.setCurrentIndex(0)

    def custom_signals(self):
        """
        Set up the signals used in the bible importer.
        """
        pass

    def add_custom_pages(self):
        """
        Add the model import specific wizard pages.
        """
        # Select Page
        self.select_page = QtWidgets.QWizardPage()
        self.select_page.setObjectName('SelectPage')
        self.select_page_layout = QtWidgets.QVBoxLayout(self.select_page)
        self.select_page_layout.setObjectName('SelectPageLayout')
        self.model_type_layout = QtWidgets.QFormLayout()
        self.model_type_layout.setObjectName('ModelTypeLayout')
        self.model_type_label = QtWidgets.QLabel(self.select_page)
        self.model_type_label.setObjectName('ModelTypeLabel')
        self.model_type_combo_box = QtWidgets.QComboBox(self.select_page)
        self.model_type_combo_box.addItems(['', ''])
        self.model_type_combo_box.setObjectName('ModelTypeComboBox')
        self.model_type_layout.addRow(self.model_type_label, self.model_type_combo_box)
        self.spacer = QtWidgets.QSpacerItem(10, 0, QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.model_type_layout.setItem(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.spacer)
        self.select_page_layout.addLayout(self.model_type_layout)
        self.select_stack = QtWidgets.QStackedLayout()
        self.select_stack.setObjectName('SelectStack')
        self.embedding_models_widget = QtWidgets.QWidget(self.select_page)
        self.embedding_models_widget.setObjectName('EmbeddingModelsWidget')
        self.embedding_models_layout = QtWidgets.QFormLayout(self.embedding_models_widget)
        self.embedding_models_layout.setObjectName('EmbeddingModelsLayout')
        self.embedding_models_table = QtWidgets.QTableWidget(self.embedding_models_widget)
        self.embedding_models_table.setObjectName('EmbeddingModelsTable')
        self.embedding_models_table.setColumnCount(5)
        self.embedding_models_table.setRowCount(0)
        self.embedding_models_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.embedding_models_table.setAlternatingRowColors(True)
        self.embedding_models_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.embedding_models_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.embedding_models_table.setSortingEnabled(True)
        self.embedding_models_table.horizontalHeader().setSortIndicator(0, QtCore.Qt.SortOrder.AscendingOrder)
        self.embedding_models_table.horizontalHeader().setSortIndicator(4, QtCore.Qt.SortOrder.AscendingOrder)
        self.embedding_models_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.embedding_models_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.embedding_models_table.horizontalHeader().setDefaultSectionSize(250)
        self.embedding_models_table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        item = QtWidgets.QTableWidgetItem()
        self.embedding_models_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.embedding_models_table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.embedding_models_table.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.embedding_models_table.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.embedding_models_table.setHorizontalHeaderItem(4, item)
        self.embedding_models_layout.setWidget(0, QtWidgets.QFormLayout.ItemRole.SpanningRole,
                                               self.embedding_models_table)
        self.select_stack.addWidget(self.embedding_models_widget)
        self.transcription_models_widget = QtWidgets.QWidget(self.select_page)
        self.transcription_models_widget.setObjectName('TranscriptionModelsWidget')
        self.transcription_models_layout = QtWidgets.QFormLayout(self.transcription_models_widget)
        self.transcription_models_layout.setObjectName('TranscriptionModelsLayout')
        self.transcription_models_table = QtWidgets.QTableWidget(self.transcription_models_widget)
        self.transcription_models_table.setObjectName('TranscriptionModelsTable')
        self.transcription_models_table.setColumnCount(5)
        self.transcription_models_table.setRowCount(0)
        self.transcription_models_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.transcription_models_table.setAlternatingRowColors(True)
        self.transcription_models_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.transcription_models_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.transcription_models_table.setSortingEnabled(True)
        self.transcription_models_table.horizontalHeader().setSortIndicator(0, QtCore.Qt.SortOrder.AscendingOrder)
        self.transcription_models_table.horizontalHeader().setSortIndicator(4, QtCore.Qt.SortOrder.AscendingOrder)
        self.transcription_models_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.transcription_models_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.transcription_models_table.horizontalHeader().setDefaultSectionSize(250)
        self.transcription_models_table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        item = QtWidgets.QTableWidgetItem()
        self.transcription_models_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.transcription_models_table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.transcription_models_table.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.transcription_models_table.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.transcription_models_table.setHorizontalHeaderItem(4, item)
        self.transcription_models_layout.setWidget(0, QtWidgets.QFormLayout.ItemRole.SpanningRole,
                                                   self.transcription_models_table)
        self.select_stack.addWidget(self.transcription_models_widget)
        self.select_page_layout.addLayout(self.select_stack)
        self.addPage(self.select_page)
        # Download Location Page
        self.download_location_page = QtWidgets.QWizardPage()
        self.download_location_page.setObjectName('DownloadLocationPage')
        self.download_location_layout = QtWidgets.QFormLayout(self.download_location_page)
        self.download_location_layout.setObjectName('DownloadLocationLayout')
        self.download_location_label = QtWidgets.QLabel(self.download_location_page)
        self.download_location_label.setObjectName('DownloadLocationLabel')
        default_path = self.settings.value('models/last directory download')
        default_path = default_path if default_path else AppLocation.get_section_data_path('models')
        self.download_location_edit = PathEdit(
            self.download_location_page,
            path_type=PathEditType.Directories,
            default_path=default_path,
            dialog_caption=translate('BiblesPlugin.ImportWizardForm', 'Select the location to download the model to.'),
        )
        self.download_location_edit.setObjectName('DownloadLocationEdit')
        self.download_location_layout.addRow(self.download_location_label, self.download_location_edit)
        self.download_location_layout.setItem(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.spacer)
        self.addPage(self.download_location_page)

    def _populate_models_table(self, table, models):
        """
        Populate the table with the models.

        :param table: The table to populate.
        :param models: The models to populate the table with.
        """
        table.setRowCount(len(models))
        i = 0
        for model_name, model_data in models.items():
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(model_name))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(model_data['description']))
            table.setItem(i, 2, QtWidgets.QTableWidgetItem(model_data['speed']))
            table.setItem(i, 3, QtWidgets.QTableWidgetItem(model_data['performance']))
            item = QtWidgets.QTableWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.DisplayRole, get_size_from_string(model_data['size']))
            table.setItem(i, 4, item)
            i += 1

    def retranslate_ui(self):
        """
        Allow for localisation of the model import wizard.
        """
        self.setWindowTitle(translate('BiblesPlugin.ImportWizardForm', 'Model Import Wizard'))
        self.title_label.setText(WizardStrings.HeaderStyle.format(text=translate('OpenLP.Ui',
                                                                                 'Welcome to the Model Import Wizard')))
        self.information_label.setText(
            translate('BiblesPlugin.ImportWizardForm',
                      'This wizard will help you to import embedding mmodels and '
                      'transcription models for use in OpenLP. Click the next button '
                      'below to start the process by selecting a format to import '
                      'from.'))
        self.select_page.setTitle(translate('BiblesPlugin.ImportWizardForm', 'Select Model'))
        self.select_page.setSubTitle(translate('BiblesPlugin.ImportWizardForm',
                                               'Select the model type and the model to import.'))
        self.model_type_label.setText(translate('BiblesPlugin.ImportWizardForm', 'Model Type:'))
        self.model_type_combo_box.setItemText(ModelType.ENCODER.value, translate('BiblesPlugin.ImportWizardForm',
                                                                                 'Embedding Models'))
        self.model_type_combo_box.setItemText(ModelType.TRANSCRIBER.value, translate('BiblesPlugin.ImportWizardForm',
                                                                                     'Transcription Models'))
        self.embedding_models_table.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Name'))
        self.embedding_models_table.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem('Description'))
        self.embedding_models_table.setHorizontalHeaderItem(2, QtWidgets.QTableWidgetItem('Speed'))
        self.embedding_models_table.setHorizontalHeaderItem(3, QtWidgets.QTableWidgetItem('Performance'))
        self.embedding_models_table.setHorizontalHeaderItem(4, QtWidgets.QTableWidgetItem('Size (MB)'))
        self._populate_models_table(self.embedding_models_table, ModelInfo.embedding_models)
        self.transcription_models_table.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Name'))
        self.transcription_models_table.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem('Description'))
        self.transcription_models_table.setHorizontalHeaderItem(2, QtWidgets.QTableWidgetItem('Speed'))
        self.transcription_models_table.setHorizontalHeaderItem(3, QtWidgets.QTableWidgetItem('Performance'))
        self.transcription_models_table.setHorizontalHeaderItem(4, QtWidgets.QTableWidgetItem('Size (MB)'))
        self._populate_models_table(self.transcription_models_table, ModelInfo.transcription_models)
        self.download_location_page.setTitle(translate('BiblesPlugin.ImportWizardForm', 'Download Location'))
        self.download_location_page.setSubTitle(translate('BiblesPlugin.ImportWizardForm',
                                                          'Select the location to download the model to.'))
        self.download_location_label.setText(translate('BiblesPlugin.ImportWizardForm', 'Download Directory:'))
        self.progress_page.setTitle(WizardStrings.Importing)
        self.progress_page.setSubTitle(translate('BiblesPlugin.ImportWizardForm',
                                                 'Please wait while your Model is installed.'))
        self.progress_label.setText(WizardStrings.Ready)
        self.progress_bar.setFormat('%p%')

    def validateCurrentPage(self):
        """
        Validate the current page before moving on to the next page.
        """
        if self.currentPage() == self.welcome_page:
            return True
        elif self.currentPage() == self.select_page:
            if self.field('model_type') == ModelType.ENCODER.value:
                if self.embedding_models_table.selectedItems():
                    return True
                else:
                    critical_error_message_box(UiStrings().NISs, translate('BiblesPlugin.ImportWizardForm',
                                                                           'Please select a model to import.'))
                    return False
            elif self.field('model_type') == ModelType.TRANSCRIBER.value:
                if self.transcription_models_table.selectedItems():
                    return True
                else:
                    critical_error_message_box(UiStrings().NISs, translate('BiblesPlugin.ImportWizardForm',
                                                                           'Please select a model to import.'))
                    return False
        elif self.currentPage() == self.download_location_page:
            if self.download_location_edit.path:
                return True
            else:
                critical_error_message_box(UiStrings().EmptyField,
                                           translate('BiblesPlugin.ImportWizardForm',
                                                     'You need to specify a download location.'))
                return False
        if self.currentPage() == self.progress_page:
            return True

    def on_web_source_combo_box_index_changed(self, index):
        """
        Setup the list of Bibles when you select a different source on the web download page.

        :param index: The index of the combo box.
        """
        self.web_translation_combo_box.clear()
        if self.web_bible_list and index in self.web_bible_list:
            bibles = list(self.web_bible_list[index].keys())
            bibles.sort(key=get_locale_key)
            self.web_translation_combo_box.addItems(bibles)

    def register_fields(self):
        """
        Register the bible import wizard fields.
        """
        self.select_page.registerField('model_type', self.model_type_combo_box)

    def set_defaults(self):
        """
        Set default values for the wizard pages.
        """
        self.restart()
        self.finish_button.setVisible(False)
        self.cancel_button.setVisible(True)
        # The PathEdit fields are not initialised since that does not work well with the UI Internals
        self.setField('model_type', 0)

    def pre_wizard(self):
        """
        Prepare the UI for the import.
        """
        super(ModelDownloadForm, self).pre_wizard()
        self.progress_label.setText(WizardStrings.StartingImport)
        self.application.process_events()

    def perform_wizard(self):
        """
        Perform the actual import.
        """
        model_type = self.field('model_type')
        model_name = None
        model_data = None
        if model_type == ModelType.ENCODER.value:
            model_name = self.embedding_models_table.selectedItems()[0].text()
        elif model_type == ModelType.TRANSCRIBER.value:
            model_name = self.transcription_models_table.selectedItems()[0].text()
        model_data = ModelInfo.get_model_info(model_name)
        self.settings.setValue('models/last directory download', self.download_location_edit.path)
        download_location = self.download_location_edit.path / clean_filename(model_name)
        create_paths(download_location)
        model_data['path'] = download_location
        model_library = model_data['library']
        model_class = None
        if model_library == ModelLibrary.WHISPER:
            from openlp.plugins.bibles.lib.models.whispertranscriber import WhisperTranscriberModel
            model_class = WhisperTranscriberModel
        elif model_library == ModelLibrary.SPEECHBRAIN:
            from openlp.plugins.bibles.lib.models.sptranscriber import SpeechBrainTranscriberModel
            model_class = SpeechBrainTranscriberModel
        elif model_library == ModelLibrary.SENTENCE_TRANSFORMERS:
            from openlp.plugins.bibles.lib.models.stencoder import SentenceTransformerEncoderModel
            model_class = SentenceTransformerEncoderModel
        elif model_library == ModelLibrary.TENSORFLOW:
            from openlp.plugins.bibles.lib.models.tfencoder import TensorFlowEncoderModel
            model_class = TensorFlowEncoderModel

        model = model_class(model_name, self.manager, **model_data)
        try:
            if not model.stop_import_flag:
                model.register(self)
                model.download()
                self.manager.reload_models()
                if model_type == ModelType.ENCODER.value:
                    self.manager.encode_bibles()
                self.progress_label.setText(WizardStrings.FinishedImport)
                return
        except Exception:
            log.exception('Importing model failed')
            trace_error_handler(log)

        self.progress_label.setText(translate('BiblesPlugin.ImportWizardForm', 'Your model import failed.'))
        self.application.process_events()

    def provide_help(self):
        """
        Provide help within the wizard by opening the appropriate page of the openlp manual in the user's browser
        """
        # QtGui.QDesktopServices.openUrl(QtCore.QUrl("https://manual.openlp.org/bibles.html"))
        # TODO: Implement help for the Model Download Wizard
        pass
