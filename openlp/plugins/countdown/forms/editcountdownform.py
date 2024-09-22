# -*- coding: utf-8 -*-
# vim: autoindent shiftwidth=4 expandtab textwidth=120 tabstop=4 softtabstop=4

###############################################################################
# OpenLP - Open Source Lyrics Projection                                      #
# --------------------------------------------------------------------------- #
# Copyright (c) 2008-2014 Raoul Snyman                                        #
# Portions copyright (c) 2008-2014 Tim Bentley, Gerald Britton, Jonathan      #
# Corwin, Samuel Findlay, Michael Gorven, Scott Guerrieri, Matthias Hub,      #
# Meinert Jordan, Armin Köhler, Erik Lundin, Edwin Lunando, Brian T. Meyer.   #
# Joshua Miller, Stevan Pettit, Andreas Preikschat, Mattias Põldaru,          #
# Christian Richter, Philip Ridout, Simon Scudder, Jeffrey Smith,             #
# Maikel Stuivenberg, Martin Thompson, Jon Tibble, Dave Warnock,              #
# Frode Woldsund, Martin Zibricky, Patrick Zimmermann                         #
# --------------------------------------------------------------------------- #
# This program is free software; you can redistribute it and/or modify it     #
# under the terms of the GNU General Public License as published by the Free  #
# Software Foundation; version 2 of the License.                              #
#                                                                             #
# This program is distributed in the hope that it will be useful, but WITHOUT #
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       #
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    #
# more details.                                                               #
#                                                                             #
# You should have received a copy of the GNU General Public License along     #
# with this program; if not, write to the Free Software Foundation, Inc., 59  #
# Temple Place, Suite 330, Boston, MA 02111-1307 USA                          #
###############################################################################

import logging

from PyQt5 import QtGui
from PyQt5.QtWidgets import QDialog

from openlp.core.common.i18n import translate

from openlp.core.common.registry import Registry
from openlp.core.lib.ui import critical_error_message_box
from openlp.plugins.countdown.lib.db import CountdownSlide
from .editcountdowndialog import Ui_CountdownEditDialog
# from .editcountdownslideform import EditCountdownSlideForm

log = logging.getLogger(__name__)


class EditCountdownForm(QDialog, Ui_CountdownEditDialog):
    """
    Class documentation goes here.
    """
    log.info('Countdown Editor loaded')

    def __init__(self, media_item, parent, manager):
        """
        Constructor
        """
        super(EditCountdownForm, self).__init__(parent)
        self.manager = manager
        self.media_item = media_item
        self.setupUi(self)
        # Create other objects and forms.
        # self.edit_slide_form = EditCountdownSlideForm(self)
        # Connecting signals and slots
        # self.preview_button.clicked.connect(self.on_preview_button_clicked)
        # self.add_button.clicked.connect(self.on_add_button_clicked)
        # self.edit_button.clicked.connect(self.on_edit_button_clicked)
        # self.edit_all_button.clicked.connect(self.on_edit_all_button_clicked)
        # self.slide_list_view.currentRowChanged.connect(self.on_current_row_changed)
        # self.slide_list_view.doubleClicked.connect(self.on_edit_button_clicked)
        Registry().register_function('theme_update_list', self.load_themes)

    def load_themes(self, theme_list):
        """
        Load a list of themes into the themes combo box.
        :param theme_list: The list of themes to load.
        """
        self.theme_combo_box.clear()
        self.theme_combo_box.addItem('')
        self.theme_combo_box.addItems(theme_list)

    def load_countdown(self, id, preview=False):
        """
        Called when editing or creating a new countdown.
        :param id: The countdown's id. If zero, then a new countdown is created
        """
        if id == 0:
            self.countdown_slide = CountdownSlide()
            self.title_edit.setText('')
            self.theme_combo_box.setCurrentIndex(0)
        else:
            self.countdown_slide = self.manager.get_object(CountdownSlide, id)
            self.title_edit.setText(self.countdown_slide.title)

            self.countdown_type_combo_box.setCurrentIndex(self.countdown_slide.countdown_type)
            if self.countdown_slide.countdown_type == 1:
                self.countdown_duration_time_edit.setTime(self.countdown_slide.countdown_duration)
            else:

                if self.countdown_slide.countdown_use_specific_date:
                    self.use_specific_date_check_box.setChecked(True)
                    self.countdown_date_date_edit.setDate(self.countdown_slide.countdown_specific_date)
                if self.countdown_slide.countdown_use_specific_time:
                    self.use_specific_time_check_box.setChecked(True)
                    self.countdown_time_time_edit.setTime(self.countdown_slide.countdown_specific_time)

            self.interval_large_combo_box.setCurrentIndex(self.countdown_slide.interval_large)
            self.interval_small_combo_box.setCurrentIndex(self.countdown_slide.interval_small)
            self.finish_action_combo_box.setCurrentIndex(self.countdown_slide.finish_action)

            self.countdown_slide.theme_name = self.theme_combo_box.currentText()
            self.media_item.auto_select_id = self.countdown_slide.id

            # theme = self.countdown_slide.theme_name
            # find_and_set_in_combo_box(self.theme_combo_box, theme)
        self.title_edit.setFocus()
        log.debug('load successful')

    def accept(self):
        """
        Override the QDialog method to check if the countdown slide has been saved before closing the dialog.
        """

        log.debug('accept')
        if self.save_countdown():
            QtGui.QDialog.accept(self)
            log.debug('Save successful')

    def save_countdown(self):
        """
        Saves the countdown.
        """
        if not self._validate():
            return False
        # sxml = CountdownXMLBuilder()
        # for count in range(self.slide_list_view.count()):
        #    sxml.add_verse_to_lyrics('countdown', str(count + 1), self.slide_list_view.item(count).text())

        self.countdown_slide.title = self.title_edit.text()
        self.countdown_slide.countdown_type = self.countdown_type_combo_box.currentIndex()
        if self.countdown_slide.countdown_type == 1:
            self.countdown_slide.countdown_duration = self.countdown_duration_time_edit.time().toPyTime()
        else:
            self.countdown_slide.countdown_use_specific_date = self.use_specific_date_check_box.isChecked()
            if self.countdown_slide.countdown_use_specific_date:
                self.countdown_slide.countdown_specific_date = self.countdown_date_date_edit.date().toPyDate()
            self.countdown_slide.countdown_use_specific_time = self.use_specific_time_check_box.isChecked()
            if self.countdown_slide.countdown_use_specific_time:
                self.countdown_slide.countdown_specific_time = self.countdown_time_time_edit.time().toPyTime()

        self.countdown_slide.interval_large = self.interval_large_combo_box.currentIndex()
        self.countdown_slide.interval_small = self.interval_small_combo_box.currentIndex()
        self.countdown_slide.finish_action = self.finish_action_combo_box.currentIndex()

        self.countdown_slide.theme_name = self.theme_combo_box.currentText()
        self.media_item.auto_select_id = self.countdown_slide.id
        success = self.manager.save_object(self.countdown_slide)
        return success

    def on_up_button_clicked(self):
        """
        Move a slide up in the list when the "Up" button is clicked.

        selected_row = self.slide_list_view.currentRow()
        if selected_row != 0:
            qw = self.slide_list_view.takeItem(selected_row)
            self.slide_list_view.insertItem(selected_row - 1, qw)
            self.slide_list_view.setCurrentRow(selected_row - 1)
         """

    def on_down_button_clicked(self):
        """
        Move a slide down in the list when the "Down" button is clicked.

        selected_row = self.slide_list_view.currentRow()
        # zero base arrays
        if selected_row != self.slide_list_view.count() - 1:
            qw = self.slide_list_view.takeItem(selected_row)
            self.slide_list_view.insertItem(selected_row + 1, qw)
            self.slide_list_view.setCurrentRow(selected_row + 1)
        """

    def on_add_button_clicked(self):
        """
        Add a new blank slide.

        self.edit_slide_form.set_text('')
        if self.edit_slide_form.exec_():
            self.slide_list_view.addItems(self.edit_slide_form.get_text())
        """

    def on_edit_button_clicked(self):
        """
        Edit the currently selected slide.

        self.edit_slide_form.set_text(self.slide_list_view.currentItem().text())
        if self.edit_slide_form.exec_():
            self.update_slide_list(self.edit_slide_form.get_text())
         """

    def on_edit_all_button_clicked(self):
        """
        Edits all slides.

        slide_text = ''
        for row in range(self.slide_list_view.count()):
            item = self.slide_list_view.item(row)
            slide_text += item.text()
            if row != self.slide_list_view.count() - 1:
                slide_text += '\n[===]\n'
        self.edit_slide_form.set_text(slide_text)
        if self.edit_slide_form.exec_():
            self.update_slide_list(self.edit_slide_form.get_text(), True)
        """

    def on_preview_button_clicked(self):
        """
        Save the countdown item and preview it.

        log.debug('onPreview')
        if self.save_countdown():
            Registry().execute('countdown_preview')
        """
    def update_slide_list(self, slides, edit_all=False):
        """
        Updates the slide list after editing slides.

        :param slides: A list of all slides which have been edited.
        :param edit_all:  Indicates if all slides or only one slide has been edited.

        if edit_all:
            self.slide_list_view.clear()
            self.slide_list_view.addItems(slides)
        else:
            old_row = self.slide_list_view.currentRow()
            # Create a list with all (old/unedited) slides.
            old_slides = [self.slide_list_view.item(row).text() for row in range(self.slide_list_view.count())]
            self.slide_list_view.clear()
            old_slides.pop(old_row)
            # Insert all slides to make the old_slides list complete.
            for slide in slides:
                old_slides.insert(old_row, slide)
            self.slide_list_view.addItems(old_slides)
        self.slide_list_view.repaint()
        """

    def on_delete_button_clicked(self):
        """
        Removes the current row from the list.

        self.slide_list_view.takeItem(self.slide_list_view.currentRow())
        self.on_current_row_changed(self.slide_list_view.currentRow())
        """

    def on_current_row_changed(self, row):
        """
        Called when the *slide_list_view*'s current row has been changed. This
        enables or disables buttons which require an slide to act on.

        :param row: The row (int). If there is no current row, the value is -1.

        if row == -1:
            self.delete_button.setEnabled(False)
            self.edit_button.setEnabled(False)
            self.up_button.setEnabled(False)
            self.down_button.setEnabled(False)
        else:
            self.delete_button.setEnabled(True)
            self.edit_button.setEnabled(True)
            # Decide if the up/down buttons should be enabled or not.
            self.down_button.setEnabled(self.slide_list_view.count() - 1 != row)
            self.up_button.setEnabled(row != 0)
        """

    def _validate(self):
        """
        Checks whether a countdown is valid or not.
        """
        # We must have a title.
        if not self.title_edit.displayText():
            self.title_edit.setFocus()
            critical_error_message_box(message=translate('CountdownPlugin.EditCountdownForm', 'You need to type in a title.'))
            return False
        # We must have at least one slide.
        # if self.slide_list_view.count() == 0:
        #    critical_error_message_box(message=translate('CountdownPlugin.EditCountdownForm',
        #                                                 'You need to add at least one slide.'))
        #    return False
        return True
