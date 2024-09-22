# -*- coding: utf-8 -*-
# vim: autoindent shiftwidth=4 expandtab textwidth=120 tabstop=4 softtabstop=4

###############################################################################
# OpenLP - Open Source Lyrics Projection                                      #
# --------------------------------------------------------------------------- #
# Copyright (c) 2008-2014 Raoul Snyman                                        #
# ountdown copyright (c) 2008-2014 Tim Bentley, Gerald Britton, Jonathan      #
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

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog
from sqlalchemy.sql import or_, func, and_

from openlp.core.common.registry import Registry
from openlp.core.common.settings import Settings
from openlp.core.common.i18n import UiStrings, translate
from openlp.core.lib.mediamanageritem import MediaManagerItem
from openlp.core.lib.serviceitem import ItemCapabilities
from openlp.core.lib.plugin import PluginStatus
from openlp.core.lib import ServiceItemContext, check_item_selected
from openlp.plugins.countdown.forms.editcountdownform import EditCountdownForm
from openlp.plugins.countdown.lib.db import CountdownSlide

log = logging.getLogger(__name__)


class CountdownSearch(object):
    """
    An enumeration for countdown search methods.
    """
    Titles = 1
    Themes = 2


class CountdownMediaItem(MediaManagerItem):
    """
    This is the countdown media manager item for Countdown Slides.
    """
    log.info('Countdown Media Item loaded')

    def __init__(self, parent, plugin):
        self.icon_path = 'countdown/countdown'
        super(CountdownMediaItem, self).__init__(parent, plugin)

    def setup_item(self):
        """
        Do some additional setup.
        """
        self.edit_countdown_form = EditCountdownForm(self, self.main_window, self.plugin.db_manager)
        self.single_service_item = False
        self.quick_preview_allowed = True
        self.has_search = True
        # Holds information about whether the edit is remotely triggered and
        # which Countdown is required.
        self.remote_countdown = -1

    def add_end_header_bar(self):
        """
        Add the Countdown End Head bar and register events and functions
        """
        self.toolbar.addSeparator()
        self.add_search_to_toolbar()
        # Signals and slots
        QtCore.QObject.connect(self.search_text_edit, QtCore.SIGNAL('cleared()'), self.on_clear_text_button_click)
        QtCore.QObject.connect(self.search_text_edit, QtCore.SIGNAL('searchTypeChanged(int)'),
                               self.on_search_text_button_clicked)
        Registry().register_function('countdown_load_list', self.load_list)
        Registry().register_function('countdown_preview', self.on_preview_click)
        # Registry().register_function('countdown_create_from_service', self.create_from_service_item)

    def config_update(self):
        """
        Config has been updated so reload values
        """
        log.debug('Config loaded')

    def retranslateUi(self):
        """


        """
        self.search_text_label.setText('%s:' % UiStrings().Search)
        self.search_text_button.setText(UiStrings().Search)

    def initialise(self):
        """
        Initialise the UI so it can provide Searches
        """
        self.search_text_edit.set_search_types(
            [(CountdownSearch.Titles, ':/songs/song_search_title.png', translate('SongsPlugin.MediaItem', 'Titles'),
              translate('SongsPlugin.MediaItem', 'Search Titles...')),
             (CountdownSearch.Themes, ':/slides/slide_theme.png', UiStrings().Themes, UiStrings().SearchThemes)])
        self.search_text_edit.set_current_search_type(Settings().value('%s/last search type' % self.settings_section))
        self.load_list(self.plugin.db_manager.get_all_objects(CountdownSlide, order_by_ref=CountdownSlide.title))
        self.config_update()

    def load_list(self, countdown_slides, target_group=None):
        # Sort out what countdown we want to select after loading the list.
        """

        :param countdown_slides:
        :param target_group:
        """
        self.save_auto_select_id()
        self.list_view.clear()
        countdown_slides.sort()
        for countdown_slide in countdown_slides:
            countdown_name = QtGui.QListWidgetItem(countdown_slide.title)
            countdown_name.setData(QtCore.Qt.UserRole, countdown_slide.id)
            self.list_view.addItem(countdown_name)
            # Auto-select the countdown.
            if countdown_slide.id == self.auto_select_id:
                self.list_view.setCurrentItem(countdown_name)
        self.auto_select_id = -1
        # Called to redisplay the countdown list screen edith from a search
        # or from the exit of the Countdown edit dialog. If remote editing is
        # active trigger it and clean up so it will not update again.

    def on_new_click(self):
        """
        Handle the New item event
        """
        self.edit_countdown_form.load_countdown(0)
        self.edit_countdown_form.exec_()
        self.on_clear_text_button_click()
        self.on_selection_change()

    def on_remote_edit(self, countdown_id, preview=False):
        """
        Called by ServiceManager or SlideController by event passing the countdown Id in the payload along with an
        indicator to say which type of display is required.

        :param countdown_id: The id of the item to be edited
        :param preview: Do we need to update the Preview after the edit. (Default False)
        """
        countdown_id = int(countdown_id)
        valid = self.plugin.db_manager.get_object(CountdownSlide, countdown_id)
        if valid:
            self.edit_countdown_form.load_countdown(countdown_id, preview)
            if self.edit_countdown_form.exec_() == QDialog.Accepted:
                self.remote_triggered = True
                self.remote_countdown = countdown_id
                self.auto_select_id = -1
                self.on_search_text_button_clicked()
                item = self.build_service_item(remote=True)
                self.remote_triggered = None
                self.remote_countdown = 1
                if item:
                    return item
        return None

    def on_edit_click(self):
        """
        Edit a countdown item
        """
        if check_item_selected(self.list_view, UiStrings().SelectEdit):
            item = self.list_view.currentItem()
            item_id = item.data(QtCore.Qt.UserRole)
            self.edit_countdown_form.load_countdown(item_id, False)
            self.edit_countdown_form.exec_()
            self.auto_select_id = -1
            self.on_search_text_button_clicked()

    def on_delete_click(self):
        """
        Remove a countdown item from the list and database
        """
        if check_item_selected(self.list_view, UiStrings().SelectDelete):
            items = self.list_view.selectedIndexes()
            if QtWidgets.QMessageBox.question(self, UiStrings().ConfirmDelete,
                                              translate('CountdownPlugin.MediaItem',
                                                        'Are you sure you want to delete the %n selected countdown slide(s)?',
                                                        '', QtCore.QCoreApplication.CodecForTr, len(items)),
                                              QtWidgets.QMessageBox.StandardButtons(
                                              QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No),
                                              QtWidgets.QMessageBox.Yes) == QtWidgets.QMessageBox.No:
                return
            row_list = [item.row() for item in self.list_view.selectedIndexes()]
            row_list.sort(reverse=True)
            id_list = [(item.data(QtCore.Qt.UserRole)) for item in self.list_view.selectedIndexes()]
            for id in id_list:
                self.plugin.db_manager.delete_object(CountdownSlide, id)
            self.on_search_text_button_clicked()

    def on_focus(self):
        """
        Set the focus
        """
        self.search_text_edit.setFocus()

    def generate_slide_data(self, service_item, item=None, xml_version=False,
                            remote=False, context=ServiceItemContext.Service):
        """
        Generate the slide data. Needs to be implemented by the plugin.
        :param service_item: To be updated
        :param item: The countdown database item to be used
        :param xml_version: No used
        :param remote: Is this triggered by the Preview Controller or Service Manager.
        :param context: Why is this item required to be build (Default Service).
        """
        item_id = self._get_id_of_item_to_generate(item, self.remote_countdown)
        service_item.add_capability(ItemCapabilities.CanEdit)
        service_item.add_capability(ItemCapabilities.CanPreview)
        service_item.add_capability(ItemCapabilities.CanLoop)
        service_item.add_capability(ItemCapabilities.CanSoftBreak)
        service_item.add_capability(ItemCapabilities.OnLoadUpdate)
        service_item.add_capability(ItemCapabilities.CanAutoStartForLive)
        service_item.add_capability(ItemCapabilities.CanEditTitle)
        countdown_slide = self.plugin.db_manager.get_object(CountdownSlide, item_id)
        title = countdown_slide.title
        service_item.edit_id = item_id
        theme = countdown_slide.theme_name
        if theme:
            service_item.theme = theme
        service_item.title = "Countdown until " + title
        service_item.processor = 'webkit'
        service_item.add_from_text("Countdown until " + title)
        return True

    def on_search_text_button_clicked(self):
        """
        Search the plugin database
        """
        # Save the current search type to the configuration.
        Settings().setValue('%s/last search type' % self.settings_section, self.search_text_edit.current_search_type())
        # Reload the list considering the new search type.
        search_type = self.search_text_edit.current_search_type()
        search_keywords = '%' + self.whitespace.sub(' ', self.search_text_edit.displayText()) + '%'
        if search_type == CountdownSearch.Titles:
            log.debug('Titles Search')
            search_results = self.plugin.db_manager.get_all_objects(CountdownSlide,
                                                                    CountdownSlide.title.like(search_keywords),
                                                                    order_by_ref=CountdownSlide.title)
            self.load_list(search_results)
        elif search_type == CountdownSearch.Themes:
            log.debug('Theme Search')
            search_results = self.plugin.db_manager.get_all_objects(CountdownSlide,
                                                                    CountdownSlide.theme_name.like(search_keywords),
                                                                    order_by_ref=CountdownSlide.title)
            self.load_list(search_results)
        self.check_search_result()

    def on_search_text_edit_changed(self, text):
        """
        If search as type enabled invoke the search on each key press. If the Title is being searched do not start until
        2 characters have been entered.

        :param text: The search text
        """
        search_length = 2
        if len(text) > search_length:
            self.on_search_text_button_clicked()
        elif not text:
            self.on_clear_text_button_click()

    def service_load(self, item):
        """
        Triggered by a countdown item being loaded by the service manager.

        :param item: The service Item from the service to load found in the database.
        """
        log.debug('service_load')
        if self.plugin.status != PluginStatus.Active:
            return
        countdown = self.plugin.db_manager.get_object_filtered(CountdownSlide, and_(CountdownSlide.title == item.title,
                                                                                    CountdownSlide.theme_name == item.theme,
                                                                                    CountdownSlide.credits ==
                                                                                    item.raw_footer[0][len(item.title) + 1:]))
        if countdown:
            item.edit_id = countdown.id
            return item
        else:
            if self.add_countdown_from_service:
                self.create_from_service_item(item)

    def on_clear_text_button_click(self):
        """
        Clear the search text.
        """
        self.search_text_edit.clear()
        self.on_search_text_button_clicked()

    def search(self, string, show_error):
        """
        Search the database for a given item.

        :param string: The search string
        :param show_error: The error string to be show.
        """
        search = '%' + string.lower() + '%'
        search_results = self.plugin.db_manager.get_all_objects(CountdownSlide,
                                                                or_(func.lower(CountdownSlide.title).like(search),
                                                                    func.lower(CountdownSlide.text).like(search)),
                                                                order_by_ref=CountdownSlide.title)
        return [[countdown.id, countdown.title] for countdown in search_results]
