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
#                                                                             #
###############################################################################
# This file contains MIT licensed code                                        #
###############################################################################
"""
The :mod:`~openlp.plugins.countdown.countdownplugin` module contains the Plugin class
for the Countdown plugin.
"""

import logging

from openlp.core.lib.plugin import Plugin, StringContent
from openlp.core.common.i18n import translate
from openlp.core.lib.db import Manager
from openlp.plugins.countdown.lib.countdowntab import CountdownTab
from openlp.plugins.countdown.lib.db import CountdownSlide, init_schema
from openlp.plugins.countdown.lib.mediaitem import CountdownMediaItem,CountdownSearch
from PyQt5 import QtGui, QtCore

log = logging.getLogger(__name__)

__default_settings__ = {
    'countdown/db type': 'sqlite',
    'countdown/last search type': CountdownSearch.Titles,
    'countdown/display event name': True,
    'countdown/display legend': True,
    'shortcuts/listViewCountdownDeleteItem': [QtGui.QKeySequence(QtCore.Qt.Key_Delete)],
    'shortcuts/listViewCountdownPreviewItem': [QtGui.QKeySequence(QtCore.Qt.Key_Enter),
                                               QtGui.QKeySequence(QtCore.Qt.Key_Return)],
    'shortcuts/listViewCountdownLiveItem': [QtGui.QKeySequence(QtCore.Qt.ShiftModifier | QtCore.Qt.Key_Enter),
                                            QtGui.QKeySequence(QtCore.Qt.ShiftModifier | QtCore.Qt.Key_Return)],
    'shortcuts/listViewCountdownServiceItem': [QtGui.QKeySequence(QtCore.Qt.Key_Plus),
                                               QtGui.QKeySequence(QtCore.Qt.Key_Equal)],
}


class CountdownPlugin(Plugin):
    """
    This plugin enables the user to create, edit and display countdown timers.
    All Countdown timers have one slide, which is dynamically updated by JavaScript running in webkit.
    The slide manager will display only one slide for all countdown items 
    with a textual representation of the countdown event
    Examples include countdowns until: worship time, Christmas, Easter.
    Examples do not include countdowns until: judgement day.
    """
    log.info('Countdown Plugin loaded')

    def __init__(self):
        super(CountdownPlugin, self).__init__('countdown', __default_settings__, CountdownMediaItem, CountdownTab)
        self.weight = -1
        self.db_manager = Manager('countdown', init_schema)
        self.icon_path = ':/plugins/plugin_custom.png'
        self.icon = self.icon_path

    @staticmethod
    def about(self):
        about_text = translate('CountdownPlugin', '<strong>Countdown Plugin </strong><br />The countdown  plugin '
                               'provides the ability to display a live countdown to an event')
        return about_text

    def uses_theme(self, theme):
        """
        Called to find out if the countdown plugin is currently using a theme.

        Returns True if the theme is being used, otherwise returns False.
        """
        if self.db_manager.get_all_objects(CountdownSlide, CountdownSlide.theme_name == theme):
            return True
        return False

    def rename_theme(self, old_theme, new_theme):
        """
        Renames a theme the countdown plugin is using making the plugin use the new name.

        :param old_theme: The name of the theme the plugin should stop using.
        :param new_theme: The new name the plugin should now use.
        """
        countdowns_using_theme = self.db_manager.get_all_objects(CountdownSlide, CountdownSlide.theme_name == old_theme)
        for countdown in countdowns_using_theme:
            countdown.theme_name = new_theme
            self.db_manager.save_object(countdown)

    def set_plugin_text_strings(self):
        """
        Called to define all translatable texts of the plugin
        """
        # Name PluginList
        self.text_strings[StringContent.Name] = {
            'singular': translate('CountdownPlugin', 'Countdown Slide', 'name singular'),
            'plural': translate('CountdownPlugin', 'Countdown Slides', 'name plural')
        }
        # Name for MediaDockManager, SettingsManager
        self.text_strings[StringContent.VisibleName] = {
            'title': translate('CountdownPlugin', 'Countdown Slides', 'container title')
        }
        # Middle Header Bar
        tooltips = {
            'load': translate('CountdownPlugin', 'Load a new countdown slide.'),
            'import': translate('CountdownPlugin', 'Import a countdown slide.'),
            'new': translate('CountdownPlugin', 'Add a new countdown slide.'),
            'edit': translate('CountdownPlugin', 'Edit the selected countdown slide.'),
            'delete': translate('CountdownPlugin', 'Delete the selected countdown slide.'),
            'preview': translate('CountdownPlugin', 'Preview the selected countdown slide.'),
            'live': translate('CountdownPlugin', 'Send the selected countdown slide live.'),
            'service': translate('CountdownPlugin', 'Add the selected countdown slide to the service.')
        }
        self.set_plugin_ui_text_strings(tooltips)

    def finalise(self):
        """
        Time to tidy up on exit
        """
        log.info('Countdown Finalising')
        self.db_manager.finalise()
        Plugin.finalise(self)

    def get_display_css(self):
        """
        Add css style sheets to htmlbuilder.
        Currently, countdowns use no custom CSS, but that will probably change, so lets build an empty class
        """
        css = """
        #countdowntext{

        }
        """

        return css

    def get_display_javascript(self):
      try:
        countdownjs_file_path = AppLocation.get_directory(AppLocation.AppDir) / '..' / 'Resources' / 'countdown.js'
        with countdownjs_file_path.open('r') as file_handle:
          content = file_handle.read()
      except (OSError):
          log.exception('Failed to open text file {text}'.format(text=text_file_path))
      return content


    def get_display_html(self):
        """
        Add html code to htmlbuilder.

        """
        log.debug('Countdown get_display_html called')
        HTML = """
            <div id="countdown" class="lyricstable"><h1 id="countdowntext">test</h1></div>
        """
        HTML = ""
        return HTML
