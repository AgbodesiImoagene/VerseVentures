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
"""
The :mod:`~openlp.plugins.countdown.lib.countdowntab` module contains the settings tab
for the Countdown Slides plugin, which is inserted into the configuration dialog.
"""

from PyQt5 import QtCore, QtWidgets
from openlp.core.common.i18n import translate


from openlp.core.common.settings import Settings
from openlp.core.lib.settingstab import SettingsTab


class CountdownTab(SettingsTab):
    """
    CountdownTab is the Countdown settings tab in the settings dialog.
    """
    def __init__(self, parent, title, visible_title, icon_path):
        super(CountdownTab, self).__init__(parent, title, visible_title, icon_path)

    def setupUi(self):
        self.setObjectName('CountdownTab')
        super(CountdownTab, self).setupUi()
        self.countdown_mode_group_box = QtWidgets.QGroupBox(self.left_column)
        self.countdown_mode_group_box.setObjectName('countdown_mode_group_box')
        self.countdown_mode_layout = QtWidgets.QFormLayout(self.countdown_mode_group_box)
        self.countdown_mode_layout.setObjectName('countdown_mode_layout')
        self.show_countdown_event_name_checkbox = QtWidgets.QCheckBox(self.countdown_mode_group_box)
        self.show_countdown_event_name_checkbox.setObjectName('show_countdown_event_name_checkbox')
        self.show_countdown_legend_checkbox = QtWidgets.QCheckBox(self.countdown_mode_group_box)
        self.show_countdown_legend_checkbox.setObjectName('show_countdown_legend_checkbox')
        self.countdown_mode_layout.addRow(self.show_countdown_event_name_checkbox)
        self.countdown_mode_layout.addRow(self.show_countdown_legend_checkbox)
        self.left_layout.addWidget(self.countdown_mode_group_box)
        self.left_layout.addStretch()
        self.right_layout.addStretch()

        self.show_countdown_event_name_checkbox.stateChanged.connect(self.on_show_countdown_event_name_check_box_changed)
        self.show_countdown_legend_checkbox.stateChanged.connect(self.on_show_countdown_legend_check_box_changed)

    def retranslateUi(self):
        self.countdown_mode_group_box.setTitle(translate('CountdownPlugin.CountdownTab', 'Countdown Options'))

        self.show_countdown_event_name_checkbox.setText(translate('CountdownPlugin.CountdownTab',
                                                                  'Show Countdown event name'))

        self.show_countdown_legend_checkbox.setText(translate('CountdownPlugin.CountdownTab',
                                                              'Show a legend with the countdon (H:M:S)'))

    def on_show_countdown_event_name_check_box_changed(self, check_state):
        """
        Toggle the setting for displaying the countdown event name.

        :param check_state: The current check box state
        """
        self.display_event_name = False
        # we have a set value convert to True/False
        if check_state == QtCore.Qt.Checked:
            self.display_event_name = True

    def on_show_countdown_legend_check_box_changed(self, check_state):
        """
        Toggle the setting for displaying the countdown legend

        :param check_state: The current check box state
        """
        self.display_legend = False
        # we have a set value convert to True/False
        if check_state == QtCore.Qt.Checked:
            self.display_legend = True

    def load(self):
        """

        Load the settings into the dialog
        """
        settings = Settings()
        settings.beginGroup(self.settings_section)
        self.display_event_name = settings.value('display event name')
        self.display_legend = settings.value('display legend')

        self.show_countdown_event_name_checkbox.setChecked(self.display_event_name)
        self.show_countdown_legend_checkbox.setChecked(self.display_legend)
        settings.endGroup()

    def save(self):
        """
        Save the Dialog settings
        """
        settings = Settings()
        settings.beginGroup(self.settings_section)
        settings.setValue('display event name', self.display_event_name)
        settings.setValue('display legend', self.display_legend)

        settings.endGroup()
        if self.tab_visited:
            self.settings_form.register_post_process('countdown_config_updated')
        self.tab_visited = False
