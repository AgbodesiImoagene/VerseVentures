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

from PyQt5 import QtGui

from openlp.core.common.i18n import translate
from openlp.core.lib import build_icon
from openlp.core.lib.ui import create_button_box


class Ui_CountdownEditDialog(object):

    def setupUi(self, countdown_edit_dialog):
        # Set up the main Window
        countdown_edit_dialog.setObjectName("countdown_edit_dialog")
        countdown_edit_dialog.setWindowIcon(build_icon(u':/icon/openlp-logo.svg'))
        countdown_edit_dialog.resize(450, 350)
        # Set up the main content layout
        self.dialog_layout = QtGui.QVBoxLayout(countdown_edit_dialog)
        self.dialog_layout.setObjectName('dialog_layout')

        # Set up a layout for the title label and edit box
        self.title_layout = QtGui.QHBoxLayout()
        self.title_layout.setObjectName('title_layout')
        self.title_label = QtGui.QLabel(countdown_edit_dialog)
        self.title_label.setObjectName('title_label')
        self.title_layout.addWidget(self.title_label)
        self.title_edit = QtGui.QLineEdit(countdown_edit_dialog)
        self.title_label.setBuddy(self.title_edit)
        self.title_edit.setObjectName('title_edit')
        self.title_layout.addWidget(self.title_edit)
        self.dialog_layout.addLayout(self.title_layout)

        # set up a layout for the countdown properties
        self.central_layout = QtGui.QFormLayout()
        self.central_layout.setObjectName('central_layout')

        # build label for countdown_type
        self.countdown_type_label = QtGui.QLabel(countdown_edit_dialog)
        self.countdown_type_label.setObjectName("countdown_type_label")
        # build countdown_type_combo_box combo box
        self.countdown_type_combo_box = QtGui.QComboBox(countdown_edit_dialog)
        self.countdown_type_combo_box.setObjectName("countdown_type_combo_box")
        self.countdown_type_combo_box.addItem("")
        self.countdown_type_combo_box.addItem("")
        self.countdown_type_label.setBuddy(self.countdown_type_combo_box)
        self.central_layout.addRow(self.countdown_type_label, self.countdown_type_combo_box)

        # build label for countdown time
        self.countdown_duration_label = QtGui.QLabel(countdown_edit_dialog)
        self.countdown_duration_label.setObjectName("countdown_duration_label")
        # build time edit box for countdown time
        self.countdown_duration_time_edit = QtGui.QTimeEdit(countdown_edit_dialog)
        self.countdown_duration_time_edit.setObjectName("countdown_duration_time_edit")
        # add to layout
        self.countdown_duration_label.setBuddy(self.countdown_duration_time_edit)
        self.central_layout.addRow(self.countdown_duration_label, self.countdown_duration_time_edit)

        # Build label for use_specific_date
        self.use_specific_date_label = QtGui.QLabel(countdown_edit_dialog)
        self.use_specific_date_label.setObjectName("use_specific_date_label")
        # Build Check box for use_specific_date
        self.use_specific_date_check_box = QtGui.QCheckBox(countdown_edit_dialog)
        self.use_specific_date_check_box.setObjectName("use_specific_date_check_box")
        # add use specific date to layout
        self.use_specific_date_label.setBuddy(self.use_specific_date_check_box)
        self.central_layout.addRow(self.use_specific_date_label, self.use_specific_date_check_box)

        # build label for countdown date
        self.countdown_date_label = QtGui.QLabel(countdown_edit_dialog)
        self.countdown_date_label.setObjectName("countdown_date_label")
        # build date edit box for countdown date
        self.countdown_date_date_edit = QtGui.QDateEdit(countdown_edit_dialog)
        self.countdown_date_date_edit.setObjectName("countdown_date_date_edit")

        # add countdown date to layout
        self.countdown_date_label.setBuddy(self.countdown_date_date_edit)
        self.central_layout.addRow(self.countdown_date_label, self.countdown_date_date_edit)

        # build label for use_specific_time_checkbox
        self.use_specific_time_label = QtGui.QLabel(countdown_edit_dialog)
        self.use_specific_time_label.setObjectName("use_specific_time_label")
        # build use_specific_time_checkbox
        self.use_specific_time_check_box = QtGui.QCheckBox(countdown_edit_dialog)
        self.use_specific_time_check_box.setObjectName("use_specific_time_check_box")
        # add to layout
        self.use_specific_time_label.setBuddy(self.use_specific_time_check_box)
        self.central_layout.addRow(self.use_specific_time_label, self.use_specific_time_check_box)

        # build label for countdown time
        self.countdown_time_label = QtGui.QLabel(countdown_edit_dialog)
        self.countdown_time_label.setObjectName("countdown_time_label")
        # build time edit box for countdown time
        self.countdown_time_time_edit = QtGui.QTimeEdit(countdown_edit_dialog)
        self.countdown_time_time_edit.setObjectName("countdown_time_time_edit")
        # add to layout
        self.countdown_time_label.setBuddy(self.countdown_time_time_edit)
        self.central_layout.addRow(self.countdown_time_label, self.countdown_time_time_edit)

        # build label for interval_large
        self.interval_large_label = QtGui.QLabel(countdown_edit_dialog)
        self.interval_large_label.setObjectName("interval_large_label")
        # build interval_large combo box
        self.interval_large_combo_box = QtGui.QComboBox(countdown_edit_dialog)
        self.interval_large_combo_box.setObjectName("interval_large_combo_box")
        self.interval_large_combo_box.addItem("")
        self.interval_large_combo_box.addItem("")
        self.interval_large_combo_box.addItem("")
        self.interval_large_combo_box.addItem("")
        self.interval_large_combo_box.addItem("")
        self.interval_large_combo_box.addItem("")
        self.interval_large_combo_box.addItem("")
        self.interval_large_combo_box.addItem("")
        # add to layout
        self.interval_large_label.setBuddy(self.interval_large_combo_box)
        self.central_layout.addRow(self.interval_large_label, self.interval_large_combo_box)

        # build label for interval small
        self.interval_small_label = QtGui.QLabel(countdown_edit_dialog)
        self.interval_small_label.setObjectName("interval_small_label")
        # build interval_small_combo_box
        self.interval_small_combo_box = QtGui.QComboBox(countdown_edit_dialog)
        self.interval_small_combo_box.setObjectName("interval_small_combo_box")
        self.interval_small_combo_box.addItem("")
        self.interval_small_combo_box.addItem("")
        self.interval_small_combo_box.addItem("")
        self.interval_small_combo_box.addItem("")
        self.interval_small_combo_box.addItem("")
        self.interval_small_combo_box.addItem("")
        self.interval_small_combo_box.addItem("")
        self.interval_small_combo_box.addItem("")
        # add to layout
        self.interval_small_label.setBuddy(self.interval_small_combo_box)
        self.central_layout.addRow(self.interval_small_label, self.interval_small_combo_box)

        # build label for finish_action
        self.finish_action_label = QtGui.QLabel(countdown_edit_dialog)
        self.finish_action_label.setObjectName("finish_action_label")
        # build finish_action_combo_box
        self.finish_action_combo_box = QtGui.QComboBox(countdown_edit_dialog)
        self.finish_action_combo_box.setObjectName("finish_action_combo_box")
        self.finish_action_combo_box.addItem("")
        self.finish_action_combo_box.addItem("")
        self.finish_action_combo_box.addItem("")
        self.finish_action_label.setBuddy(self.finish_action_combo_box)
        self.central_layout.addRow(self.finish_action_label, self.finish_action_combo_box)

        # add the central_layout to the dialog_layout
        self.dialog_layout.addLayout(self.central_layout)
        # create the bottom layout
        self.bottom_form_layout = QtGui.QFormLayout()
        self.bottom_form_layout.setObjectName('bottom_form_layout')
        self.theme_label = QtGui.QLabel(countdown_edit_dialog)
        self.theme_label.setObjectName('theme_label')
        self.theme_combo_box = QtGui.QComboBox(countdown_edit_dialog)
        self.theme_combo_box.setSizeAdjustPolicy(QtGui.QComboBox.AdjustToContents)
        self.theme_combo_box.setObjectName('theme_combo_box')
        self.theme_label.setBuddy(self.theme_combo_box)
        self.bottom_form_layout.addRow(self.theme_label, self.theme_combo_box)

        # add the bottom_layout to the dialog_layout
        self.dialog_layout.addLayout(self.bottom_form_layout)
        # Create the cancel, save button box, and add it to the dialog_layout
        self.button_box = create_button_box(countdown_edit_dialog, 'button_box', ['cancel', 'save'])
        self.dialog_layout.addWidget(self.button_box)
        # retranslate the UI
        self.retranslateUi(countdown_edit_dialog)

    def retranslateUi(self, countdown_edit_dialog):
        countdown_edit_dialog.setWindowTitle(translate("customEditDialog", "Edit Countdown Slides", None))
        self.theme_label.setText(translate("customEditDialog", "Theme:", None))
        self.title_label.setText(translate("customEditDialog", "Title:", None))
        self.countdown_type_label.setText(translate("customEditDialog", "Countdown Type", None))
        self.countdown_type_combo_box.setItemText(0, translate("customEditDialog", "Future Date / Time", None))
        self.countdown_type_combo_box.setItemText(1, translate("customEditDialog", "Specific Duration", None))
        self.countdown_duration_label.setText(translate("customEditDialog", "Countdown Duration", None))
        self.use_specific_date_label.setText(translate("customEditDialog", "Use Specific Date", None))
        self.countdown_date_label.setText(translate("customEditDialog", "Countdown Date", None))
        self.use_specific_time_label.setText(translate("customEditDialog", "Use Specific Time", None))
        self.countdown_time_label.setText(translate("customEditDialog", "Countdown Time", None))
        self.interval_large_label.setText(translate("customEditDialog", "Largest Interval to Display", None))
        self.interval_large_combo_box.setItemText(0, translate("customEditDialog", "Years", None))
        self.interval_large_combo_box.setItemText(1, translate("customEditDialog", "Months", None))
        self.interval_large_combo_box.setItemText(2, translate("customEditDialog", "Weeks", None))
        self.interval_large_combo_box.setItemText(3, translate("customEditDialog", "Days", None))
        self.interval_large_combo_box.setItemText(4, translate("customEditDialog", "Hours", None))
        self.interval_large_combo_box.setItemText(5, translate("customEditDialog", "Minues", None))
        self.interval_large_combo_box.setItemText(6, translate("customEditDialog", "Seconds", None))
        self.interval_large_combo_box.setItemText(7, translate("customEditDialog", "Miliseconds", None))
        self.interval_small_label.setText(translate("customEditDialog", "Smallest Interval to Display", None))
        self.interval_small_combo_box.setItemText(0, translate("customEditDialog", "Years", None))
        self.interval_small_combo_box.setItemText(1, translate("customEditDialog", "Months", None))
        self.interval_small_combo_box.setItemText(2, translate("customEditDialog", "Weeks", None))
        self.interval_small_combo_box.setItemText(3, translate("customEditDialog", "Days", None))
        self.interval_small_combo_box.setItemText(4, translate("customEditDialog", "Hours", None))
        self.interval_small_combo_box.setItemText(5, translate("customEditDialog", "Minues", None))
        self.interval_small_combo_box.setItemText(6, translate("customEditDialog", "Seconds", None))
        self.interval_small_combo_box.setItemText(7, translate("customEditDialog", "Miliseconds", None))
        self.finish_action_label.setText(translate("customEditDialog", "Finish Action", None))
        self.finish_action_combo_box.setItemText(0, translate("customEditDialog", "Blink", None))
        self.finish_action_combo_box.setItemText(1, translate("customEditDialog", "Blank", None))
        self.finish_action_combo_box.setItemText(2, translate("customEditDialog", "Next Slide", None))
