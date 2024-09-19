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
The MIDI plugin in OpenLP provides integration with MIDI devices and allows users to configure MIDI events to interact
with OpenLP functions.

Forms in OpenLP's MIDI plugin follow a two-class approach similar to other plugins:

1. The **Dialog** class, named `Ui_<name>Dialog`, which holds the graphical elements. This is a version adapted from the
output of the `pyuic5` tool, typically using single quotes and OpenLP's `translate()` function for translatable strings.

2. The **Form** class, named `<name>Form`, which contains the functionality. It is instantiated and used in the main
application. This class inherits from both a Qt widget (often `QtWidgets.QDialog`) and the corresponding Ui class,
enabling access to GUI elements via `self.object`.

For example::

    class MidiSettingsForm(QtWidgets.QDialog, Ui_MidiSettingsDialog):

        def __init__(self, parent=None):
            super(MidiSettingsForm, self).__init__(parent)
            self.setup_ui(self)

This dual inheritance pattern allows for a separation of GUI and logic, and also facilitates future changes to the GUI
from the .ui files, if required.
"""
