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

default_profile_name = 'Default Profile'

disabled_midi_device = dict()
disabled_midi_device['input'] = "... (Disabled) No input MIDI device selected ..."
disabled_midi_device['output'] = "... (Disabled) No output MIDI device selected ..."

# TODO: replace the values in the ORM
default_midi_device = dict()
default_midi_device['input'] = disabled_midi_device['input']
default_midi_device['input_channel'] = 15
default_midi_device['output'] = disabled_midi_device['output']
default_midi_device['output_channel'] = 16

openlp_midi_device = dict()
openlp_midi_device['gui_label'] = "=== OpenLP MIDI Input ==="
openlp_midi_device['name'] = 'OpenLP-MIDI-Input'

assignment_message = "== Waiting for MIDI input =="
midi_ch_any = "Any"
