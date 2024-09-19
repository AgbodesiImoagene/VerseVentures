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
The :mod:`midi` module provides the MIDI plugin.
The MIDI plugin offers the capability for duplex MIDI control and interaction within OpenLP.
"""

# NOTE: Technically it would be good to have a separate class to act as a profile template.
#       An instance of this template could be unpacked in the ORM, or alternatively, the template could be created
#       (unpacked from the ORM). This will provide validation methods and consistency.
#       However, at this point it is not necessary. The midi-action-event mappings are always utilized dynamically.
#       The full profile is only fully utilized by the configuration menu tab. When it exports it's state
#       it packs it into a dictionary(not as important) and the midi-action-event into a SimpleNamespace to maintain
#       consistency when the set property is called. Therefore, this is sufficient.
#       All other methods only read the full profile and get it in the form of a dictionary,
#       thus consistency is maintained. If there are more ways to import a profile then a template calls should be used.
#       In all cases the database manager will ensure the type checking of the input profile state or individual fields.
