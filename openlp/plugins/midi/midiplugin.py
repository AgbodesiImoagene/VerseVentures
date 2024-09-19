# -*- coding: utf-8 -*-

##########################################################################
# OpenLP - Open Source Lyrics Projection                                 #
# ---------------------------------------------------------------------- #
# Copyright (c) 2008-2023 OpenLP Developers                              #
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

import logging

from openlp.core.common.i18n import translate
from openlp.core.common.registry import Registry
from openlp.core.lib import build_icon
from openlp.core.lib.plugin import Plugin, StringContent
from openlp.core.state import State
from openlp.core.ui.icons import UiIcons
from openlp.core.common.enum import PluginStatus

from openlp.plugins.midi.forms.midisettingstab import MidiSettingsTab

from openlp.plugins.midi.lib.handlers_managers.midi_service import MidiControlService

logging.basicConfig(level=logging.DEBUG)  # TODO: remove me!
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)  # TODO: remove me!

midi_settings = {
    'midi/status': PluginStatus.Inactive,
    'midi/db type': 'sqlite',
    'midi/db username': '',
    'midi/db password': '',
    'midi/db hostname': '',
    'midi/db database': ''
}
midi_icons = {
    'midi': {'icon': 'mdi.midi'},
    'midi_port': {'icon': 'mdi.midi-port'}
}


class MidiPlugin(Plugin):

    def __init__(self):
        """
        Class __init__ method to Initialize the MIDI plugin.
        """
        log.info('MIDI Plugin loaded')
        print("---------------------- midi ----------------------")  # TODO: for testing/debugging. Remove later
        super(MidiPlugin, self).__init__('midi', settings_tab_class=MidiSettingsTab)

        # Basic Initializations
        self.weight = -6
        UiIcons().load_icons(midi_icons)
        self.icon_path = UiIcons().midi_port
        self.icon = build_icon(self.icon_path)
        self.settings.extend_default_settings(midi_settings)

        # State services
        State().add_service(self.name, self.weight, is_plugin=True, requires='media_live')
        # TODO: we need a provision for requires where requires states multiple modules

        # Pre-conditions
        State().update_pre_conditions(self.name, self.check_pre_conditions())

        Registry().set_flag('midi_service_active', False)  # Set to False so that it will wait
        # Midi event service
        self.midi_event_handler = MidiControlService(self)

    def check_pre_conditions(self):
        """
        Check the plugin can run.
        For now, I'm returning True.
        You may want to add more conditions as needed.
        """
        live_controller_up = Registry().get('live_controller') is not None
        service_manager_up = Registry().get('service_manager') is not None
        media_controller_up = Registry().get('media_controller') is not None
        media_controller_enabled = State().is_module_active('mediacontroller')
        return live_controller_up and service_manager_up and media_controller_up and media_controller_enabled

    def initialise(self):
        """
        Initialise plugin
        """
        log.info('Midi plugin Initialising')
        Registry().set_flag('midi_service_active', True)
        super(MidiPlugin, self).initialise()
        self.midi_event_handler.start()

        # Register midi control service update callbacks in the configuration panel
        # Every time the plugin is activated the callbacks will be overwritten.
        # This is ok, those that are the same will remain the same and new instances will overwrite the old callbacks.

        # Register midi control service update callbacks in the configuration panel
        MidiSettingsTab.add_listener_callback(listener_id="MidiService_close", callback_type="on_cfg_open",
                                              callback=lambda event: self.midi_event_handler.close())
        MidiSettingsTab.add_listener_callback(listener_id="MidiService_start", callback_type="on_cfg_close",
                                              callback=lambda event: self.midi_event_handler.start())

        log.info('MIDI Plugin successfully initialized.')

    def finalise(self):
        """
        Tidy up on exit
        """
        log.info('Midi plugin Finalising')
        super(MidiPlugin, self).finalise()

        self.midi_event_handler.close()

        Registry().set_flag('midi_service_active', False)
        log.info('Midi plugin Finalized')
        return

    @staticmethod
    def about():
        about_text = translate('MidiPlugin', '<strong>MIDI Plugin</strong>'
                                             '<br />The MIDI plugin provides the capability for duplex MIDI control '
                                             'with OpenLP.')
        return about_text

    def set_plugin_text_strings(self):
        """
        Called to define all translatable texts of the plugin.
        """
        # Name PluginList
        self.text_strings[StringContent.Name] = {
            'singular': translate('MidiPlugin', 'MIDI Control', 'name singular'),
            'plural': translate('MidiPlugin', 'MIDI Controls', 'name plural')
        }
        # Name for MediaDockManager, SettingsManager
        self.text_strings[StringContent.VisibleName] = {'title': translate('MidiPlugin', 'MIDI', 'container title')}

    # def exec_settings_dialog(self):
    #     """
    #     Display the MIDI settings dialog.
    #     """
    #     # if not self.midi_settings_dialog:
    #     #     self.midi_settings_dialog = MidiSettingsDialog(self.main_window, self)
    #     # self.midi_settings_dialog.exec()
    #     pass
