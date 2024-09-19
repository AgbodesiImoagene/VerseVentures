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
from functools import wraps
from PyQt5 import QtCore
from openlp.core.common.mixins import RegistryProperties
from openlp.core.ui.media import is_looping_playback
from openlp.plugins.midi.lib.types_definitions.midi_event_action_map import ActionType


def error_handling_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"EventStateManager | ERROR: An error occurred in {func.__name__}: {e}")
            # Handle the error or re-raise
            # raise
            return None

    return wrapper


def dict_diff(dict1, dict2):
    diffs = {}

    def compare(d1, d2, path=''):
        for key in d1.keys() | d2.keys():
            if isinstance(d1.get(key), dict) and isinstance(d2.get(key), dict):
                compare(d1[key], d2[key], path + key + '.')
            elif d1.get(key) != d2.get(key):
                full_key = path + key
                diffs[full_key] = {'from': d1.get(key), 'to': d2.get(key)}

    compare(dict1, dict2)
    return diffs


class EventStateManager(QtCore.QObject, RegistryProperties):
    """
    Accessed by midi controllers to get status type information from the application
    """
    # Listener callback
    doEventCallbackSignal = QtCore.pyqtSignal([str, int])
    # Triggers the Transmission callback
    poller_changed = QtCore.pyqtSignal()

    def __init__(self):
        """
        Constructor for the midi controller poll builder class.
        """
        super(EventStateManager, self).__init__()
        self.event_callback_mapping = None
        self._prev_state = None
        self._state = None
        self._variable_control_velocity_offset = 0
        self._play_action_type = ActionType.TRIGGER
        self.add_receive_event_callback_mappings()

        self._triggered_ = {event: False for event in
                            ['event_item_next', 'event_item_previous', 'event_slide_next', 'event_slide_previous']}

        # Connect signals to respective slots
        self.doEventCallbackSignal.connect(self._do_event_callback)

        # TODO: i might have to use " live = Registry().get('live_controller') "

    def set_variable_control_offset(self, offset: int = 0):
        self._variable_control_velocity_offset = offset

    def set_play_action_type(self, action_type: bool):
        if action_type:
            self._play_action_type = ActionType.TOGGLE
        else:
            self._play_action_type = ActionType.TRIGGER

    # ---------------------------------- Used for event reception ----------------------------------
    def add_receive_event_callback_mappings(self):
        # Mapping of event names to their callbacks
        self.event_callback_mapping = {
            # Callback Group 1: Screen-related actions
            'event_screen_show': self.event_screen_show_cb,
            'event_screen_theme': self.event_screen_theme_cb,
            'event_screen_blank': self.event_screen_blank_cb,
            'event_screen_desktop': self.event_screen_desktop_cb,
            'event_clear_live': self.event_clear_live_cb,

            # Callback Group 2: Video item actions
            'event_video_play': self.event_video_play_cb,
            'event_video_pause': self.event_video_pause_cb,
            'event_video_stop': self.event_video_stop_cb,
            'event_video_loop': self.event_video_loop_cb,
            'event_video_seek': self.event_video_seek_cb,
            'event_video_volume': self.event_video_volume_cb,

            # Callback Group 3: General item actions
            'event_item_goto': self.event_item_goto_cb,
            'event_item_next': self.event_item_next_cb,
            'event_item_previous': self.event_item_previous_cb,

            # Callback Group 4: Slide/Song-specific actions
            'event_slide_goto': self.event_slide_goto_cb,
            'event_slide_next': self.event_slide_next_cb,
            'event_slide_previous': self.event_slide_previous_cb,

            # Callback Group 5: Song-specific transpose actions
            'event_transpose_up': self.event_transpose_up_cb,
            'event_transpose_down': self.event_transpose_down_cb
        }

    def _do_event_callback(self, event_key, value=None):

        # Locate the correct event and call the event
        callback = self.event_callback_mapping.get(event_key)
        if callback:
            callback(value)
        else:
            print(f"EventStateManager | No callback found for event key: {event_key}")
            # Handle the case where no callback is found for the gi

    # ---------------------------------- Used for event transmission ----------------------------------
    def get_openlp_state(self):
        LC = self.live_controller
        MC = self.media_controller

        state = {
            # Other state changes
            'counter': LC.slide_count if LC.slide_count else 0,
            'service_id': self.service_manager.service_id,
            'chordNotation': self.settings.value('songs/chord notation'),

            # TODO: those sections need to be finished!
            # TODO: It would be a good to have a test to check if all states are matching
            # 'twelve': self.settings.value('api/twelve hour'),
            # 'isSecure': self.settings.value('api/authentication enabled'),

            # MIDI event action Group 1: Screen-related actions
            'event_screen_show': LC.show_screen.isChecked(),
            'event_screen_theme': LC.theme_screen.isChecked(),
            'event_screen_blank': LC.blank_screen.isChecked(),
            'event_screen_desktop': LC.desktop_screen.isChecked(),
            'event_clear_live': True if LC.service_item else False,

            # MIDI event action Group 2: Video item actions
            'event_video_play': LC.media_info.is_playing,
            'event_video_pause': not LC.media_info.is_playing and LC.controller_type in MC.current_media_players,
            'event_video_stop': LC.controller_type not in MC.current_media_players,
            'event_video_loop': self.is_video_loop(),
            'event_video_seek': self.get_current_video_position(),
            'event_video_volume': self.get_audio_level(),

            # MIDI event action Group 3: General item actions
            'event_item_goto': self.get_event_item_goto(),
            'event_item_next': self._triggered_['event_item_next'],
            'event_item_previous': self._triggered_['event_item_previous'],

            # MIDI event action Group 4: Slide/Song-specific actions
            'event_slide_goto': self.live_controller.selected_row + self._variable_control_velocity_offset or 0,
            'event_slide_next': self._triggered_['event_slide_next'],
            'event_slide_previous': self._triggered_['event_slide_previous'],

            # Callback Group 5: Song-specific transpose actions
            # TODO: maybe we need a goto for the transpose
            'event_transpose_up': None,  # TODO: Needs an implementation
            'event_transpose_reset': None,  # TODO: Needs an implementation
            'event_transpose_down': None,  # TODO: Needs an implementation
        }

        # Reset Triggers
        self._triggered_ = {event: False for event in
                            ['event_item_next', 'event_item_previous', 'event_slide_next', 'event_slide_previous']}

        return state

    def get_state_diff(self):  # The internal state variable will not be updated
        A = self._prev_state.copy()
        B = self._state.copy()
        diff = dict_diff(A, B)
        return diff

    def hook_signals(self):
        self.live_controller.slidecontroller_changed.connect(self.on_signal_received_for_state_change)
        self.service_manager.servicemanager_changed.connect(self.on_signal_received_for_state_change)
        self.media_controller.vlc_live_media_tick.connect(self.on_video_position_change)

    def unhook_signals(self):
        try:
            self.live_controller.slidecontroller_changed.disconnect(self.on_signal_received_for_state_change)
            self.service_manager.servicemanager_changed.disconnect(self.on_signal_received_for_state_change)
            self.media_controller.vlc_live_media_tick.disconnect(self.on_video_position_change)
        except Exception:
            pass

    def post_initialization_get_state(self):
        if self._prev_state is None:
            self._state = self.get_openlp_state()

    @QtCore.pyqtSlot(list)
    @QtCore.pyqtSlot(str)
    @QtCore.pyqtSlot()
    def on_signal_received_for_state_change(self):
        self._prev_state = self._state.copy()
        self._state = self.get_openlp_state()
        self.poller_changed.emit()

    def on_video_position_change(self):
        if (self.get_current_video_position() != self._state['event_video_seek'] or
                self.get_audio_level() != self._state['event_video_volume']):
            # Call a signal only when there is actual change
            self.on_signal_received_for_state_change()

    # ================================ List of state check callbacks ================================

    def get_current_live_item_type(self):  # TODO: will that actually be used or should it be removed
        """
        Get the type of the currently live item.
        """
        live_item = self.live_controller.service_item
        return type(live_item).__name__ if live_item else None

    @error_handling_decorator
    def is_video_loop(self):
        if self.live_controller.service_item and self.live_controller.service_item.name == 'media':
            return is_looping_playback(self.live_controller)

    @error_handling_decorator
    def get_current_video_position(self):
        """
        Get the current playback position in milliseconds.

        :return: Current playback position in milliseconds, or -1 if no media is loaded.
        """
        if self.live_controller.service_item and self.live_controller.service_item.name == 'media':
            # if self.live_controller.vlc_media_player.is_playing():
            L = self.live_controller.vlc_media_player.get_length()
            if L < 0:
                return 0
            t = self.live_controller.vlc_media_player.get_time()
            val = 127 * (t / L)
            val = round(val)
            return val
        else:
            return 0

    def get_audio_level(self):
        if self.live_controller.service_item and self.live_controller.service_item.name == 'media':
            volume = self.live_controller.vlc_media_player.audio_get_volume()
            return volume if volume >= 0 else 0
        else:
            return 0

    @error_handling_decorator
    def get_event_item_goto(self):
        live_item = self.live_controller.service_item
        if live_item:
            for index, item in enumerate(self.service_manager.service_items):
                if item['service_item'] == live_item:
                    return index + self._variable_control_velocity_offset  # Position of live item in the service
        return 0

    # ================================ List of event action callbacks ================================

    # ---------------------- # Callback Group 1: Screen-related actions ----------------------
    @error_handling_decorator
    def event_screen_show_cb(self, vel):
        self.live_controller.slidecontroller_toggle_display.emit('show')

    @error_handling_decorator
    def event_screen_theme_cb(self, vel):
        if vel > 0:
            self.live_controller.slidecontroller_toggle_display.emit('theme')
        else:
            self.live_controller.slidecontroller_toggle_display.emit('show')

    @error_handling_decorator
    def event_screen_desktop_cb(self, vel):
        if vel > 0:
            self.live_controller.slidecontroller_toggle_display.emit('desktop')
        else:
            self.live_controller.slidecontroller_toggle_display.emit('show')

    @error_handling_decorator
    def event_screen_blank_cb(self, vel):
        # if self.live_controller.service_item and self.live_controller.service_item.name == 'media':
        if vel > 0:
            self.live_controller.slidecontroller_toggle_display.emit('blank')
        else:
            self.live_controller.slidecontroller_toggle_display.emit('show')

    @error_handling_decorator
    def event_clear_live_cb(self, vel):
        self.event_video_stop_cb(vel)
        self.live_controller.slidecontroller_live_clear.emit()

    # ---------------------- # Callback Group 2: Video item actions ----------------------
    @error_handling_decorator
    def event_video_play_cb(self, vel):
        # This is toggle mode for the play button
        if self.live_controller.service_item and self.live_controller.service_item.name == 'media':
            if self._play_action_type == ActionType.TOGGLE and vel == 0:
                self.live_controller.mediacontroller_live_pause.emit()
            else:
                self.live_controller.mediacontroller_live_play.emit()
                # TODO : consider resetting when not showing from MC.media_reset

    @error_handling_decorator
    def event_video_pause_cb(self, vel):
        if self.live_controller.service_item and self.live_controller.service_item.name == 'media':
            self.live_controller.mediacontroller_live_pause.emit()

    @error_handling_decorator
    def event_video_stop_cb(self, vel):
        if self.live_controller.service_item and self.live_controller.service_item.name == 'media':
            self.live_controller.mediacontroller_live_stop.emit()

    @error_handling_decorator
    def event_video_loop_cb(self, vel):
        if self.live_controller.service_item and self.live_controller.service_item.name == 'media':
            self.media_controller.media_loop(self.live_controller)

    @error_handling_decorator
    def event_video_seek_cb(self, vel):
        if self.live_controller.service_item and self.live_controller.service_item.name == 'media':
            p = (vel - self._variable_control_velocity_offset) / (127 - self._variable_control_velocity_offset)
            L = self.live_controller.vlc_media_player.get_length()
            time = round(p * L, None)
            print(f" => P={p * 100}%  L={L} time={time}")
            self.media_controller.media_seek(self.live_controller, time)

    @error_handling_decorator
    def event_video_volume_cb(self, vel):
        if self.live_controller.service_item and self.live_controller.service_item.name == 'media':
            self.media_controller.media_volume(self.live_controller, vel)

    # ---------------------- # MIDI event action Group 3: General item actions ----------------------
    @error_handling_decorator
    def event_item_previous_cb(self, vel):
        self.service_manager.servicemanager_previous_item.emit()

    @error_handling_decorator
    def event_item_next_cb(self, vel):
        self.service_manager.servicemanager_next_item.emit()

    @error_handling_decorator
    def event_item_goto_cb(self, vel):
        vel = vel - self._variable_control_velocity_offset
        self.service_manager.servicemanager_set_item.emit(vel)

    # ---------------------- # MIDI event action Group 4: Slide/Song-specific actions ----------------------
    @error_handling_decorator
    def event_slide_previous_cb(self, vel):
        self.live_controller.slidecontroller_live_previous.emit()

    @error_handling_decorator
    def event_slide_next_cb(self, vel):
        self.live_controller.slidecontroller_live_next.emit()

    @error_handling_decorator
    def event_slide_goto_cb(self, vel):
        vel = vel - self._variable_control_velocity_offset
        self.live_controller.slidecontroller_live_set.emit([vel])

    # ---------------------- # Callback Group 5: Song-specific transpose actions ----------------------
    @error_handling_decorator
    def event_transpose_up_cb(self, vel):
        pass

    @error_handling_decorator
    def event_transpose_reset_cb(self, vel):
        pass

    @error_handling_decorator
    def event_transpose_down_cb(self, vel):
        pass
