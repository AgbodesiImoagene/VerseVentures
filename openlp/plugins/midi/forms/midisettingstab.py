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
    This file contains the GUI for the configuration panel for the MIDI control plugin
"""

from types import SimpleNamespace
# Qt Imports
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QMetaObject, Qt
# Settings tab
from openlp.core.lib.settingstab import SettingsTab
from openlp.core.common.i18n import translate
# Icon imports
from openlp.core.ui.icons import UiIcons
from openlp.core.lib import build_icon
# MIDI related imports
from openlp.plugins.midi.lib.types_definitions.constants import openlp_midi_device, default_midi_device, \
    assignment_message, disabled_midi_device
from openlp.plugins.midi.lib.types_definitions.midi_definitions import MIDI_Def
from openlp.plugins.midi.lib.midi.listener import MidiEventListener
from openlp.plugins.midi.lib.handlers_managers.device_handler import MidiDeviceHandler
from openlp.plugins.midi.lib.handlers_managers.profile_db_manager import MidiProfileManager, \
    get_default_action_midi_mappings_as_dict

ico_sp = " "  # Define space between the text and the icon


def add_icon_to(qt_ui_element, icon_label, position="left"):
    ui_icons = UiIcons()
    icon = None

    # Check if UiIcons has the icon
    if isinstance(icon_label, str) and hasattr(ui_icons, icon_label):
        icon_path = getattr(ui_icons, icon_label)
        icon = build_icon(icon_path)

    # If icon is not built, try to build it from a direct path
    if icon is None or icon.isNull():
        try:
            icon_dict = icon_label
            icon_label = qt_ui_element.text()
            for symbol in ['[', ']', ' ']:
                icon_label = icon_label.replace(symbol, '').lower()
            ui_icons.load_icons({icon_label: icon_dict})
            icon_path = getattr(ui_icons, icon_label)
            icon = build_icon(icon_path)
            # icon = build_icon(icon_label)  # TODO: we could potentially build it too
        except Exception as e:
            raise ValueError(f"Failed to build icon from UiIcons or direct path: {icon_label}") from e

    # Check if the icon is still not valid
    if icon.isNull():
        raise ValueError(f"Invalid icon: {icon_label}")

    # Set the icon on the button
    pixel_size = 24
    qt_ui_element.setIcon(icon)
    qt_ui_element.setIconSize(QtCore.QSize(pixel_size, pixel_size))
    if position == "right":
        qt_type = qt_ui_element.__class__.__name__
        qt_ui_element.setStyleSheet(qt_type + " { qproperty-iconSize: 24px; qproperty-layoutDirection: RightToLeft; }")


event_to_icon_mapping = {
    'event_screen_show': 'live_presentation',
    'event_screen_theme': 'live_theme',
    'event_screen_blank': 'live_black',
    'event_screen_desktop': 'desktop',
    'event_clear_live': 'live_desktop',  # 'delete'
    'event_video_play': 'play',
    'event_video_pause': 'pause',
    'event_video_stop': 'stop',
    'event_video_loop': 'loop',
    'event_video_seek': {'icon': 'mdi6.video-switch-outline'},
    'event_video_volume': 'music',
    'event_item_previous': 'arrow_left',
    'event_item_next': 'arrow_right',
    'event_item_goto': {'icon': 'mdi.cursor-pointer'},
    'event_slide_previous': 'move_up',
    'event_slide_next': 'move_down',
    'event_slide_goto': {'icon': 'mdi.select-place'},
    'event_transpose_up': {'icon': 'mdi6.music-note-plus'},
    'event_transpose_reset': {'icon': 'mdi6.music-note-quarter'},
    'event_transpose_down': {'icon': 'mdi6.music-note-minus'},
    'checkbox_play_is_toggle': {'icon': 'mdi.play-pause'}
}


class MidiSettingsTab(SettingsTab):
    """
    MidiSettingsTab is the MIDI settings tab in the settings dialog.
    """
    listener_callbacks = {
        'on_cfg_open': {},
        'on_cfg_close': {},
        'on_cfg_update': {},
    }

    def __init__(self, parent, title, visible_title, icon_path):
        # The 'midi' identifier should probably be an argument, not hard coded. For now, leaving as is.
        self._current_assignment_ = None
        self._current_assignment_mel = None
        self.profile_group = None
        self.midi_devices_group = None
        self.midi_actions_group = None
        self.db_manager = None  # Placeholder, instantiated in initialise()
        self.previous_ui_action_values = {}  # To store previous values for each action
        super(MidiSettingsTab, self).__init__(parent, 'midi', visible_title, icon_path)

    # ============== UI setup section ==============
    def setup_ui(self):
        """
        Set up the tab's interface specific to Midi Settings.
        """
        self.setObjectName('MidiControlTab')
        super().setup_ui()

        # Group 1: Profile Management Group
        self.profile_group = self._create_profile_group()

        # Group 2: MIDI Devices and Channels Group
        self.midi_devices_group = self._create_midi_devices_group()

        # Group 3: MIDI Action Mapping Group
        self.midi_actions_group = self._create_midi_actions_group()

        # Set the layouts
        self.left_layout.addWidget(self.profile_group)
        self.left_layout.addWidget(self.midi_devices_group)
        self.right_layout.addWidget(self.midi_actions_group)

    def _create_profile_group(self):
        """
        Create and return the Profile Management group box.
        """
        # Initialize the profile_dropdown
        self.profile_dropdown = QtWidgets.QComboBox()
        # Connecting this method to the profile dropdown change signal
        self.profile_dropdown.currentIndexChanged.connect(self._update_profile_callback)

        # Create a QLabel for the "Profile:" label, spanning the entire row width and centered
        self.profile_label = QtWidgets.QLabel("Profile:")
        self.profile_label.setAlignment(Qt.AlignCenter)

        # Initialize the profile management buttons
        self.create_profile_button = QtWidgets.QPushButton("Create" + ico_sp)
        self.create_profile_button.clicked.connect(self._create_profile_callback)
        add_icon_to(self.create_profile_button, "new")

        # Initialize the save button
        self.save_profile_button = QtWidgets.QPushButton("Save" + ico_sp)
        self.save_profile_button.clicked.connect(self._save_profile_callback)
        add_icon_to(self.save_profile_button, "save")

        self.rename_profile_button = QtWidgets.QPushButton("Rename" + ico_sp)
        self.rename_profile_button.clicked.connect(self._rename_profile_callback)
        add_icon_to(self.rename_profile_button, "edit")

        self.delete_profile_button = QtWidgets.QPushButton("Delete" + ico_sp)
        self.delete_profile_button.clicked.connect(self._delete_profile_callback)
        add_icon_to(self.delete_profile_button, "delete")

        # Create the profile group
        profile_group = QtWidgets.QGroupBox("Profile Management", self)
        profile_group_layout = QtWidgets.QGridLayout()

        # Add the buttons to the profile management layout
        self.profile_mgmt_layout = QtWidgets.QHBoxLayout()
        self.profile_mgmt_layout.addWidget(self.create_profile_button)
        self.profile_mgmt_layout.addWidget(self.save_profile_button)
        self.profile_mgmt_layout.addWidget(self.rename_profile_button)
        self.profile_mgmt_layout.addWidget(self.delete_profile_button)

        # Merge the above layout into the main grid layout
        # profile_group_layout.addWidget(self.profile_label, 0, 0)
        profile_group_layout.addWidget(self.profile_dropdown, 0, 1)
        profile_group_layout.addLayout(self.profile_mgmt_layout, 1, 1)

        # Set vertical spacing for the layout
        profile_group_layout.setVerticalSpacing(20)

        profile_group.setLayout(profile_group_layout)
        return profile_group

    def _create_midi_devices_group(self):
        """
        Create and return the MIDI Devices and Channels group box.
        """
        col_span = 4  # Number of columns

        # Initialize the input and output device dropdowns
        self.midi_device_input_dropdown = QtWidgets.QComboBox()
        self.midi_channel_input_dropdown = QtWidgets.QComboBox()

        self.midi_device_output_dropdown = QtWidgets.QComboBox()
        self.midi_channel_output_dropdown = QtWidgets.QComboBox()

        # Create checkbox for MIDI device state reset when connected
        self.reset_midi_state_checkbox = QtWidgets.QCheckBox("Reset MIDI state on connect")
        self.reset_midi_state_checkbox.setChecked(False)

        # Create checkbox for MIDI device state sync with OpenLP
        self.device_sync_checkbox = QtWidgets.QCheckBox("Maintain MIDI device Sync")
        self.device_sync_checkbox.setChecked(False)

        self.play_button_is_toggle_checkbox = QtWidgets.QCheckBox("[Play] is a toggle state action")
        self.play_button_is_toggle_checkbox.setChecked(False)
        # TODO: why is it not displaying well
        add_icon_to(self.play_button_is_toggle_checkbox, 'checkbox_play_is_toggle')

        def _play_button_is_toggle_callback_(value):
            text = self.midi_assignment_buttons['event_video_play'].text()
            if value:
                new_text = text.replace('[Trigger]', '[Toggle]')
            else:
                new_text = text.replace('[Toggle]', '[Trigger]')
            self.midi_assignment_buttons['event_video_play'].setText(new_text)
            self.retranslate_ui()
        self.play_button_is_toggle_checkbox.stateChanged.connect(_play_button_is_toggle_callback_)

        # Create checkbox for deferred_control_mode
        self.deferred_control_mode_checkbox = QtWidgets.QCheckBox("Deferred Control Mode")
        # Initially set the checkbox based on the default value
        # (will be updated later based on actual value from the database)
        self.deferred_control_mode_checkbox.setChecked(False)
        # TODO: is disabled because implementation maybe done or not
        self.deferred_control_mode_checkbox.setDisabled(True)

        # Create a spinner for the control value modulation
        self.control_offset_label = QtWidgets.QLabel('"Go-to" control offset')
        self.control_offset_spinner = QtWidgets.QSpinBox()
        self.control_offset_spinner.setRange(-127, 127)
        self.control_offset_spinner.setValue(0)
        # Add to a sub-layout
        self.control_offset_layout = QtWidgets.QHBoxLayout()  # Create a horizontal layout
        self.control_offset_layout.addWidget(self.control_offset_label)  # Add the label to the layout
        self.control_offset_layout.addWidget(self.control_offset_spinner)  # Add the spin box to the layout

        self.input_devices_label = QtWidgets.QLabel("Input Devices:")
        self.input_devices_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.output_devices_label = QtWidgets.QLabel("Output Devices:")
        self.output_devices_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # Create the MIDI devices group
        midi_devices_group = QtWidgets.QGroupBox("MIDI Devices, Channels and Options", self)
        midi_devices_layout = QtWidgets.QGridLayout()

        visual_option = 2  # TODO: we have to commit to one of the arrangement options here
        if visual_option == 1:
            midi_devices_layout.addWidget(self.input_devices_label, 0, 0)
            midi_devices_layout.addWidget(self.midi_device_input_dropdown, 0, 1)
            midi_devices_layout.addWidget(self.midi_channel_input_dropdown, 0, 3)

            midi_devices_layout.addWidget(self.output_devices_label, 1, 0)
            midi_devices_layout.addWidget(self.midi_device_output_dropdown, 1, 1)
            midi_devices_layout.addWidget(self.midi_channel_output_dropdown, 1, 3)

            # Add the checkbox to the layout # Span the checkbox across all columns
            midi_devices_layout.addWidget(self.reset_midi_state_checkbox, 2, 0, 1, col_span)
            midi_devices_layout.addWidget(self.device_sync_checkbox, 3, 0, 1, col_span)
            midi_devices_layout.addWidget(self.play_button_is_toggle_checkbox, 4, 0, 1, col_span)
            # midi_devices_layout.addWidget(self.deferred_control_mode_checkbox, 2, 2, 1, col_span)
            midi_devices_layout.addLayout(self.control_offset_layout, 5, 2, 1, col_span)

        if visual_option == 2:
            self.input_devices_label.setAlignment(Qt.AlignCenter | Qt.AlignTop)
            self.output_devices_label.setAlignment(Qt.AlignCenter | Qt.AlignTop)

            midi_devices_layout.addWidget(self.input_devices_label, 0, 0)
            midi_devices_layout.addWidget(self.midi_device_input_dropdown, 1, 0)
            midi_devices_layout.addWidget(self.midi_channel_input_dropdown, 1, 1)

            midi_devices_layout.addWidget(self.output_devices_label, 2, 0)
            midi_devices_layout.addWidget(self.midi_device_output_dropdown, 3, 0)
            midi_devices_layout.addWidget(self.midi_channel_output_dropdown, 3, 1)

            # Add the checkbox to the layout # Span the checkbox across all columns
            midi_devices_layout.addWidget(self.reset_midi_state_checkbox, 4, 0)
            midi_devices_layout.addWidget(self.device_sync_checkbox, 5, 0)
            midi_devices_layout.addWidget(self.play_button_is_toggle_checkbox, 6, 0, 1, col_span)
            # midi_devices_layout.addWidget(self.deferred_control_mode_checkbox, 2, 2, 1, col_span)
            midi_devices_layout.addLayout(self.control_offset_layout, 7, 0)

        # Set vertical spacing for the layout
        midi_devices_layout.setVerticalSpacing(20)

        # Add a stretch factor to the last row
        # midi_devices_layout.setRowStretch(5, 1)

        midi_devices_group.setLayout(midi_devices_layout)
        return midi_devices_group

    def _create_midi_actions_group(self):
        """
        Create and return the MIDI Action Mapping group box.
        """
        midi_actions_group = QtWidgets.QGroupBox("MIDI Event Action Mapping", self)
        # -----------------------------------------------------------------------
        _S_ = 1  # Scrollable rows
        midi_actions_content = QtWidgets.QWidget()
        # Create a widget and layout for the contents inside the scroll area
        if _S_ == 0:
            midi_actions_layout = QtWidgets.QGridLayout()
        else:
            midi_actions_layout = QtWidgets.QGridLayout(midi_actions_content)
        # -----------------------------------------------------------------------

        # Retrieve the dictionary with the default midi-event-action mappings
        actions_dict = get_default_action_midi_mappings_as_dict()

        self.midi_assignment_buttons = {}
        self.midi_msg_data_spinboxes = {}  # Using a dictionary to store QLineEdit references by action name
        self.midi_msg_type_dropdown = {}
        self.midi_data_note_label = {}

        # NOTE: static default values are being assigned here.
        # This can also be done at the beginning of the load() method.

        for index, action in enumerate(actions_dict):
            MAE = actions_dict[action]  # Shorthand for MidiEventAction
            if not MAE.is_mappable_in_ui:
                continue

            # Initialize previous values for each action
            self.previous_ui_action_values[action] = {'midi_type': None, 'data_value': None}

            # Create button for the dynamic assignment action
            assignment_button = QtWidgets.QPushButton(MAE.format_ui_action_label())
            assignment_button.setStyleSheet('text-align: left;')  # Align text to the left
            assignment_button.setToolTip(MAE.description)
            assignment_button.clicked.connect(
                lambda _, _a=action, _l=MAE.ui_action_label: self.handle_assignment_button_click(_a, _l))
            self.midi_assignment_buttons[action] = assignment_button
            add_icon_to(assignment_button, event_to_icon_mapping[action])
            # -----------------------------------------------------------------------
            # TODO: here we either use the label or the button. The "_S_" is an A/B switch to test both visualizations
            # if _S_ == 0:
            midi_actions_layout.addWidget(assignment_button, index, 0)
            # else:
            #     midi_actions_layout.addWidget(assignment_button, (_S_ + 1) * index, 0, 1, 3)

            # index = (_S_ + 1) * index + (1 if _S_ > 0 else 0)
            # -----------------------------------------------------------------------
            # Create spin box for MIDI value
            midi_msg_data_spinbox = QtWidgets.QSpinBox()
            midi_msg_data_spinbox.setRange(0, 127)
            midi_msg_data_spinbox.setValue(MAE.midi_data)
            midi_actions_layout.addWidget(midi_msg_data_spinbox, index, 1)  # - _S_
            # Storing reference using the original action name
            self.midi_msg_data_spinboxes[action] = midi_msg_data_spinbox

            # Create label for MIDI note
            self.midi_data_note_label[action] = QtWidgets.QLabel(MIDI_Def.int_to_note(MAE.midi_data))
            midi_actions_layout.addWidget(self.midi_data_note_label[action], index, 2)  # - _S_

            # Create dropdown for MIDI message type
            midi_message_type_dropdown = QtWidgets.QComboBox()
            midi_message_type_dropdown.addItems(MIDI_Def.MIDI_MESSAGE_TYPES_LIST)
            midi_actions_layout.addWidget(midi_message_type_dropdown, index, 3)  # - _S_
            self.midi_msg_type_dropdown[action] = midi_message_type_dropdown
            self.set_midi_message_type_dropdown(action, MAE.midi_type)

            # Update MIDI note label when spin box value changes
            midi_msg_data_spinbox.valueChanged.connect(
                lambda value, _action=action: self.on_midi_data_ui_state_change_callback(_action)
            )

            midi_message_type_dropdown.currentIndexChanged.connect(
                lambda value, _action=action: self.on_midi_data_ui_state_change_callback(_action)
            )

        # -----------------------------------------------------------------------
        if _S_ == 0:
            midi_actions_group.setLayout(midi_actions_layout)  # TODO: original
        else:
            # Create a scroll area and set its widget to be the midi_actions_content
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)  # Make the scroll area resizable
            scroll_area.setWidget(midi_actions_content)

            # Create a layout for the group box and add the scroll area to it
            group_layout = QtWidgets.QVBoxLayout(midi_actions_group)
            group_layout.addWidget(scroll_area)
        # -----------------------------------------------------------------------

        return midi_actions_group

    def retranslate_ui(self):
        """
        Setup the interface translation strings for MidiSettingsTab.
        """
        # Using the translation function with the correct context
        context = self.objectName()

        # Translating group box titles
        # TODO: that's a bit manual and hardcoded here
        self.profile_group.setTitle(translate(context, "Profile Management"))
        self.midi_devices_group.setTitle(translate(context, "MIDI Devices, Channels and Options"))
        self.midi_actions_group.setTitle(translate(context, "MIDI Event Action Mapping"))

        # Translating labels within the groups
        self.profile_label.setText(translate(context, self.profile_label.text()))
        self.input_devices_label.setText(translate(context, self.input_devices_label.text()))
        self.output_devices_label.setText(translate(context, self.output_devices_label.text()))
        self.reset_midi_state_checkbox.setText(translate(context, self.reset_midi_state_checkbox.text()))
        self.device_sync_checkbox.setText(translate(context, self.device_sync_checkbox.text()))
        self.play_button_is_toggle_checkbox.setText(translate(context, self.play_button_is_toggle_checkbox.text()))
        self.deferred_control_mode_checkbox.setText(translate(context, self.deferred_control_mode_checkbox.text()))
        self.control_offset_label.setText(translate(context, self.control_offset_label.text()))

        # Translating buttons within the Profile Management group
        self.create_profile_button.setText(translate(context, self.create_profile_button.text()))
        self.save_profile_button.setText(translate(context, self.save_profile_button.text()))
        self.rename_profile_button.setText(translate(context, self.rename_profile_button.text()))
        self.delete_profile_button.setText(translate(context, self.delete_profile_button.text()))

        # Translating labels within the MIDI Event Action Mapping group
        for action, midi_data_field in self.midi_msg_data_spinboxes.items():
            tal = translate(context, self.midi_assignment_buttons[action].text())
            self.midi_assignment_buttons[action].setText(tal)

    def initialise(self):
        # Create the db manager before the constructor.
        # The constructor will call the UI and will require the settings to be initialized.
        self.db_manager = MidiProfileManager()

    # ============== MIDI configuration tab state section ==============
    # NOTE: this method overwrites the parent
    def load(self):
        """
        Load settings and populate the profile dropdown and device dropdowns.
        """
        self.update_listeners("on_cfg_open")
        # Load profiles and devices
        self._load_midi_devices()
        self._load_profile_names()
        self._load_profile_state()
        # NOTE: The order here matters as the devices and the profile names should be loaded first.
        # When the profile state is reloaded it will determine how to handle the preselected profile
        # device with the currently available devices.

    # NOTE: this method overwrites the parent
    def save(self):
        self.ensure_mel_is_joined()
        # self._save_profile_callback() # TODO: consider if you want to save the current profile on clicking "OK"
        # TODO: maybe there should be a dialogue asking if the changes should be saved.
        # TODO: That means we have to have a way to extract state and compare it to change if there is something to save
        MidiSettingsTab.update_listeners(callback_type="on_cfg_close")
        pass

    # NOTE: this method overwrites the parent
    def cancel(self):
        self.ensure_mel_is_joined()
        MidiSettingsTab.update_listeners(callback_type="on_cfg_close")

    # ============== MIDI configuration loading section ==============

    def _load_midi_devices(self):
        """
        Load MIDI devices into the input and output device dropdowns.
        Load MIDI channels into the input and output channel dropdowns.
        """
        # Populating the midi devices dropdowns
        midi_input_devices = MidiDeviceHandler.get_input_midi_devices()
        midi_output_devices = MidiDeviceHandler.get_output_midi_devices()

        # Add placeholder options
        midi_input_devices.insert(0, disabled_midi_device['input'])
        midi_input_devices.insert(1, openlp_midi_device['gui_label'])
        midi_output_devices.insert(0, disabled_midi_device['output'])
        # TODO: we may want to do the same for the output device and set OpenLP as output device
        # TODO : put those text labels in the constants file

        self.midi_device_input_dropdown.clear()
        self.midi_device_output_dropdown.clear()

        self.midi_device_input_dropdown.addItems(midi_input_devices)
        self.midi_device_output_dropdown.addItems(midi_output_devices)

        # Populating the midi channel dropdowns
        midi_channels = MidiDeviceHandler.get_midi_channels_list()

        # Clear the existing items in the dropdowns
        self.midi_channel_input_dropdown.clear()
        self.midi_channel_output_dropdown.clear()

        # Add the MIDI channels to the dropdowns
        self.midi_channel_input_dropdown.addItems(midi_channels)
        self.midi_channel_output_dropdown.addItems(midi_channels)

        # Select the last entry in the MIDI channel dropdowns
        _in_ch_ind = MidiDeviceHandler.get_channel_index(default_midi_device['output_channel'])
        self.midi_channel_input_dropdown.setCurrentIndex(_in_ch_ind)
        _out_ch_ind = MidiDeviceHandler.get_channel_index(default_midi_device['input_channel'])
        self.midi_channel_output_dropdown.setCurrentIndex(_out_ch_ind)

    def _load_profile_names(self):
        """
        Load profile names into the profile dropdown.
        """
        profile_names = self.db_manager.get_all_profiles()
        selected_profile_name = None

        # Loop to find the selected profile
        for profile in profile_names:
            if self.db_manager.get_property(profile, 'is_selected_profile'):
                selected_profile_name = profile
                break

        self.profile_dropdown.clear()
        # We need to temporarily disconnect the callback because we are adding and item,
        # and we don't want unwanted callbacks to get triggered
        self.profile_dropdown.blockSignals(True)
        self.profile_dropdown.addItems(profile_names)
        self.profile_dropdown.blockSignals(False)

        # Set the selected profile if found
        if selected_profile_name:
            index = self.profile_dropdown.findText(selected_profile_name)
            if index != -1:
                self.profile_dropdown.setCurrentIndex(index)

        self._prevent_rename_or_delete_of_default_profile()

    def _load_profile_state(self):
        """
        Load the profile state based on the currently selected profile.
        This function acts as a callback for the profile dropdown change event.
        """
        currently_selected_profile = self.profile_dropdown.currentText()
        if not currently_selected_profile and currently_selected_profile not in self.db_manager.get_all_profiles():
            return

        self.db_manager.set_profile_as_currently_selected(currently_selected_profile)
        # Fetch the profile state from the db_manager.
        # I'm assuming a method `get_profile_state` exists in the db_manager.
        # This method should return a dictionary with keys corresponding to the settings.
        profile_state = self.db_manager.get_profile_state(currently_selected_profile)

        # Set the MIDI input and output devices and channels from the loaded state.
        # If not present in the profile state, keep the default (i.e., do not change).
        if "input_midi_device" in profile_state:
            index = self.midi_device_input_dropdown.findText(profile_state["input_midi_device"])
            if index != -1:
                self.midi_device_input_dropdown.setCurrentIndex(index)
            else:
                self.midi_device_input_dropdown.setCurrentIndex(0)

        if "input_device_channel" in profile_state:
            index = self.midi_channel_input_dropdown.findText(str(profile_state["input_device_channel"]))
            if index != -1:
                self.midi_channel_input_dropdown.setCurrentIndex(index)

        if "output_midi_device" in profile_state:
            index = self.midi_device_output_dropdown.findText(profile_state["output_midi_device"])
            if index != -1:
                self.midi_device_output_dropdown.setCurrentIndex(index)
            else:
                self.midi_device_output_dropdown.setCurrentIndex(0)

        if "output_device_channel" in profile_state:
            index = self.midi_channel_output_dropdown.findText(str(profile_state["output_device_channel"]))
            if index != -1:
                self.midi_channel_output_dropdown.setCurrentIndex(index)

        # Set the MIDI state reset checkbox based on the loaded profile.
        if "reset_midi_state" in profile_state:
            self.reset_midi_state_checkbox.setChecked(profile_state["reset_midi_state"])

        if "device_sync" in profile_state:
            self.device_sync_checkbox.setChecked(profile_state["device_sync"])

        if "play_button_is_toggle" in profile_state:
            self.play_button_is_toggle_checkbox.setChecked(profile_state["play_button_is_toggle"])

        # Set the deferred control mode checkbox based on the loaded profile.
        if "deferred_control_mode" in profile_state:
            self.deferred_control_mode_checkbox.setChecked(profile_state["deferred_control_mode"])

        if "control_offset" in profile_state:
            self.control_offset_spinner.setValue(profile_state["control_offset"])

        # Load action mappings
        for action in self.midi_msg_data_spinboxes.keys():
            if action in profile_state:
                # NOTE: We block the signals because we don't want to trigger the duplication check during assignment
                self.midi_msg_type_dropdown[action].blockSignals(True)
                self.midi_msg_data_spinboxes[action].blockSignals(True)
                self.set_midi_message_type_dropdown(action, profile_state[action].midi_type)
                self.set_midi_data_value(action, profile_state[action].midi_data)
                self.midi_msg_type_dropdown[action].blockSignals(False)
                self.midi_msg_data_spinboxes[action].blockSignals(False)
                # After all values are updated we can manually trigger the callback once to make sure the UI updated
                self.on_midi_data_ui_state_change_callback(action)

    # ============== MIDI configuration get from UI state section ==============
    def get_ui_configuration_state(self) -> dict:
        """
        Extracts the current state of all mappings from the UI settings tab.

        Returns:
            dict: A dictionary containing the current UI settings.
        """
        ui_settings = dict()

        # Get the currently selected profile
        ui_settings["profile"] = self.profile_dropdown.currentText()

        # Get the currently selected MIDI input and output devices and channels
        ui_settings["input_midi_device"] = self.midi_device_input_dropdown.currentText()
        ui_settings["input_device_channel"] = self.midi_channel_input_dropdown.currentText()
        ui_settings["output_midi_device"] = self.midi_device_output_dropdown.currentText()
        ui_settings["output_device_channel"] = self.midi_channel_output_dropdown.currentText()

        # Get the state of the deferred control mode checkbox
        ui_settings["reset_midi_state"] = self.reset_midi_state_checkbox.isChecked()
        ui_settings["device_sync"] = self.device_sync_checkbox.isChecked()
        ui_settings["play_button_is_toggle"] = self.play_button_is_toggle_checkbox.isChecked()
        ui_settings["deferred_control_mode"] = self.deferred_control_mode_checkbox.isChecked()
        ui_settings["control_offset"] = int(self.control_offset_spinner.value())

        # Loop through the midi value spin boxes to get the current mapping values
        action_mappings = {}
        for action, midi_data_field in self.midi_msg_data_spinboxes.items():
            # NOTE: Here we simply export the modifiable fields as dictionary.
            # TODO: alternatively "self.midi_msg_data_spinboxes[action]"
            action_mappings[action] = {
                'midi_data': midi_data_field.text(),
                'midi_type': self.midi_msg_type_dropdown[action].currentText()
            }

        # Merge the 2 dictionaries together and return the dictionary
        configuration_state = {**ui_settings, **action_mappings}
        return configuration_state

    # ============== Profile management handlers section ==============

    def _create_profile_callback(self):
        """
        Slot to handle the creation of a new profile.
        """
        profile_name, ok = QtWidgets.QInputDialog.getText(self, 'Create Profile', 'Enter profile name:')
        if ok and profile_name:
            try:
                # Logic to create profile
                self.db_manager.create_profile(profile_name)

                # Add the profile name at the bottom and select it in the dropdown
                # We need to temporarily disconnect the callback because we are adding and item,
                # and we don't want unwanted callbacks to get triggered
                self.profile_dropdown.blockSignals(True)
                self.profile_dropdown.addItem(profile_name)
                # TODO: there is some weirdness related to creating a profile.
                #  it's not behaving as expected and is creating default profile.
                # Set the selected profile if found
                index = self.profile_dropdown.findText(profile_name)
                if index != -1:
                    self.profile_dropdown.setCurrentIndex(index)
                self.profile_dropdown.blockSignals(False)

                self._save_profile_callback()
                self._prevent_rename_or_delete_of_default_profile()
            except Exception as e:
                # Display the error message
                error_dialog = QtWidgets.QMessageBox(self)
                error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
                error_dialog.setWindowTitle("Error Creating Profile")
                error_dialog.setText("An error occurred while creating the profile.")
                error_dialog.setInformativeText(str(e))
                error_dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
                error_dialog.exec_()

    def _save_profile_callback(self):
        """
        Save the current UI settings to the database.
        """
        # 1. Get the current UI settings using the get_action_event_mapping_from_ui method
        config_state = self.get_ui_configuration_state()

        # 2. Check if the dictionary keys from the UI match the database column names (consistency check)
        action_keys = self.db_manager.get_midi_event_action_key_list()
        all_db_properties = (self.db_manager.get_midi_config_properties_key_list() + action_keys)

        # Check if all keys in ui_settings match with database columns (consistency check)
        for key in config_state.keys():
            if key not in all_db_properties:
                raise ValueError(f"UI setting '{key}' does not match any database column.")

        # 3. Save the current UI settings to the relevant database fields
        currently_selected_profile = config_state["profile"]

        # If no profile is selected, create a new profile called default and select it
        if currently_selected_profile == '':
            currently_selected_profile = "default"
            self.db_manager.create_profile("default")
            self._load_profile_names()
            config_state["profile"] = currently_selected_profile

        self.db_manager.set_profile_as_currently_selected(currently_selected_profile)

        for property_name, value in config_state.items():
            # NOTE: We are not using a template or copy of the default midi-action-event mapping, therefore,
            #       for consistency, we use a SimpleNamespace form the dictionary. This way we ensure that
            #       the set method treats the input the same, regardless of whether the modified fields passed
            #       like this or with a midi-event mapping instance.
            if property_name in action_keys:
                value = SimpleNamespace(**value)
            self.db_manager.set_property(currently_selected_profile, property_name, value)

        # Done! The settings from the UI have been saved to the database.

    def _rename_profile_callback(self):
        """
        Slot to handle renaming an existing profile.
        """
        current_name = self.profile_dropdown.currentText()
        new_name, ok = QtWidgets.QInputDialog.getText(self, 'Rename Profile', 'Enter new profile name:',
                                                      text=current_name)
        if ok and new_name:
            try:
                # Logic to rename profile
                self.db_manager.rename_profile(current_name, new_name)
                self.db_manager.set_profile_as_currently_selected(new_name)
                self._load_midi_devices()
                self._load_profile_names()
                self._load_profile_state()
            except Exception as e:
                # Display the error message
                error_dialog = QtWidgets.QMessageBox(self)
                error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
                error_dialog.setWindowTitle("Error Renaming Profile")
                error_dialog.setText("An error occurred while renaming the profile.")
                error_dialog.setInformativeText(str(e))
                error_dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
                error_dialog.exec_()

    def _delete_profile_callback(self):
        """
        Slot to handle deletion of a profile.
        """
        profile_name = self.profile_dropdown.currentText()
        confirm = QtWidgets.QMessageBox.question(self, 'Delete Profile',
                                                 f'Are you sure you want to delete the profile "{profile_name}"?')
        if confirm == QtWidgets.QMessageBox.Yes:
            try:
                # Logic to delete profile
                self.db_manager.delete_profile(profile_name)
                self._load_midi_devices()
                self._load_profile_names()
                self._load_profile_state()
            except Exception as e:
                # Display the error message
                error_dialog = QtWidgets.QMessageBox(self)
                error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
                error_dialog.setWindowTitle("Error Deleting Profile")
                error_dialog.setText("An error occurred while deleting the profile.")
                error_dialog.setInformativeText(str(e))
                error_dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
                error_dialog.exec_()

    def _update_profile_callback(self):
        self._load_profile_state()
        self._prevent_rename_or_delete_of_default_profile()
        # TODO : this is effective the reset call
        MidiSettingsTab.update_listeners(callback_type="on_cfg_update")

    def _prevent_rename_or_delete_of_default_profile(self):
        # Prevent the default profile from being renamed or deleted
        isEnabled = True
        if self.profile_dropdown.currentText() == "default":
            isEnabled = False
        self.rename_profile_button.setEnabled(isEnabled)
        self.delete_profile_button.setEnabled(isEnabled)

    # ============== MIDI assignment handlers section ==============

    def handle_assignment_button_click(self, action, ui_action_label):

        if self._current_assignment_ is None:  # Activate assignment
            # NOTE: Only the main thread can enter here. Ensure the assignment MIDI event listener thread was joined
            self.ensure_mel_is_joined()

            # Start a listener thread to get single MIDI event
            self._current_assignment_mel = MidiEventListener(on_receive_callback_ref=self.handle_midi_data_assignment)
            # Try to connect to the current device and return if unsuccessful
            device_name = self.midi_device_input_dropdown.currentText()
            device_name = MidiDeviceHandler.match_exact_input_device_name(device_name)
            if not self._current_assignment_mel.connect_to_device(device_name):
                return
            # Start the listening tread
            self._current_assignment_mel.get_single_midi_event_thread_start()

            # Store the current assignment configuration
            self._current_assignment_ = {'action': action,
                                         'original_label': self.midi_assignment_buttons[action].text(),
                                         'original_style': self.midi_assignment_buttons[action].styleSheet(),
                                         'ui_action_label': ui_action_label}
            self._current_assignment_ = SimpleNamespace(**self._current_assignment_)

            # Set temporary properties for label and style
            self.midi_assignment_buttons[action].setText(assignment_message)
            self.midi_assignment_buttons[action].setStyleSheet("background-color: red;")
            print(f"Dynamic midi assignment!\n"
                  f"Waiting for the first MIDI input to be assigned to [{ui_action_label}]!\n"
                  "Please press the midi button you want to assign!")

        else:  # Deactivate assignment
            action = self._current_assignment_.action
            # Revert to the original label and style properties
            self.midi_assignment_buttons[action].setText(self._current_assignment_.original_label)
            self.midi_assignment_buttons[action].setStyleSheet(self._current_assignment_.original_style)
            # Get the MIDI event listener & close any threads
            self._current_assignment_mel.stop_from_inside()
            self._current_assignment_ = None
            print("Dynamic midi assignment disabled!")

    def handle_midi_data_assignment(self, midi_message):
        action = self._current_assignment_.action
        label = self._current_assignment_.ui_action_label
        text_info = f"Dynamic MIDI assignment message detected [{midi_message}] which will be assigned to[{label}]!\n"
        print(text_info)

        if midi_message is not None:
            duplicate_found, duplicate_action = self.check_for_mapping_duplication_from_message(action, midi_message)
            if not duplicate_found:
                midi_data = self._current_assignment_mel.extract_midi_data1(midi_message)

                # Assign the MIDI event to the UI elements
                self.set_midi_message_type_dropdown(action, midi_message.type)
                self.set_midi_data_value(action, midi_data)
            else:
                actions_dict = get_default_action_midi_mappings_as_dict()
                duplicate_action_label = actions_dict[duplicate_action].ui_action_label
                # self.show_duplicate_mapping_message(duplicate_action_label)
                # Use QMetaObject.invokeMethod to call the method in the main thread
                QMetaObject.invokeMethod(self, "show_duplicate_mapping_message", Qt.QueuedConnection,
                                         QtCore.Q_ARG(str, duplicate_action_label))

        # Stop the listening thread and reset the UI
        self.handle_assignment_button_click("", "")

    def ensure_mel_is_joined(self):
        # NOTE: This can't be initialized from inside the thread.
        # There fore we can call that separately on the main thread on UI state change relevant to this listener.
        if self._current_assignment_mel is not None:
            thread = self._current_assignment_mel.thread
            if thread and thread.is_alive():
                thread.join()

    # ============== MIDI assignment duplication/validation section ==============

    def check_for_mapping_duplication_from_message(self, event_action, midi_message):
        message_type = midi_message.type
        midi_data = self._current_assignment_mel.extract_midi_data1(midi_message)
        return self.check_for_mapping_duplication(event_action, message_type, midi_data)

    def check_for_mapping_duplication(self, event_action, message_type, midi_data):
        duplicate_found = False
        duplicate_action = None
        # Ensure the argument states are normalized
        message_type = message_type.replace(' ', '_').lower()  # TODO: again the ugly conversion
        midi_data = int(midi_data)

        # Get the current UI configuration state
        mappings = self.get_ui_configuration_state()

        # Filter only event mappings
        event_mappings = {event: mapping for event, mapping in mappings.items() if event.startswith("event_")}

        # Iterate through existing mappings to check for duplication
        for action, mapping in event_mappings.items():
            # If the action is overwritten with the same value then we just skip
            if action == event_action:  # Avoid self-conflict check
                continue
            # Ensure the argument states are normalized
            mapping['midi_type'] = mapping['midi_type'].replace(' ', '_').lower()  # TODO: again the ugly conversion
            mapping['midi_data'] = int(mapping['midi_data'])
            if mapping['midi_type'] == message_type and mapping['midi_data'] == midi_data:
                print(f"Duplicate mapping found for action: {action}")
                duplicate_found = True
                duplicate_action = action
                break

        return duplicate_found, duplicate_action

    @QtCore.pyqtSlot(str)
    def show_duplicate_mapping_message(self, duplicate_action_label):
        # This method will run in the main thread
        # Show message box for duplication
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle("Duplicate MIDI Mapping Detected!")
        msg_box.setText(f"Duplicate MIDI mapping found for action: [ {duplicate_action_label} ]!")
        msg_box.setInformativeText("Please assign a different MIDI control or note.")
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg_box.exec_()  # Display the message box

    # ============== Set MIDI type and value section ==============

    def set_midi_message_type_dropdown(self, action, message_type):
        """
        Set the MIDI message type for a specific action in the dropdown.
        Ignore capitalization, spaces, and underscores.
        """

        # TODO: generally we should be using mapping rather that normalization, but it's kinda easy ... for now
        # mapping = MIDI_states.map_midi_type_to_ui_label()
        # ui_label = mapping[message_type]
        def normalize(text):
            return text.lower().replace(" ", "").replace("_", "")

        normalized_message_type = normalize(message_type)

        if action in self.midi_msg_type_dropdown:
            dropdown = self.midi_msg_type_dropdown[action]
            for i in range(dropdown.count()):
                if normalize(dropdown.itemText(i)) == normalized_message_type:
                    dropdown.setCurrentIndex(i)
                    self.previous_ui_action_values[action]['midi_type'] = dropdown.itemText(i)
                    break

    def set_midi_data_value(self, action, data_value):
        """
        Set the MIDI data value for a specific action. The value can be a number or a note representation.
        """
        if action in self.midi_msg_data_spinboxes:
            spin_box = self.midi_msg_data_spinboxes[action]

            if isinstance(data_value, str):
                data_value = int(data_value)
            spin_box.setValue(data_value)
            self.previous_ui_action_values[action]['data_value'] = data_value

    def on_midi_data_ui_state_change_callback(self, action):
        """
        Callback for when MIDI UI elements are changed manually.

        Args:
            action (str): The action associated with the UI elements.
        """
        duplicate_found, duplicate_action = self.check_for_mapping_duplication(
            event_action=action,
            message_type=self.midi_msg_type_dropdown[action].currentText(),
            midi_data=int(self.midi_msg_data_spinboxes[action].value())
        )
        if duplicate_found and duplicate_action != action:  # Avoid self-conflict check
            actions_dict = get_default_action_midi_mappings_as_dict()
            # Notify user of duplication and revert changes or handle as needed
            self.show_duplicate_mapping_message(actions_dict[duplicate_action].ui_action_label)
            # Revert to previous state
            self.set_midi_message_type_dropdown(action, self.previous_ui_action_values[action]['midi_type'])
            self.set_midi_data_value(action, self.previous_ui_action_values[action]['data_value'])

        # Get the normalized midi type string
        midi_type = self.midi_msg_type_dropdown[action].currentText()
        spin_box = self.midi_msg_data_spinboxes[action]

        # Determine the midi type data 1 & normalize the midi data
        data_value = spin_box.value()
        norm_midi_type = midi_type.lower()  # .replace(" ", "").replace("_", "")

        # Save the current state
        self.previous_ui_action_values[action]['midi_type'] = midi_type
        self.previous_ui_action_values[action]['data_value'] = data_value

        # Handle the text label
        text = ""
        if "note" in norm_midi_type:
            text = MIDI_Def.int_to_note(data_value)
        if "control" in norm_midi_type:
            text = "CC"
        self.midi_data_note_label[action].setText(text)

        # Handle the spin box state
        if not ("note" in norm_midi_type or "control" in norm_midi_type):
            spin_box.setDisabled(True)
            spin_box.setVisible(False)
        else:
            spin_box.setDisabled(False)
            spin_box.setVisible(True)

    # ============== Listeners management section ==============

    @staticmethod
    def add_listener_callback(listener_id: str, callback_type: str, callback: classmethod):
        """
        Registers a callback method for a given listener.

        This method allows external components to subscribe to updates or changes
        in the MIDI settings. When the MIDI settings are updated, these subscribed
        listeners will be notified.

        Args:
            listener_id (str): A unique identifier for the listener.
            callback_type (str): The type of callback (e.g., 'on_open', 'on_close').
            callback (classmethod): The callback method to be called.
        """
        if callback_type not in MidiSettingsTab.listener_callbacks:
            raise ValueError(f"Unknown callback type: {callback_type}")
        MidiSettingsTab.listener_callbacks[callback_type][listener_id] = callback

    @staticmethod
    def update_listeners(callback_type: str, event=None):
        """
        Notifies all registered listeners about a change in the MIDI settings.

        This method iterates through all the registered listeners and invokes their
        callback methods. This is typically called after a change or update in the
        MIDI settings to inform all listeners about the change.
        """
        event = None  # TODO: pause here
        if callback_type not in MidiSettingsTab.listener_callbacks:
            raise ValueError(f"Unknown callback type: {callback_type}")

        for listener_id, callback in MidiSettingsTab.listener_callbacks[callback_type].items():
            print(f"Update {listener_id} about midi settings change.")
            callback(event)  # Pass the event to the callback
