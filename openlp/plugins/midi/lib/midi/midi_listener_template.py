from abc import abstractmethod  # ABC, # TODO: abstraction could be used
from typing import Callable
import time
from threading import Thread

from openlp.plugins.midi.lib.handlers_managers.device_handler import MidiDeviceHandler
from openlp.plugins.midi.lib.handlers_managers.profile_db_manager import get_midi_configuration
from openlp.plugins.midi.lib.types_definitions.constants import midi_ch_any


class MidiListenerTemplate():
    def __init__(self, on_receive_callback_ref: Callable):
        """
        Initialize the MIDI listener with a callback for MIDI events.

        :param on_receive_callback_ref: A callback function that will be called with each MIDI event.
        """
        self.thread = None
        self._event_callback = on_receive_callback_ref
        self.transmit_callback = None

        # Device connection retry time out
        self.device_not_available_timeout = 1  # wait for that many seconds before rechecking

        self.listening_is_active_flag = False
        self.exit_listener_flag = False
        self.receiver_disabled_flag = False

        # MIDI configuration
        self.midi_config = None
        # Shorthand mappings
        self.mappings = None

    @abstractmethod
    def connect_to_device(self, device: str) -> bool:
        """
        Establish a connection to a specified MIDI device and port.
        """
        pass

    @abstractmethod
    def listen_for_a_single_midi_event(self):
        """
        Listen for MIDI events on the selected MIDI input device and return the first incoming MIDI message.
        """
        pass

    @abstractmethod
    def listen_for_midi_events(self):
        """
        Continuously listen for MIDI messages.
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stop listening for MIDI messages.
        """
        pass

    @abstractmethod
    def stop_from_inside(self):
        """
        Stop listening for MIDI messages but doesn't stop the thread. Used for MIDI mapping assignments.
        """
        pass

    # ======= Methods to inherit =======
    def is_disabled(self):
        return self.receiver_disabled_flag

    def _async_pause(self, pause_duration=0.01):
        """
        Pause the thread execution asynchronously for a specified duration.

        :param pause_duration: Duration in seconds for each pause cycle. Default is 0.1 seconds.
        """
        # NOTE: TO AVOID 100% THREAD UTILIZATION DO AN UGLY PAUSE
        time.sleep(pause_duration)

    @staticmethod
    def get_input_midi_device_list() -> list:
        """
        Return a list of available MIDI input devices.
        """
        return MidiDeviceHandler.get_input_midi_devices()

    def start(self):
        self.thread = Thread(target=self.retry_to_connect_and_listen)
        self.thread.start()

    def get_single_midi_event_thread_start(self):
        self.thread = Thread(target=self.listen_for_a_single_midi_event)
        self.thread.start()

    def retry_to_connect_and_listen(self):
        while not self.connect_to_device() and not self.exit_listener_flag and not self.is_disabled():
            print(f"MidiListenerTemplate | Retrying to connect to in {self.device_not_available_timeout} seconds...")
            time.sleep(self.device_not_available_timeout)

        if self.listening_is_active_flag:
            self.listen_for_midi_events()

    def request_exit(self):
        self.exit_listener_flag = True

    def request_restart(self):
        if self.listening_is_active_flag:
            self.stop()  # Stop the current listening process
        self.start()  # Start a new listening process

    def handle_midi_event(self, midi_message):

        # Filter messages based on the selected channel
        if ((self.midi_config.input_device_channel == midi_ch_any or
             (midi_message.channel + 1) == self.midi_config.input_device_channel)):

            # Get the MIDI data value, i.e. Note or CC or Program etc.
            midi_data_1 = self.extract_midi_data1(midi_message)
            velocity_or_value = self.extract_midi_data2(midi_message)

            # Check for event matches
            matching_events = []
            for event_key, event in self.mappings.items():
                if event.midi_type == midi_message.type and event.midi_data == midi_data_1:
                    matching_events.append(event)

            # Check and validate the matching
            if len(matching_events) == 1:
                # Found exactly one matching event
                print(f"MidiListenerTemplate | MIDI message detected {midi_message}")
                self._event_callback(matching_events[0].mapping_key, velocity_or_value)
                # We can just update only if the midi event was recognized
                if self.midi_config.device_sync:
                    self.transmit_callback()

            if len(matching_events) > 1:
                # We should never really come here if the mapping is unique as it should be.
                raise ValueError("MidiListenerTemplate | Multiple events found for MIDI message.")

    def unpacked_midi_configuration(self):
        self.midi_config = get_midi_configuration()
        # Extract only the mappings
        self.mappings = {key: value for key, value in self.midi_config.__dict__.items() if key.startswith("event")}

        # UI labels need to be formatted # TODO: again the direct conversion (ugly)
        for key, value in self.mappings.items():
            self.mappings[key].midi_type = value.midi_type.lower().replace(" ", "_")

        if MidiDeviceHandler.should_sanitize_names():
            self.midi_config.input_midi_device = (
                MidiDeviceHandler.match_exact_input_device_name(self.midi_config.input_midi_device))

    def extract_midi_data1(self, midi_message) -> int:
        """
        Extracts the relevant MIDI data 1 from the message (e.g. note value, control CC parameter, program number, etc.)
        """
        if hasattr(midi_message, 'note'):
            return midi_message.note
        elif hasattr(midi_message, 'control'):
            return midi_message.control
        elif hasattr(midi_message, 'program'):
            return 0  # NOTE: here we return a static value to identify the control type.
        elif hasattr(midi_message, 'pitch'):
            return 0  # NOTE: here we return a static value to identify the control type.
        else:
            raise NotImplementedError(f'MidiEventListener | MIDI conversion not implemented for {midi_message}.')

    def extract_midi_data2(self, midi_message) -> int:
        """
        Extracts the relevant MIDI data 2 (like note velocity or control CC value ) from a mido message.
        """
        if hasattr(midi_message, 'velocity'):
            return midi_message.velocity
        elif hasattr(midi_message, 'value'):
            return midi_message.value
        # NOTE: The program and pitch are technically not data2.
        # However, in terms of control, here it's more useful as such!
        # Because data2 is the control variable similar to velocity.
        # TODO: we should probably reflect that in the UI assignment
        elif hasattr(midi_message, 'program'):
            return midi_message.program
        elif hasattr(midi_message, 'pitch'):
            return midi_message.pitch
        # Add more cases as needed for different message types
        else:
            print(f'MidiEventListener | MIDI conversion not implemented or supported for {midi_message}.')
            return 0  # Default value for messages without a 'data2' equivalent

    @abstractmethod
    def __str__(self):
        return 'MidiEventListenerTemplate'
