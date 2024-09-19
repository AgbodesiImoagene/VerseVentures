import pygame.midi
from openlp.plugins.midi.lib.midi.midi_listener_template import MidiListenerTemplate

from collections import namedtuple

from openlp.plugins.midi.lib.types_definitions.constants import openlp_midi_device, disabled_midi_device

# Define a simple MIDI message structure similar to Mido
MidiMessage = namedtuple('MidiMessage', ['type', 'note', 'velocity', 'channel'])


def convert_pygame_midi_to_mido(pygame_midi_data):
    """
    Convert Pygame MIDI data to a format similar to Mido's Message.
    This is a basic example and might need adjustments based on the specific MIDI data format.
    """
    status_byte = pygame_midi_data[0][0][0]
    data_bytes = pygame_midi_data[0][0][1:]

    # Extract MIDI message components
    channel = status_byte & 0x0F
    status = status_byte & 0xF0

    # Define MIDI message type based on the status byte
    if status == 0x90:  # Note On
        message_type = 'note_on'
    elif status == 0x80:  # Note Off
        message_type = 'note_off'
    else:
        message_type = 'unknown'

    # TODO : a full list should be implemented here!!!!!

    # Assuming standard Note On/Off messages with two data bytes: note and velocity
    note = data_bytes[0] if len(data_bytes) > 0 else None
    velocity = data_bytes[1] if len(data_bytes) > 1 else None

    return MidiMessage(type=message_type, note=note, velocity=velocity, channel=channel)


class MidiEventListener(MidiListenerTemplate):
    def __init__(self, on_receive_callback_ref):
        super().__init__(on_receive_callback_ref)
        pygame.midi.init()
        self.midi_in = None
        self.thread = None
        self.listening_is_active_flag = False

    def connect_to_device(self, device: str = None):
        self.unpacked_midi_configuration()
        if device is None:
            device = self.midi_config.input_midi_device

        if device == disabled_midi_device['input']:
            self.receiver_disabled_flag = True
            return False

        if device == openlp_midi_device['gui_label']:  # Replace with your virtual device identifier
            # Handle virtual device creation
            # Note: pygame.midi might not support virtual MIDI devices directly
            print("Virtual MIDI device creation is not supported by pygame.midi.")
            return False

        device_id = self._get_device_id_by_name(device)
        if device_id is not None:
            try:
                self.midi_in = pygame.midi.Input(device_id)
                self.listening_is_active_flag = True
                return True
            except Exception as e:
                print(f"Failed to connect to {device}. Error: {e}")
                return False
        else:
            print(f"Device '{device}' not found.")
            return False

    def listen_for_midi_events(self):
        while self.listening_is_active_flag:
            if self.midi_in.poll():
                pygame_midi_message = self.midi_in.read(1)
                self.handle_midi_event(convert_pygame_midi_to_mido(pygame_midi_message))
            self._async_pause()

    def listen_for_a_single_midi_event(self):
        while self.listening_is_active_flag:
            if self.midi_in.poll():
                pygame_midi_message = self.midi_in.read(1)
                self._event_callback(convert_pygame_midi_to_mido(pygame_midi_message))
                return  # Return after the message is processed
            self._async_pause(0.001)

    def stop(self):
        print("Request received to stop the MIDI listener.")
        self.request_exit()
        self.listening_is_active_flag = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.midi_in:
            self.midi_in.close()
        print("Stop the MIDI listener. End of method")

    def stop_from_inside(self):
        print("Request received to stop the MIDI listener from INSIDE.")
        self.request_exit()
        self.listening_is_active_flag = False
        self._async_pause(0.02)
        if self.midi_in:
            self.midi_in.close()
        print("Stop the MIDI listener. End of method from INSIDE")

    def _get_device_id_by_name(self, name):
        device_count = pygame.midi.get_count()
        for i in range(device_count):
            info = pygame.midi.get_device_info(i)
            if info[1].decode() == name and info[2] == 1:
                return i
        return None

    def __str__(self):
        return 'PygameMidiEventListener'
