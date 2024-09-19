import rtmidi
from collections import namedtuple
import sys
from openlp.plugins.midi.lib.midi.midi_listener_template import MidiListenerTemplate
from openlp.plugins.midi.lib.types_definitions.constants import openlp_midi_device, disabled_midi_device

# Define a simple MIDI message structure similar to Mido
MidiMessage = namedtuple('MidiMessage', ['type', 'note', 'velocity', 'channel'])


def convert_rtmidi_to_mido(rtmidi_data):
    """
    Convert rtmidi MIDI data to a format similar to Mido's Message.
    """
    midi_data, _ = rtmidi_data
    status_byte = midi_data[0]
    data_bytes = midi_data[1:]

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
    # TODO: here and in py game we have a similar thing.
    # TODO...: We should unify booth methods and put the in the MIDI definitions class

    # Assuming standard Note On/Off messages with two data bytes: note and velocity
    note = data_bytes[0] if len(data_bytes) > 0 else None
    velocity = data_bytes[1] if len(data_bytes) > 1 else None

    return MidiMessage(type=message_type, note=note, velocity=velocity, channel=channel)


class MidiEventListener(MidiListenerTemplate):
    def __init__(self, on_receive_callback_ref):
        super().__init__(on_receive_callback_ref)
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

        self.midi_in = rtmidi.MidiIn()
        available_ports = self.midi_in.get_ports()

        # Virtual port creation (if supported)
        if device == openlp_midi_device['gui_label']:
            if sys.platform != "win32":  # Virtual ports not supported on Windows
                try:
                    self.midi_in.open_virtual_port(openlp_midi_device['name'])
                    print("Virtual MIDI device created.")
                    self.listening_is_active_flag = True
                    return True
                except Exception as e:
                    print(f"Failed to create virtual MIDI device. Error: {e}")
                    return False
            else:
                print("Virtual MIDI ports are not supported on Windows.")
                return False

        # Connect to a physical MIDI device
        if device in available_ports:
            try:
                port_index = available_ports.index(device)
                self.midi_in.open_port(port_index)
                self.listening_is_active_flag = True
                print(f"Listening to MIDI port: {device}")
                return True
            except Exception as e:
                print(f"Failed to connect to {device}. Error: {e}")
                return False
        else:
            print(f"Device '{device}' not found.")
            return False

    def listen_for_midi_events(self):
        while self.listening_is_active_flag:
            rtmidi_message = self.midi_in.get_message()
            if rtmidi_message:
                self.handle_midi_event(convert_rtmidi_to_mido(rtmidi_message))
            self._async_pause(1 / 1000)  # 1ms

    def listen_for_a_single_midi_event(self):
        while self.listening_is_active_flag:
            rtmidi_message = self.midi_in.get_message()
            if rtmidi_message:
                self._event_callback(convert_rtmidi_to_mido(rtmidi_message))
                return  # Return after the message is processed
            self._async_pause(1 / 1000)  # 1ms

    def stop(self):
        print("Request received to stop the MIDI listener.")
        self.request_exit()
        self.listening_is_active_flag = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.midi_in:
            self.midi_in.close_port()
        print("Stop the MIDI listener. End of method")

    def stop_from_inside(self):
        print("Request received to stop the MIDI listener from INSIDE.")
        self.request_exit()
        self.listening_is_active_flag = False
        self._async_pause(0.02)
        if self.midi_in:
            self.midi_in.close_port()
        print("Stop the MIDI listener. End of method from INSIDE")

    def __str__(self):
        return 'PythonRtMidiEventListener'
