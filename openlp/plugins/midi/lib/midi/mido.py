import mido
from openlp.plugins.midi.lib.midi.midi_listener_template import MidiListenerTemplate
from openlp.plugins.midi.lib.types_definitions.constants import openlp_midi_device, disabled_midi_device


class MidiEventListener(MidiListenerTemplate):
    def __init__(self, on_receive_callback_ref):
        super().__init__(on_receive_callback_ref)
        self.inport = None
        self.thread = None
        self.listening_is_active_flag = False

    def connect_to_device(self, device: str = None) -> bool:
        self.unpacked_midi_configuration()
        if device is None:
            device = self.midi_config.input_midi_device

        if device == disabled_midi_device['input']:
            self.receiver_disabled_flag = True
            return False

        # Create and connect to own virtual device
        if device == openlp_midi_device['gui_label']:
            try:
                self.inport = mido.open_input(name=openlp_midi_device['name'], virtual=True)  # TODO: hardcode some more
                print("MidiEventListener | Virtual MIDI device created.")
                return True
            except Exception as e:
                print(f"MidiEventListener | Failed to create virtual MIDI device. Error: {e}")
                return False

        try:
            self.inport = mido.open_input(device)  # Assuming 'device' is the name of the MIDI input port
            self.listening_is_active_flag = True
            print(f"MidiEventListener | Listening to MIDI port: {self.inport.name}")
            return True
        except IOError as e:
            print(f"MidiEventListener | Failed to connect to {device}. Error {e}")
            return False

    def listen_for_midi_events(self):
        # TODO: NOTE itis not tested if this helps mitigate any errors. The try bock might have to be removed.
        try:  # TODO: consider having extra redundancy in the other listener implementations
            while self.listening_is_active_flag:
                for msg in self.inport.iter_pending():
                    self.handle_midi_event(msg)
                self._async_pause()
        except Exception as e:
            print(f"MidiEventListener | Error: Midi Receiver error detected {e}")
            self.request_restart()

    def listen_for_a_single_midi_event(self):
        while self.listening_is_active_flag:
            for msg in self.inport.iter_pending():
                self._event_callback(msg)
                return  # Return after the message is processed
            self._async_pause(0.001)

    def stop(self):
        print("MidiEventListener| Request received to stop the MIDI listener.")
        self.request_exit()
        self.listening_is_active_flag = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.inport:
            self.inport.close()
        print("MidiEventListener | Stop the MIDI listener. End of method")

    def stop_from_inside(self):
        print("MidiEventListener | Request received to stop the MIDI listener from INSIDE.")
        self.request_exit()
        self.listening_is_active_flag = False
        self._async_pause(0.005)
        if self.inport:
            self.inport.close()
        print("MidiEventListener | Stop the MIDI listener. End of method from INSIDE")

    def __str__(self):
        return 'MidiEventListener-MIDO'
