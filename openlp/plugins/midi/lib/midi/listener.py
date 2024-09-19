import time
from typing import Callable

from openlp.core.threading import ThreadWorker  # , run_thread, make_remove_thread # TODO: might be used, if not cleanup
from openlp.plugins.midi.lib.midi.mido import MidiEventListener as Selected_MidiEventListener

MidiEventListener = Selected_MidiEventListener


# Service <= ThreadWorker Class <= Listener Async class
class MidiEventListener_DISABLE(ThreadWorker):
    def __init__(self, event_callback_reference: Callable):
        ThreadWorker.__init__(self)
        self.worker_thread = Selected_MidiEventListener(on_receive_callback_ref=event_callback_reference)
        self.running = False

    def start(self):
        self.running = True
        self.worker_thread.start()
        while self.running:
            time.sleep(0.1)  # Adjust the sleep time as needed
        print("Exit WORKER THREAD START METHOD")

    def stop(self):
        self.running = False
        time.sleep(0.2)
        self.worker_thread.stop()
        print("IN WORKER THREAD METHOD STOP ")
        time.sleep(2)
        print("IN WORKER THREAD METHOD STOP after pause")

    def __str__(self):
        return 'MidiEventListener_INSEPTION'
