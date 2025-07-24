# sb_ui.py

from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
import os
import textwrap

class TranscriptUI:
    def __init__(self):
        # UI sections
        self.partial_box = TextArea(style="class:partial", scrollbar=True, multiline=True, wrap_lines=True)
        self.final_box = TextArea(style="class:final", scrollbar=True, multiline=True, wrap_lines=True)

        # Style
        self.style = Style.from_dict({
            "partial": "fg:cyan",
            "final": "fg:green",
            "title": "bold underline",
        })

        # Keybindings
        self.kb = KeyBindings()
        self.kb.add("c-c")(self._exit)
        self.kb.add("c-s")(self._save)

        # History
        self.partial_history = []
        self.final_words = []

        # Layout
        self.layout = Layout(HSplit([
            TextArea(text="Partial Transcript", style="class:title", height=1, focusable=False),
            self.partial_box,
            TextArea(text="Final Transcript", style="class:title", height=1, focusable=False),
            self.final_box,
        ]))

        self.app = Application(layout=self.layout, key_bindings=self.kb, full_screen=True, style=self.style)

    def _exit(self, event):
        self._save_to_file("transcript.txt")
        event.app.exit()

    def _save(self, event):
        self._save_to_file("transcript_saved.txt")

    def _save_to_file(self, filename):
        os.makedirs("audiodata", exist_ok=True)
        path = os.path.join("audiodata", filename)
        with open(path, "w", encoding="utf-8") as f:
            wrapped = textwrap.fill(" ".join(self.final_words), width=80)
            f.write(wrapped)

    def update_partial(self, new_text):
        text = new_text.strip()
        if not text:
            return
        self.partial_history.append(text)
        if len(self.partial_history) > 10:
            self.partial_history.pop(0)
        self.partial_box.text = "\n".join(self.partial_history)
        self.partial_box.buffer.cursor_position = len(self.partial_box.text)

    def update_final(self, text):
        if not text:
            return
        words = text.strip().split()
        self.final_words = self._merge_overlap(self.final_words, words)
        joined = " ".join(self.final_words)
        self.final_box.text = joined
        self.final_box.buffer.cursor_position = len(joined)

    def _merge_overlap(self, current, incoming):
        overlap = 0
        for i in range(1, min(len(current), len(incoming)) + 1):
            if current[-i:] == incoming[:i]:
                overlap = i
        return current + incoming[overlap:]

    def run(self):
        self.app.run()

    def stop(self):
        self.app.exit()

# Optional: demo usage
if __name__ == "__main__":
    ui = TranscriptUI()
    from threading import Thread
    import time

    def simulate():
        samples = ["turn left heading", "left heading two one zero", "maintain flight level three two zero"]
        for phrase in samples:
            ui.update_partial(phrase)
            time.sleep(0.5)
            ui.update_final(phrase)
            time.sleep(0.5)

    Thread(target=simulate).start()
    ui.run()
