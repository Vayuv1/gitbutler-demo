"""Terminal UI helpers for displaying real-time ASR output."""

from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
import os
import textwrap
import re


class AudioMeter(TextArea):
    """Simple horizontal bar meter for audio input level."""

    def __init__(self, **kwargs):
        super().__init__(read_only=True, multiline=False, wrap_lines=False,
                         height=1, **kwargs)
        self.text = ""
        self.max_width = 78  # Adjust according to terminal width
        self.app = None

    def update_level(self, level: float) -> None:
        """Update the meter display based on average input amplitude."""
        bars = int(level * self.max_width)
        self.text = "▮" * bars
        if self.app:
            self.app.invalidate()


class TranscriptUI:
    """Interactive UI for partial and final transcription results."""

    def __init__(self) -> None:
        # Boxes used to display running text
        self.partial_box = TextArea(style="class:partial", scrollbar=True,
                                    multiline=True, wrap_lines=True)
        self.final_box = TextArea(style="class:final", scrollbar=True,
                                  multiline=True, wrap_lines=True)
        self.audio_meter = AudioMeter(style="class:meter")

        # Colours and headings for the UI
        self.style = Style.from_dict({
            "header": "bold underline",
            "partial": "fg:cyan",
            "final": "fg:green",
            "meter": "fg:yellow bg:black",
        })

        # Global keybindings
        self.kb = KeyBindings()
        self.kb.add("c-c")(self._exit)
        self.kb.add("c-s")(self._save)

        # Internal state
        self.partial_history: list[str] = []
        self.final_words: list[str] = []
        self.last_partial_display = "• ..."

        # Layout structure
        self.layout = Layout(HSplit([
            TextArea(text="\U0001F539 Partial Transcript Updates",
                     style="class:header", height=1, focusable=False),
            self.partial_box,
            TextArea(text="\U0001F538 Final Transcript Paragraph",
                     style="class:header", height=1, focusable=False),
            self.final_box,
            TextArea(text="\U0001F4C9 Audio Input Level",
                     style="class:header", height=1, focusable=False),
            self.audio_meter,
        ]))

        # Application instance
        self.app = Application(layout=self.layout, key_bindings=self.kb,
                               full_screen=True, style=self.style)

        # Widgets need access to the app to trigger redraws
        self.partial_box.app = self.app
        self.final_box.app = self.app
        self.audio_meter.app = self.app

    # ------------------------------------------------------------------
    # Keybinding handlers
    # ------------------------------------------------------------------
    def _exit(self, event) -> None:
        """Save transcript and exit the UI."""
        self._save_to_file("transcript.txt")
        event.app.exit()

    def _save(self, event) -> None:
        """Save transcript to a secondary file."""
        self._save_to_file("transcript_saved.txt")

    def _save_to_file(self, filename: str) -> None:
        """Write the final transcript to disk."""
        os.makedirs("audiodata", exist_ok=True)
        path = os.path.join("audiodata", filename)
        with open(path, "w", encoding="utf-8") as f:
            wrapped = textwrap.fill(" ".join(self.final_words), width=80)
            f.write(wrapped)

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------
    def update_partial(self, new_text: str) -> None:
        """Display the newest sentence fragment as a bullet point."""
        text = new_text.strip().lower()

        if not text:
            if not self.partial_box.text.strip():
                self.partial_box.text = self.last_partial_display
                self.partial_box.buffer.cursor_position = len(self.last_partial_display)
            return

        sentences = re.split(r"(?<=[.!?])\s+", text)
        last = sentences[-1].strip()
        if not last:
            return

        bullet = f"• {last}"
        if self.partial_history:
            prev = self.partial_history[-1]
            import difflib
            if difflib.SequenceMatcher(None, prev[2:], last).ratio() > 0.8:
                self.partial_history[-1] = bullet
            else:
                self.partial_history.append(bullet)
        else:
            self.partial_history.append(bullet)

        if len(self.partial_history) > 20:
            self.partial_history = self.partial_history[-20:]

        updated = "\n".join(self.partial_history)
        if updated != self.partial_box.text:
            self.partial_box.text = updated
            self.partial_box.buffer.cursor_position = len(updated)
            self.last_partial_display = updated
            self.app.invalidate()

    def update_final(self, text: str) -> None:
        """Merge new stabilized text into the final transcript."""
        if not text:
            return
        new_words = self._clean_words(text)
        attach = self._find_attachment_point(self.final_words, new_words)
        self.final_words = self.final_words[:attach] + new_words
        joined = self._format_paragraph(self.final_words)
        self.final_box.text = joined
        self.final_box.buffer.cursor_position = len(joined)
        self.app.invalidate()

    def update_level(self, level: float) -> None:
        """Forward audio level updates to the meter widget."""
        self.audio_meter.update_level(level)

    # ------------------------------------------------------------------
    # Text utilities
    # ------------------------------------------------------------------
    def _clean_words(self, text: str) -> list[str]:
        """Lowercase text and remove immediate duplicate words."""
        words = text.lower().split()
        cleaned: list[str] = []
        for w in words:
            if not cleaned or w != cleaned[-1]:
                cleaned.append(w)
        return cleaned

    @staticmethod
    def _find_attachment_point(existing: list[str], new: list[str]) -> int:
        """Find where to splice new words to avoid duplicate regions."""
        if not existing or not new:
            return len(existing)

        for length in range(len(new), 0, -1):
            prefix = new[:length]
            search = existing[-(len(new) + 10):]
            for i in range(len(search)):
                if search[i : i + length] == prefix:
                    return len(existing) - len(search) + i
        return len(existing)

    @staticmethod
    def _format_paragraph(words: list[str], per_sentence: int = 12) -> str:
        """Break a long list of words into simple sentences."""
        sentences: list[str] = []
        for i in range(0, len(words), per_sentence):
            sentence = " ".join(words[i : i + per_sentence]).strip()
            if sentence:
                sentences.append(sentence)
        if not sentences:
            return ""
        return ". ".join(sentences) + "."

    # ------------------------------------------------------------------
    # Run loop helpers
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start the prompt_toolkit application."""
        self.app.run()

    def stop(self) -> None:
        """Exit the prompt_toolkit application."""
        self.app.exit()


# ---------------------------------------------------------------------------
# Optional demonstration when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ui = TranscriptUI()
    from threading import Thread
    import time

    def simulate() -> None:
        samples = [
            "turn left heading",
            "left heading two one zero",
            "maintain flight level three two zero",
        ]
        for phrase in samples:
            ui.update_partial(phrase)
            time.sleep(0.5)
            ui.update_final(phrase)
            time.sleep(0.5)

    Thread(target=simulate).start()
    ui.run()