# Vii-assistant
A prototype for an assistant that does computer-use tasks, using screen capture and synthetic user actions.

Mainly the app now has a transcription (STT) service that's useful - you can toggle voice recording with a shortcut, and have the transcribed text be pasted (via ydotool) at the current cursor position. Transcription is done continuously, so the wait time is max 10-15 secs.

# Install instructions
TBD. There's a pyproject in `assistant/`. You have to run the inference server and the desktop app separately (they can be on different machines )

# Credits
Thank you to the developers of the (Handy)[https://github.com/cjpais/Handy] app which heavily inspired the transcription feature in vii.
