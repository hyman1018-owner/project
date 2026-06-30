"""AutoCamTracker main entry point."""

import sys
import tkinter as tk
from autocamtracker.ui.app import AutoCamTrackerApp, AppConfig

def main() -> None:
    root = tk.Tk()
    app = AutoCamTrackerApp(root, AppConfig())
    if len(sys.argv) > 1:
        app.input_config.source_type = "video_file"
        app.input_config.video_path = sys.argv[1]
    root.mainloop()

if __name__ == "__main__":
    main()
