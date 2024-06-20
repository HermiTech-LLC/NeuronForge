import sys
from PyQt5.QtWidgets import QApplication
import threading
import uvicorn
from uvicorn import Config, Server
from app.app import App

def run_fastapi():
    """Function to run the FastAPI server."""
    config = Config("api:app", host="0.0.0.0", port=8000, reload=True)
    server = Server(config)
    server.run()

if __name__ == "__main__":
    # Start the FastAPI server in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    # Start the PyQt application
    app = QApplication(sys.argv)
    main_window = App()
    main_window.show()
    sys.exit(app.exec_())
