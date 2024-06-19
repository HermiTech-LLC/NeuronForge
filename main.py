import sys
from PyQt5.QtWidgets import QApplication
from app.app import App
import threading
import uvicorn

def run_fastapi():
    """Function to run the FastAPI server."""
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    # Start the FastAPI server in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    # Start the PyQt application
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
