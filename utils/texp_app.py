import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget, QLabel, QPushButton, QTextEdit, QFileDialog, QMessageBox, QLineEdit, QFormLayout
)
from PyQt5.QtGui import QIcon
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from collections import Counter
import re

def apply_styles(widget):
    """Apply the defined styles to the widget."""
    widget.setStyleSheet("""
        QWidget {
            background-color: black;
            color: lime;
            font-family: "Courier New";
            font-size: 14px;
            font-weight: bold;
        }
        QPushButton {
            background-color: #333;
            color: lime;
            border: 1px solid lime;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #444;
        }
        QLineEdit, QTextEdit {
            background-color: #222;
            color: lime;
            border: 1px solid lime;
        }
        QLabel {
            color: lime;
        }
        QTabWidget::pane {
            border: 1px solid lime;
        }
        QTabBar::tab {
            background: #333;
            color: lime;
            padding: 10px;
        }
        QTabBar::tab:selected {
            background: #444;
            border-bottom: 2px solid lime;
        }
    """)

class TexPApp(QWidget):
    def __init__(self):
        """Initialize the main application window."""
        super().__init__()
        self.text_data = ""
        self.processed_data = ""
        self.init_ui()

    def init_ui(self):
        """Set up the user interface."""
        self.setWindowTitle('TexP - Text Processing App')
        self.setGeometry(300, 200, 800, 600)  # Set the window to open smaller and centered

        layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_data_preprocessing_tab(), "Data Preprocessing")
        self.tabs.addTab(self.create_text_analysis_tab(), "Text Analysis")
        self.tabs.addTab(self.create_visualization_tab(), "Visualization")
        self.tabs.addTab(self.create_csv_organization_tab(), "CSV Organization")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

        apply_styles(self)

    def create_data_preprocessing_tab(self):
        """Create the Data Preprocessing tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        load_data_button = QPushButton("Load Text Document")
        load_data_button.clicked.connect(self.load_text_data)
        process_data_button = QPushButton("Process Text")
        process_data_button.clicked.connect(self.process_text_data)
        self.data_output = QTextEdit()
        self.data_output.setReadOnly(True)

        layout.addWidget(QLabel("Data Preprocessing"))
        layout.addWidget(load_data_button)
        layout.addWidget(process_data_button)
        layout.addWidget(self.data_output)

        tab.setLayout(layout)
        return tab

    def create_text_analysis_tab(self):
        """Create the Text Analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        analyze_text_button = QPushButton("Analyze Text")
        analyze_text_button.clicked.connect(self.analyze_text_data)
        self.analysis_output = QTextEdit()
        self.analysis_output.setReadOnly(True)

        layout.addWidget(QLabel("Text Analysis"))
        layout.addWidget(analyze_text_button)
        layout.addWidget(self.analysis_output)

        tab.setLayout(layout)
        return tab

    def create_visualization_tab(self):
        """Create the Visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        visualize_data_button = QPushButton("Visualize Data")
        visualize_data_button.clicked.connect(self.visualize_text_data)
        self.visualization_output = QTextEdit()
        self.visualization_output.setReadOnly(True)

        layout.addWidget(QLabel("Visualization"))
        layout.addWidget(visualize_data_button)
        layout.addWidget(self.visualization_output)

        tab.setLayout(layout)
        return tab

    def create_csv_organization_tab(self):
        """Create the CSV Organization tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        form_layout = QFormLayout()
        self.csv_filename_input = QLineEdit()
        self.csv_columns_input = QLineEdit()
        self.csv_separator_input = QLineEdit()
        self.csv_separator_input.setText(',')

        form_layout.addRow("CSV Filename:", self.csv_filename_input)
        form_layout.addRow("Columns (comma-separated):", self.csv_columns_input)
        form_layout.addRow("Separator:", self.csv_separator_input)

        save_csv_button = QPushButton("Save to CSV")
        save_csv_button.clicked.connect(self.save_to_csv)

        self.csv_output = QTextEdit()
        self.csv_output.setReadOnly(True)

        layout.addLayout(form_layout)
        layout.addWidget(save_csv_button)
        layout.addWidget(self.csv_output)

        tab.setLayout(layout)
        return tab

    def load_text_data(self):
        """Load text data from a file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Text Document", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.text_data = file.read()
                    self.data_output.setText(self.text_data)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def process_text_data(self):
        """Process the loaded text data."""
        if not self.text_data:
            QMessageBox.warning(self, "Warning", "No text document loaded.")
            return

        processed_text = self.text_data.lower()
        processed_text = re.sub(r'\d+', '', processed_text)
        processed_text = re.sub(r'[^\w\s]', '', processed_text)
        processed_text = re.sub(r'\s+', ' ', processed_text)
        self.processed_data = processed_text.strip()
        self.data_output.setText(self.processed_data[:1000])  # Display the first 1000 characters

    def analyze_text_data(self):
        """Analyze the processed text data."""
        if not self.processed_data:
            QMessageBox.warning(self, "Warning", "No processed text available.")
            return

        word_count = Counter(self.processed_data.split())
        self.analysis_output.setText(str(word_count.most_common(10)))

    def visualize_text_data(self):
        """Visualize the processed text data."""
        if not self.processed_data:
            QMessageBox.warning(self, "Warning", "No processed text available.")
            return

        word_count = Counter(self.processed_data.split())
        common_words = word_count.most_common(10)
        words, counts = zip(*common_words)

        plt.figure(figsize=(10, 6))
        plt.bar(words, counts)
        plt.title("Top 10 Most Common Words")
        plt.xlabel("Words")
        plt.ylabel("Counts")

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        self.visualization_output.setHtml(f'<img src="data:image/png;base64,{img_base64}">')

    def save_to_csv(self):
        """Save the processed data to a CSV file based on user-defined parameters."""
        if not self.processed_data:
            QMessageBox.warning(self, "Warning", "No processed text available.")
            return

        filename = self.csv_filename_input.text().strip()
        columns = [col.strip() for col in self.csv_columns_input.text().split(',')]
        separator = self.csv_separator_input.text().strip()

        if not filename.endswith('.csv'):
            filename += '.csv'

        word_count = Counter(self.processed_data.split())
        data = {column: [] for column in columns}

        for word, count in word_count.items():
            if word in data:
                data[word].append(count)

        # Ensure all lists in data have the same length
        max_len = max(len(lst) for lst in data.values())
        for key, lst in data.items():
            while len(lst) < max_len:
                lst.append(0)

        df = pd.DataFrame(data)
        try:
            df.to_csv(filename, sep=separator, index=False)
            self.csv_output.setText(f"Data saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save CSV file: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('icon.png'))  # Set an appropriate icon if available
    tex_p_app = TexPApp()
    tex_p_app.show()
    sys.exit(app.exec_())
