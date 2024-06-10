def apply_styles(widget):
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
        QLineEdit, QTextEdit, QComboBox {
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
