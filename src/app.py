import sys
from PyQt6.QtWidgets import QApplication

from gui import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Optimization Algorithms")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
