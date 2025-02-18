import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog, QLabel, QVBoxLayout,
    QProgressBar, QTextEdit, QGridLayout
)
from PyQt5.QtGui import QPixmap, QFont, QMovie, QCursor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image



class ModelLoader(QThread):
    """Loads the AI model in a separate thread to prevent UI freezing."""
    model_loaded = pyqtSignal(object, object)  # Signal to send model & processor

    def run(self):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model_loaded.emit(processor, model)  # Send back the model when done


class SplashScreen(QLabel):
    def __init__(self, gif_path):
        super().__init__()

        # Make splash screen borderless and transparent
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)  

        # Load GIF
        movie = QMovie(gif_path)
        self.setMovie(movie)
        movie.start()

        self.setAlignment(Qt.AlignCenter)  # Center the GIF within the label
        self.adjustSize()

        # Center splash screen on the monitor
        self.centerOnScreen()

    def centerOnScreen(self):
        screen_rect = QApplication.primaryScreen().geometry()
        splash_size = self.sizeHint()  # Get splash size based on GIF

        x = (screen_rect.width() - splash_size.width()) // 2
        y = (screen_rect.height() - splash_size.height()) // 2

        self.setGeometry(QRect(x, y, splash_size.width(), splash_size.height()))


class ImageCaptionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.processor = None
        self.model = None

        # Show Transparent, Centered Splash Screen
        self.splash = SplashScreen("loading.gif")
        self.splash.show()

        # Load Model in Background Thread
        self.loader_thread = ModelLoader()
        self.loader_thread.model_loaded.connect(self.onModelLoaded)
        self.loader_thread.start()

    def onModelLoaded(self, processor, model):
        """Called when the model finishes loading."""
        self.processor = processor
        self.model = model
        self.splash.close()  # Close splash screen
        self.initUI()  # Launch UI

    def initUI(self):
        self.setWindowTitle("AI Image Caption Generator")
        self.setFixedSize(800, 600)
        self.setStyleSheet(self.getStyleSheet())

        # File Picker Button
        self.btn_select = QPushButton("📁 Select Images", self)
        self.btn_select.setCursor((QCursor(Qt.CursorShape.PointingHandCursor)))
        self.btn_select.clicked.connect(self.openFilePicker)

        # Progress Bar
        self.progress = QProgressBar(self)
        self.progress.setValue(0)
        self.progress.setStyleSheet("""
            QProgressBar {
                border-radius: 10px;
                background-color: #444;
                height: 15px;
            }
            QProgressBar::chunk {
                background-color: #29A19C;
                border-radius: 10px;
            }
        """)

        # Start Processing Button
        self.btn_generate = QPushButton("Generate Captions", self)
        self.btn_generate.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_generate.clicked.connect(self.processImages)
        self.btn_generate.setEnabled(False)

        # Grid Layout for Images + Captions
        self.grid_layout = QGridLayout()

        # Main Layout
        layout = QVBoxLayout()
        layout.addWidget(self.btn_select)
        layout.addWidget(self.progress)
        layout.addWidget(self.btn_generate)
        layout.addLayout(self.grid_layout)
        self.setLayout(layout)

        self.image_paths = []

        self.show()  # Show the main UI after loading

    def openFilePicker(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select up to 5 Images", "", "Images (*.png *.jpg *.jpeg)")
        
        if files:
            self.image_paths = files[:5]  # Limit to 5 images
            self.populateGrid()
            self.btn_generate.setEnabled(True)

    def populateGrid(self):
        """Clears the grid and populates it with selected images."""
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, img_path in enumerate(self.image_paths):
            # Image Preview
            img_label = QLabel(self)
            pixmap = QPixmap(img_path).scaled(150, 150, Qt.KeepAspectRatio)
            img_label.setPixmap(pixmap)

            # Caption Placeholder
            caption_label = QTextEdit(self)
            caption_label.setReadOnly(True)
            caption_label.setFixedHeight(50)
            caption_label.setFont(QFont("Arial", 10))

            self.grid_layout.addWidget(img_label, i, 0)
            self.grid_layout.addWidget(caption_label, i, 1)

    def processImages(self):
        if not self.image_paths:
            return

        self.progress.setMaximum(len(self.image_paths))

        for i, img_path in enumerate(self.image_paths):
            caption = self.generateCaption(img_path)

            caption_widget = self.grid_layout.itemAtPosition(i, 1).widget()
            caption_widget.setText(caption)

            self.progress.setValue(i + 1)

    def generateCaption(self, img_path):
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        
        with torch.no_grad():
            output = self.model.generate(**inputs)

        return self.processor.decode(output[0], skip_special_tokens=True)

    def getStyleSheet(self):
        return """
            QWidget {
                background-color: #1E1E1E;
                color: #E0E0E0;
                font-family: 'Arial';
            }

            QPushButton {
                background-color: #29A19C;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #257F76;
            }

            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }

            QTextEdit {
                background-color: #2E2E2E;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 6px;
                color: #E0E0E0;
            }
        """


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ImageCaptionApp()
    sys.exit(app.exec_())
