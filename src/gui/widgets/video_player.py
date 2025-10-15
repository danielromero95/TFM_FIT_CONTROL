# src/gui/widgets/video_player.py
import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QStyle
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt

class VideoPlayerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        video_widget = QVideoWidget()

        self.play_button = QPushButton()
        self.play_button.setEnabled(False)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_play)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.sliderMoved.connect(self.set_position)

        # --- CAMBIO CLAVE: Conectamos la señal de error ---
        self.media_player.error.connect(self.handle_error)
        
        self.media_player.stateChanged.connect(self.update_play_button_icon)
        self.media_player.positionChanged.connect(self.update_slider_position)
        self.media_player.durationChanged.connect(self.update_slider_range)

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.position_slider)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(video_widget)
        main_layout.addLayout(controls_layout)
        
        self.media_player.setVideoOutput(video_widget)

    def load_video(self, video_path: str):
        if video_path and os.path.exists(video_path):
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
            self.play_button.setEnabled(True)
            self.media_player.play()
        else:
            self.play_button.setEnabled(False)
            self.media_player.setMedia(QMediaContent())

    # --- Método para imprimir errores ---
    def handle_error(self):
        if self.media_player.error() != QMediaPlayer.NoError:
            print(f"ERROR DEL REPRODUCTOR: {self.media_player.errorString()}")

    def toggle_play(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def update_play_button_icon(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def set_position(self, position):
        self.media_player.setPosition(position)
    def update_slider_position(self, position):
        self.position_slider.setValue(position)
    def update_slider_range(self, duration):
        self.position_slider.setRange(0, duration)