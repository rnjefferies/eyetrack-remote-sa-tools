# ============================================================================
# Manual_Sync_Tool.py  —  manual video/audio synchronisation and event labelling
# ============================================================================
# Purpose:  Align the scene/gaze video with the audio recording on a common
#           timeline and mark/label events, preserving AI-detected events.
# Inputs:   scene video (.mp4), audio recording (.wav), event CSV
# Outputs:  updated event CSV with synchronised, labelled markers
# Usage:    python Manual_Sync_Tool.py   (PyQt GUI)
# Requires: PyQt5, pandas, numpy, scipy, matplotlib
# Part of:  EyeTrack Remote-SA Tools (see repo README). Contains no data.
# ============================================================================

import sys, os, pandas as pd, re, numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QWidget, QFileDialog, QLabel, QSlider, 
                             QShortcut, QInputDialog, QFrame, QDialog, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QAbstractItemView)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl

# --- Waveform Component ---
class WaveformCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(5, 1), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.set_axis_off()
        self.fig.patch.set_facecolor('#f0f0f0')
        self.playhead = None
        self.num_points = 0

    def plot_wav(self, wav_path):
        try:
            self.ax.clear()
            fs, data = wavfile.read(wav_path)
            if len(data.shape) > 1: data = data[:, 0]
            if len(data) == 0: return
            
            step = max(1, len(data) // 5000)
            downsampled = data[::step]
            self.num_points = len(downsampled)
            
            self.ax.plot(downsampled, color='#e67e22', linewidth=0.5) 
            self.ax.set_axis_off()
            
            # Create a red vertical line at position 0
            self.playhead = self.ax.axvline(x=0, color='red', linewidth=2)
            
            self.fig.tight_layout()
            self.draw()
        except Exception as e:
            print(f"❌ Waveform Error: {e}")

    def update_playhead(self, ratio):
        # Move the red line across the graph based on audio percentage
        if self.playhead and self.num_points > 0:
            x_pos = ratio * self.num_points
            self.playhead.set_xdata([x_pos, x_pos])
            self.draw_idle()

# --- Main Application ---
class PrecisionMapper(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeTrack Sync & Mapper (Preserves AI Events)")
        self.setGeometry(100, 100, 1400, 950)
        
        self.video_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.audio_player = QMediaPlayer(None)
        
        # State
        self.active_focus = "video"
        self.video_clap_ms = 0
        self.audio_clap_ms = 0
        self.manual_markers = []
        self.target_folder = ""
        self.start_ns = 0
        self.rec_id = ""

        self.init_ui()
        self.setup_shortcuts()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # --- VIDEO SECTION ---
        v_box = QVBoxLayout()
        self.video_widget = QVideoWidget()
        v_box.addWidget(self.video_widget, stretch=5)
        
        self.v_slider = QSlider(Qt.Horizontal)
        self.v_slider.sliderMoved.connect(self.video_player.setPosition)
        v_box.addWidget(self.v_slider)
        
        v_ctrls = QHBoxLayout()
        self.lbl_v_time = QLabel("Video: 0.00s")
        self.btn_mark_v = QPushButton("📍 Set Video Clap Point (Optional)")
        self.btn_mark_v.clicked.connect(self.mark_v)
        self.lbl_v_status = QLabel("Clap: Not Set")
        v_ctrls.addWidget(self.lbl_v_time)
        v_ctrls.addWidget(self.btn_mark_v)
        v_ctrls.addWidget(self.lbl_v_status)
        v_box.addLayout(v_ctrls)
        main_layout.addLayout(v_box)

        main_layout.addWidget(QFrame(frameShape=QFrame.HLine))

        # --- AUDIO SECTION ---
        a_box = QVBoxLayout()
        self.waveform = WaveformCanvas(self)
        a_box.addWidget(self.waveform, stretch=1)
        
        self.a_slider = QSlider(Qt.Horizontal)
        self.a_slider.sliderMoved.connect(self.audio_player.setPosition)
        a_box.addWidget(self.a_slider)
        
        a_ctrls = QHBoxLayout()
        self.lbl_a_time = QLabel("App Audio: 0.00s")
        self.btn_mark_a = QPushButton("📍 Set Audio Clap Point (Optional)")
        self.btn_mark_a.clicked.connect(self.mark_a)
        self.lbl_a_status = QLabel("Clap: Not Set")
        a_ctrls.addWidget(self.lbl_a_time)
        a_ctrls.addWidget(self.btn_mark_a)
        a_ctrls.addWidget(self.lbl_a_status)
        a_box.addLayout(a_ctrls)
        main_layout.addLayout(a_box)

        # --- GLOBAL UI ---
        footer = QHBoxLayout()
        self.lbl_focus = QLabel("FOCUS: VIDEO (Shortcuts control ET Glasses)")
        self.lbl_focus.setStyleSheet("background-color: #2980b9; color: white; padding: 5px; font-weight: bold;")
        
        btn_load = QPushButton("📂 LOAD FOLDER")
        btn_load.clicked.connect(self.load_folder)

        self.btn_view = QPushButton("📋 VIEW EVENTS")
        self.btn_view.setEnabled(False)
        self.btn_view.clicked.connect(self.show_events_popout)
        
        self.btn_save = QPushButton("💾 SAVE MARKERS to events_with_AI.csv")
        self.btn_save.setEnabled(False) 
        self.btn_save.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_all)

        footer.addWidget(self.lbl_focus)
        footer.addWidget(btn_load)
        footer.addWidget(self.btn_view)
        footer.addWidget(self.btn_save)
        main_layout.addLayout(footer)

        self.video_player.positionChanged.connect(self.update_v_ui)
        self.video_player.durationChanged.connect(lambda d: self.v_slider.setRange(0, d))
        self.audio_player.positionChanged.connect(self.update_a_ui)
        self.audio_player.durationChanged.connect(lambda d: self.a_slider.setRange(0, d))

        c = QWidget(); c.setLayout(main_layout); self.setCentralWidget(c)

    def setup_shortcuts(self):
        QShortcut(Qt.Key_Space, self, self.toggle_play)
        QShortcut(Qt.Key_Tab, self, self.swap_focus)
        
        QShortcut(Qt.Key_D, self, lambda: self.nudge(33))
        QShortcut(Qt.Key_A, self, lambda: self.nudge(-33))
        QShortcut(Qt.Key_S, self, lambda: self.nudge(1000))
        QShortcut(Qt.Key_W, self, lambda: self.nudge(-1000))

        QShortcut(Qt.Key_I, self, lambda: self.add_marker("video_in"))
        QShortcut(Qt.Key_O, self, lambda: self.add_marker("video_out"))
        QShortcut(Qt.Key_M, self, self.prompt_marker)

    def swap_focus(self):
        self.video_player.pause(); self.audio_player.pause()
        if self.active_focus == "video":
            self.active_focus = "audio"
            self.lbl_focus.setText("FOCUS: APP AUDIO (Shortcuts control Waveform)")
            self.lbl_focus.setStyleSheet("background-color: #d35400; color: white; padding: 5px;")
        else:
            self.active_focus = "video"
            self.lbl_focus.setText("FOCUS: VIDEO (Shortcuts control ET Glasses)")
            self.lbl_focus.setStyleSheet("background-color: #2980b9; color: white; padding: 5px;")

    def toggle_play(self):
        p = self.video_player if self.active_focus == "video" else self.audio_player
        if p.state() == QMediaPlayer.PlayingState: p.pause()
        else: p.play()

    def nudge(self, ms):
        p = self.video_player if self.active_focus == "video" else self.audio_player
        p.setPosition(p.position() + ms)

    def mark_v(self):
        self.video_clap_ms = self.video_player.position()
        self.lbl_v_status.setText(f"Clap: {self.video_clap_ms/1000.0:.3f}s")

    def mark_a(self):
        self.audio_clap_ms = self.audio_player.position()
        self.lbl_a_status.setText(f"Clap: {self.audio_clap_ms/1000.0:.3f}s")

    def update_v_ui(self, p):
        self.v_slider.setValue(p); self.lbl_v_time.setText(f"Video: {p/1000.0:.3f}s")
        
    def update_a_ui(self, p):
        self.a_slider.setValue(p)
        self.lbl_a_time.setText(f"App Audio: {p/1000.0:.3f}s")
        # Update the red playhead on the waveform
        duration = self.audio_player.duration()
        if duration > 0:
            self.waveform.update_playhead(p / duration)

    def add_marker(self, name):
        if not self.target_folder: return
        v_sec = self.video_player.position() / 1000.0
        ns = int(self.start_ns + (v_sec * 1e9))
        self.manual_markers.append({
            'recording id': self.rec_id, 
            'timestamp [ns]': ns, 
            'name': name, 
            'type': 'cloud', 
            'AI_Heard': 'MANUAL_MAP'
        })
        print(f"📍 Marker {name} at {v_sec:.3f}s")
        if hasattr(self, 'events_dialog') and self.events_dialog.isVisible():
            self.show_events_popout()

    def prompt_marker(self):
        self.video_player.pause()
        name, ok = QInputDialog.getText(self, "Manual Marker", "Name (e.g., pedal_in):")
        if ok and name.strip(): self.add_marker(name.strip())

    # --- POPOUT LOGIC ---
    def show_events_popout(self):
        if not self.target_folder: return
        
        path = os.path.join(self.target_folder, "events_with_AI.csv")
        if not os.path.exists(path):
            path = os.path.join(self.target_folder, "events.csv")
            
        df = pd.read_csv(path)
        
        if self.manual_markers:
            df = pd.concat([df, pd.DataFrame(self.manual_markers)], ignore_index=True)
        
        df['sec'] = (df['timestamp [ns]'] - self.start_ns) / 1e9
        df = df.sort_values('sec')
        
        if not hasattr(self, 'events_dialog'):
            self.events_dialog = QDialog(self)
            self.events_dialog.setWindowTitle("Timeline Events")
            self.events_dialog.setGeometry(100, 100, 500, 600)
            self.events_layout = QVBoxLayout()
            
            self.events_table = QTableWidget()
            self.events_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.events_table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.events_layout.addWidget(self.events_table)
            
            # --- NEW: Delete Button ---
            self.btn_delete_event = QPushButton("🗑️ Delete Selected Event")
            self.btn_delete_event.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; padding: 10px;")
            self.btn_delete_event.clicked.connect(self.delete_selected_event)
            self.events_layout.addWidget(self.btn_delete_event)
            
            self.events_dialog.setLayout(self.events_layout)

        self.events_dialog.setWindowTitle(f"Timeline Events - {os.path.basename(self.target_folder)}")
        table = self.events_table
        table.clear()
        
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Time (s)", "Marker Name", "AI / Data"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        table.setRowCount(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            # Store the raw nanosecond timestamp invisibly in the time cell for exact deletion
            item_time = QTableWidgetItem(f"{row['sec']:.2f}s")
            item_time.setData(Qt.UserRole, int(row['timestamp [ns]']))
            table.setItem(i, 0, item_time)
            
            table.setItem(i, 1, QTableWidgetItem(str(row['name'])))
            
            data_val = str(row.get('AI_Heard', ''))
            item_data = QTableWidgetItem(data_val)
            if data_val != "nan" and data_val:
                item_data.setBackground(Qt.yellow if "MANUAL" not in data_val else Qt.cyan)
                
            table.setItem(i, 2, item_data)
            
        self.events_dialog.show()

    def delete_selected_event(self):
        selected_items = self.events_table.selectedItems()
        if not selected_items: return
        
        row_idx = selected_items[0].row()
        ns_to_delete = self.events_table.item(row_idx, 0).data(Qt.UserRole)
        name_to_delete = self.events_table.item(row_idx, 1).text()
        
        # 1. Remove from temporary manual markers
        self.manual_markers = [m for m in self.manual_markers if not (m['timestamp [ns]'] == ns_to_delete and m['name'] == name_to_delete)]
        
        # 2. Remove from actual saved CSV if it exists
        ai_path = os.path.join(self.target_folder, "events_with_AI.csv")
        if os.path.exists(ai_path):
            df = pd.read_csv(ai_path)
            # Filter out the specific row
            df = df[~((df['timestamp [ns]'] == ns_to_delete) & (df['name'] == name_to_delete))]
            df.to_csv(ai_path, index=False)
            print(f"🗑️ Permanently deleted '{name_to_delete}' from {ai_path}")
            
        self.show_events_popout() # Refresh the view

    def load_folder(self):
        self.target_folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not self.target_folder: return
        vids = [f for f in os.listdir(self.target_folder) if f.endswith(".mp4")]
        auds = [f for f in os.listdir(self.target_folder) if f.endswith(".wav")]
        if vids and auds:
            base_df = pd.read_csv(os.path.join(self.target_folder, "events.csv"))
            self.rec_id = base_df['recording id'].iloc[0]
            self.start_ns = base_df[base_df['name'] == 'recording.begin']['timestamp [ns]'].iloc[0]
            self.video_player.setVideoOutput(self.video_widget)
            self.video_player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.join(self.target_folder, vids[0]))))
            self.audio_player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.join(self.target_folder, auds[0]))))
            self.waveform.plot_wav(os.path.join(self.target_folder, auds[0]))
            
            self.btn_save.setEnabled(True)
            self.btn_view.setEnabled(True) 
            self.manual_markers = [] 
            print(f"Loaded {self.target_folder}. Ready to add markers.")

    def save_all(self):
        ai_file_path = os.path.join(self.target_folder, "events_with_AI.csv")
        orig_file_path = os.path.join(self.target_folder, "events.csv")
        
        if os.path.exists(ai_file_path):
            final_df = pd.read_csv(ai_file_path)
        else:
            final_df = pd.read_csv(orig_file_path)

        if self.video_clap_ms > 0 and self.audio_clap_ms > 0:
            offset_sec = (self.video_clap_ms - self.audio_clap_ms) / 1000.0
            
            if os.path.exists("flagged_events.csv"):
                df_flags = pd.read_csv("flagged_events.csv")
                folder_name = os.path.basename(self.target_folder)
                match = re.search(r'R(\d+)_(\d+)', folder_name)
                
                if match:
                    route_num, part_id = int(match.group(1)), f"{match.group(2)}_RT"
                    sess_flags = df_flags[(df_flags['Participant ID'] == part_id) & (df_flags['Route'] == route_num)]
                    
                    ai_entries = []
                    for _, row in sess_flags.iterrows():
                        e_id = str(row['Event ID'])
                        e_num_match = re.search(r'\d+', e_id)
                        e_num = e_num_match.group() if e_num_match else "X"
                        q_name = f"Q{e_num}_Q"

                        q_video_sec = row['Question Timestamp (s)'] + offset_sec
                        q_ns = int(self.start_ns + (q_video_sec * 1e9))
                        
                        if not ((final_df['timestamp [ns]'] == q_ns) & (final_df['name'] == q_name)).any():
                            ai_entries.append({
                                'recording id': self.rec_id,
                                'timestamp [ns]': q_ns,
                                'name': q_name,
                                'type': 'cloud',
                                'AI_Heard': 'RECALCULATED_SYNC'
                            })
                    
                    if ai_entries:
                        final_df = pd.concat([final_df, pd.DataFrame(ai_entries)], ignore_index=True)

        if self.manual_markers:
            manual_df = pd.DataFrame(self.manual_markers)
            final_df = pd.concat([final_df, manual_df], ignore_index=True)

        final_df = final_df.sort_values('timestamp [ns]').drop_duplicates(subset=['timestamp [ns]', 'name'])
        final_df.to_csv(ai_file_path, index=False)
        
        self.lbl_v_status.setText(f"💾 SAVED! Added {len(self.manual_markers)} manual markers.")
        print(f"Success: Updated {ai_file_path}")
        
        self.manual_markers = []
        if hasattr(self, 'events_dialog') and self.events_dialog.isVisible():
            self.show_events_popout() 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = PrecisionMapper()
    ex.show()
    sys.exit(app.exec_())