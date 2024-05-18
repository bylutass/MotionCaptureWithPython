import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import threading

class MainScreen:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Capture Program")
        self.root.configure(bg='dark grey')
        self.root.geometry("800x600")  # Ekran boyutunu 800x600 olarak ayarla

        # Ana çerçeve
        self.main_frame = tk.Frame(root, bg='dark grey', width=800, height=600)
        self.main_frame.pack(fill='both', expand=True)

        # Video oynatma alanı
        self.video_label = tk.Label(self.main_frame, bg='dark grey')
        self.video_label.pack()

        # "Add Video" butonu
        self.add_video_button = tk.Button(self.main_frame, text="Add Video", command=self.open_file_dialog)
        self.add_video_button.pack(side='bottom')

        self.video_playing = False  # Video oynuyor mu kontrolü için bayrak

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        if file_path:
            self.add_video_button.pack_forget()  # "Add Video" butonunu gizle
            self.video_thread = threading.Thread(target=self.display_video, args=(file_path,))
            self.video_thread.start()  # Videoyu arka plan iş parçacığında başlat

    def display_video(self, video_path):
        # Video dosyasını aç
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Video dosyası yüklenemedi.")
            return

        # MediaPipe Pose modelini yükle
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Video boyutlarını al
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Video'nun fps değerini al
        frame_duration = 1 / fps  # Her bir kare arasındaki süreyi hesapla

        self.video_playing = True  # Video oynuyor olarak işaretle

        # Videoyu görüntüle
        while self.video_playing:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe Pose modelini kullanarak pose tahmini yap
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            landmarks = results.pose_landmarks

            # Pose landmarks'larını çiz
            if landmarks:
                self.draw_pose(frame_rgb, landmarks)

            # Görüntüyü PIL formatına dönüştür
            pil_image = Image.fromarray(frame_rgb)

            # Görüntüyü ekran boyutuna uyacak şekilde yeniden boyutlandır
            resized_image = self.resize_image_to_fit(pil_image, (800, 600))

            # PIL görüntüsünü tkinter görüntüsüne dönüştür
            tk_image = ImageTk.PhotoImage(image=resized_image)

            # Görüntüyü etikete ekle
            self.video_label.config(image=tk_image)
            self.video_label.image = tk_image

            # Ekranı güncelle
            self.root.update()

            # Her bir karenin gösterilme süresini hesapla ve video hızını arttırmak için bekleyen süreyi ayarla
            frame_duration_ms = int(frame_duration * 1000 / 2)  # Video hızını artırmak için süreyi yarıya indir
            self.root.after(frame_duration_ms)  # Bir sonraki kareyi göstermek için bir süre bekle

        # Kullanılan kaynakları serbest bırak
        cap.release()

        # MediaPipe Pose modelini kapat
        pose.close()

    def draw_pose(self, frame, landmarks):
        # Görüntü üzerindeki pose landmarks'larını çizmek için kullanılacak renk ve kalınlık ayarları
        connections = mp.solutions.pose.POSE_CONNECTIONS
        landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

        # Pose landmarks'larını görüntü üzerinde çiz
        mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, connections, landmark_drawing_spec)

    def stop_video(self):
        self.video_playing = False  # Video oynatmayı durdur

    def resize_image_to_fit(self, image, target_size):
        # Görüntüyü hedef boyuta uyacak şekilde yeniden boyutlandır
        target_width, target_height = target_size
        width, height = image.size
        aspect_ratio = min(target_width / width, target_height / height)
        new_width = int(width * aspect_ratio)
        new_height = int(height * aspect_ratio)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Hedef boyutlarına uyacak şekilde arka plan oluştur
        background = Image.new('RGB', target_size, (0, 0, 0))
        offset = ((target_width - new_width) // 2, (target_height - new_height) // 2)
        background.paste(resized_image, offset)
        
        return background

if __name__ == "__main__":
    root = tk.Tk()
    mainScreen = MainScreen(root)
    root.mainloop()
