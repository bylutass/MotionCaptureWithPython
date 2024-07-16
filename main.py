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
        self.root.geometry("800x600")

        # Arka plan resmi
        self.bg_image = Image.open("bc.png")  # Kullanmak istediğiniz resmin dosya yolu
        self.bg_image = self.bg_image.resize((800, 600), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        # Tuval üzerine arka plan resmini yerleştirme
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor='nw')

        # Başlık etiketi
        self.title_label = tk.Label(root, text="Basic Motion Capture Program", bg='dark grey', fg='black', font=('Arial', 24, 'bold'))
        self.title_label.place(relx=0.5, rely=0.1, anchor='center')

        # Video oynatma alanı
        self.video_label = tk.Label(root, bg='dark grey')
        self.canvas.create_window(400, 300, window=self.video_label)

        # "Add Video" butonu
        self.add_video_button = tk.Button(root, text="Add Video", command=self.open_file_dialog, bg='black', fg='white', font=('Arial', 14, 'bold'), width=20, height=2)
        self.add_video_button.place(relx=0.5, rely=0.9, anchor='center')

        self.video_playing = False
        self.posList = []  # Pose landmarks listesi
        self.frame_height = 0

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        if file_path:
            self.add_video_button.place_forget()
            self.video_thread = threading.Thread(target=self.display_video, args=(file_path,))
            self.video_thread.start()
            self.open_save_window()

    def open_save_window(self):
        self.save_window = tk.Toplevel(self.root)
        self.save_window.title("Save Landmarks")
        self.save_window.geometry("300x100")
        self.save_label = tk.Label(self.save_window, text="", fg='green')
        self.save_label.pack(pady=10)
        self.save_button = tk.Button(self.save_window, text="Save", command=self.save_landmarks)
        self.save_button.pack(pady=10)

    def display_video(self, video_path):
        # Video dosyasını aç
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Video dosyası yüklenemedi.")
            return

        # MediaPipe Pose modelini yükle
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_duration = 1 / fps

        self.video_playing = True
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while self.video_playing:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            landmarks = results.pose_landmarks

            if landmarks:
                self.draw_pose(frame_rgb, landmarks)
                self.posList.append([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])

            pil_image = Image.fromarray(frame_rgb)
            resized_image = self.resize_image_to_fit(pil_image, (800, 600))
            tk_image = ImageTk.PhotoImage(image=resized_image)
            self.video_label.config(image=tk_image)
            self.video_label.image = tk_image

            self.root.update()
            self.root.after(int(frame_duration * 1000 / 2))

        cap.release()
        pose.close()

    def draw_pose(self, frame, landmarks):
        connections = mp.solutions.pose.POSE_CONNECTIONS
        landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, connections, landmark_drawing_spec)

    def resize_image_to_fit(self, image, target_size):
        target_width, target_height = target_size
        width, height = image.size
        aspect_ratio = min(target_width / width, target_height / height)
        new_width = int(width * aspect_ratio)
        new_height = int(height * aspect_ratio)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        background = Image.new('RGB', target_size, (0, 0, 0))
        offset = ((target_width - new_width) // 2, (target_height - new_height) // 2)
        background.paste(resized_image, offset)
        
        return background

    def stop_video(self):
        self.video_playing = False

    def key_event(self, event):
        if event.char == 's':
            self.save_landmarks()

    def save_landmarks(self):
        if self.posList:
            with open("AnimationFile.txt", 'w') as f:
                for frame_landmarks in self.posList:
                    for lm in frame_landmarks:
                        x = int(lm[0] * 800)  # Scaling x-coordinate to match your desired format
                        y = int(self.frame_height - lm[1] * self.frame_height)  # Scaling and flipping y-coordinate
                        z = int(lm[2] * 100)  # Scaling z-coordinate (if needed)
                        f.write(f'{x},{y},{z},')
                    f.write('\n')
            self.save_label.config(text="Data written to AnimationFile.txt")
        else:
            self.save_label.config(text="posList is empty, nothing to write.")

if __name__ == "__main__":
    root = tk.Tk()
    mainScreen = MainScreen(root)
    root.bind('<KeyPress>', mainScreen.key_event)
    root.mainloop()
