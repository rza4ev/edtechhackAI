import os
import cv2
import PIL
import numpy as np
from mediapipe.python.solutions import hands, drawing_utils
from mistralai import Mistral
from gtts import gTTS
from warnings import filterwarnings
filterwarnings(action='ignore')

class calculator:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(propId=cv2.CAP_PROP_FRAME_WIDTH, value=950)
        self.cap.set(propId=cv2.CAP_PROP_FRAME_HEIGHT, value=550)
        self.cap.set(propId=cv2.CAP_PROP_BRIGHTNESS, value=130)
        self.imgCanvas = np.zeros(shape=(550,950,3), dtype=np.uint8)
        self.mphands = hands.Hands(max_num_hands=1, min_detection_confidence=0.75)
        self.p1, self.p2 = 0, 0
        self.p_time = 0
        self.fingers = []
        self.client = Mistral(api_key="zGGCgb1vjalsFqrtP6AFF5mAkoF0aWde")
        self.is_drawing = False

    def process_frame(self):
        success, img = self.cap.read()
        img = cv2.resize(src=img, dsize=(950,550))
        self.img = cv2.flip(src=img, flipCode=1)
        self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def process_hands(self):
        result = self.mphands.process(image=self.imgRGB)
        self.landmark_list = []
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                drawing_utils.draw_landmarks(image=self.img, landmark_list=hand_lms, 
                                             connections=hands.HAND_CONNECTIONS)
                for id, lm in enumerate(hand_lms.landmark):
                    h, w, c = self.img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.landmark_list.append([id, cx, cy])

    def identify_fingers(self):
        self.fingers = []
        if self.landmark_list != []:
            for id in [4,8,12,16,20]:
                if id != 4:
                    self.fingers.append(1 if self.landmark_list[id][2] < self.landmark_list[id-2][2] else 0)
                else:
                    self.fingers.append(1 if self.landmark_list[id][1] < self.landmark_list[id-2][1] else 0)
            for i in range(0, 5):
                if self.fingers[i] == 1:
                    cx, cy = self.landmark_list[(i+1)*4][1], self.landmark_list[(i+1)*4][2]
                    cv2.circle(img=self.img, center=(cx,cy), radius=5, color=(255,0,255), thickness=1)

    def handle_drawing_mode(self):
        if self.is_drawing and self.landmark_list:
            cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(img=self.imgCanvas, pt1=(self.p1,self.p2), pt2=(cx,cy), color=(255,0,255), thickness=5)
            self.p1,self.p2 = cx,cy

    def blend_canvas_with_feed(self):
        img = cv2.addWeighted(src1=self.img, alpha=0.7, src2=self.imgCanvas, beta=1, gamma=0)
        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(src=imgGray, thresh=50, maxval=255, type=cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(src1=img, src2=imgInv)
        self.img = cv2.bitwise_or(src1=img, src2=self.imgCanvas)

    def analyze_image_with_mistral(self):
        imgCanvas = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2RGB)
        imgCanvas = PIL.Image.fromarray(imgCanvas)
        prompt = "write about the triangle"
        response = self.client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content
        tts = gTTS(text=response_text, lang="en")
        tts.save("mistral_answer.mp3")
        os.system("start mistral_answer.mp3" if os.name == "nt" else "afplay mistral_answer.mp3")
        return response_text

    def main(self):
        print("[i] 'w' yazmağa başla | 's' yazını dayandır | 'r' AI göndər | 'c' təmizlə | 'q' çıxış")
        while True:
            if not self.cap.isOpened():
                print("❌ Kamera açıla bilmədi. Cihaz bağlıdırmı?")
                break

            self.process_frame()
            self.process_hands()
            self.identify_fingers()
            self.handle_drawing_mode()
            self.blend_canvas_with_feed()
            cv2.imshow("AI Math Solver", self.img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('w'):
                self.is_drawing = True
                print("✍ Yazmağa başladın")
            elif key == ord('s'):
                self.is_drawing = False
                self.p1, self.p2 = 0, 0
                print("✋ Yazını dayandırdın")
            elif key == ord('c'):
                self.imgCanvas = np.zeros(shape=(550,950,3), dtype=np.uint8)
                print("🧼 Lövhə təmizləndi")
            elif key == ord('r'):
                print("🤖 AI cavabı göndərilir...")
                result = self.analyze_image_with_mistral()
                print("Cavab:", result)
            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calc = calculator()
    calc.main()
