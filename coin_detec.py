import cv2
import numpy as np

class CoinDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.detected_coins = []

    def load_image(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print(f"Error: Unable to load image from {self.image_path}")
            exit()

    def detect_coins(self):
        model = cv2.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "model.pbtxt")
        blob = cv2.dnn.blobFromImage(self.image, size=(300, 300), swapRB=True, crop=False)
        model.setInput(blob)
        detections = model.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                if class_id == 1:  # Class ID for "person" in COCO dataset
                    box = detections[0, 0, i, 3:7] * np.array([self.image.shape[1], self.image.shape[0], self.image.shape[1], self.image.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    self.detected_coins.append({
                        'name': 'Coin',
                        'area_mm2': (endX - startX) * (endY - startY),
                        'center': (int((startX + endX) / 2), int((startY + endY) / 2)),
                        'radius': max((endX - startX) / 2, (endY - startY) / 2)
                    })

    def draw_detected_coins(self):
        for coin in self.detected_coins:
            center = coin['center']
            radius = coin['radius']
            cv2.circle(self.image, center, int(radius), (0, 255, 0), 2)

    def display_result(self):
        cv2.imshow('Detected Coins', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_image(self):
        self.load_image()
        self.detect_coins()
        self.draw_detected_coins()
        self.display_result()

def main():
    image_path = 'seribu.png'
    coin_detector = CoinDetector(image_path)
    coin_detector.process_image()

if __name__ == "__main__":
    main()
