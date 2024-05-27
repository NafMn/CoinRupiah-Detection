import cv2
import os
import numpy as np

class CoinDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.detected_coins = []
        self.coin_count = {}

    def load_image(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print(f"Error: Unable to load image from {self.image_path}")
            exit()

    def preprocess_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (25, 25), 2)
        thresholded = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresholded

    def detect_circles(self, image):
        edges = cv2.Canny(image, 50, 150)
        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, dp=0.5, minDist=35, param1=50, param2=65, minRadius=10, maxRadius=80
        )
        return circles

    def process_detected_coins(self, circles):
        self.coin_count = len(circles[0]) if circles is not None else 0
        if circles is not None:
            circles = circles.astype(int)
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                area_mm2 = np.pi * (radius ** 1.15)
                coin_name = self.get_coin_name(area_mm2)

                self.detected_coins.append({
                    'name': coin_name,
                    'area_mm2': area_mm2,
                    'center': center,
                    'radius': radius
                })

    def get_coin_name(self, area):
        rupiah_coin_ranges = {
            "100 Rupiah": (250, 260),      # Sekitar 254.47 mm²
            "200 Rupiah": (280, 295),      # Sekitar 346.36 mm²
            "500 Rupiah": (300, 360),      # Sekitar 452.39 mm²
            "1000 Rupiah": (270, 280)      # Sekitar 530.93 mm²
        }

        for name, area_range in rupiah_coin_ranges.items():
            if area_range[0] <= area <= area_range[1]:
                return name

        return f"Unknown Coin ({area:.2f} mm2)"

    def draw_text_on_image(self):
        # Draw individual coin information
        for coin in self.detected_coins:
            text = f"{coin['name']}"
            center = coin['center']
            radius = coin['radius']

            # Draw the text on the image with anti-aliasing
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.2, 1)
            cv2.rectangle(self.image, (center[0] - text_size[0] // 2 - 5, center[1] + radius + 5),
                        (center[0] + text_size[0] // 2 + 5, center[1] + radius + text_size[1] + 10), (0, 0, 0), -1)
            cv2.putText(self.image, text, (center[0] - text_size[0] // 2, center[1] + radius + text_size[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw total detected object count
        text_count = f"Jumlah Objek Terdeteksi: {self.coin_count}"
        
        # Display the total object count in the top left corner of the image
        cv2.putText(self.image, text_count, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    def draw_contours_on_image(self):
        for coin in self.detected_coins:
            center = coin['center']
            radius = coin['radius']
            cv2.circle(self.image, center, radius, (0, 255, 0), 2)

    def resize_image(self, factor_x=1.95, factor_y=1.8):
        return cv2.resize(self.image, (int(self.image.shape[1] * factor_x), int(self.image.shape[0] * factor_y)))

    def display_result(self):
        resized_image = self.resize_image()
        cv2.imshow('Coin Rupiah Boos', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_image(self):
        self.load_image()
        preprocessed_image = self.preprocess_image()
        circles = self.detect_circles(preprocessed_image)
        self.process_detected_coins(circles)
        self.draw_contours_on_image()
        self.draw_text_on_image()
        self.display_result() 


def main():
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image.png')

    coin_detector = CoinDetector(image_path)
    coin_detector.process_image()

if __name__ == "__main__":
    main()
