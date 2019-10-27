from Interface.GERDetection import GERDetection
import cv2

class UserInputGERDetection(GERDetection):

    def __init__(self):
        super().__init__()
        self.selection = []

    def detect_regions(self, frame):

        self.points4Polygon(frame.copy())

        return [self.selection]


    def click_and_crop(self, event, x, y, flags, param):
        # grab references to the global variables
        # if the left mouse button was clickeds
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection.append((x, y))
            cv2.circle(param, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow('image', param)


    def points4Polygon(self, img):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.click_and_crop, img)
        # keep looping until the 'q' key is pressed

        clone = img.copy()
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", img)
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                img = clone.copy()

            # if the 'c' key is pressed, break from the loo
            if len(self.selection) == 4:
                break

        cv2.destroyAllWindows()

