from Interface.Decoder import Decoder
import cv2


class VideoFileDecoder(Decoder):

    def __init__(self, file_path, resize=None, skip=0):
        super().__init__()
        self.cap = cv2.VideoCapture(file_path)
        self.curr_frame = None
        self.status = self.cap.isOpened()
        print(self.status)
        self.resize = resize

        assert self.cap.isOpened()

        for _ in range(skip):
            self.read_frame()


    def read_frame(self):
        ret, frame = self.cap.read()

        if ret and self.resize is not None:
            frame = cv2.resize(frame, self.resize)

        self.curr_frame = frame
        self.status = ret

        return self.status


    def get_frame_size(self):
        if self.resize is not None:
            return self.resize
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        return int(width), int(height)


    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        return self.curr_frame

    def is_open(self):
        return self.status

