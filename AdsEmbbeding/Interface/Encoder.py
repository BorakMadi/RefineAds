import cv2

class Encoder(object):

    def __init__(self, output_video_name=None, dimensions=(800, 600), fps=30, codec='mp4v'):
        self.id = id(self)
        if output_video_name is None:
            output_video_name = "output-{}".format(self.id)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.out = cv2.VideoWriter('./outputs/{}.mp4'.format(output_video_name), fourcc, fps, dimensions)

    def write_frame(self, edited_frame):
        self.out.write(edited_frame)


    def release(self):
        self.out.release()
