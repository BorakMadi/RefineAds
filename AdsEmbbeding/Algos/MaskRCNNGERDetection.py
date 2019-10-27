from Interface.GERDetection import GERDetection
import subprocess as sp
import json


class MaskRCNNGERDetection(GERDetection):

    def __init__(self, weights, shape):
        super().__init__()
        self.detection_process = sp.Popen(['python3.6', './AdDetection/Detect.py', '--weights={}'.format(weights),
                                           '--size={}x{}'.format(shape[1], shape[0])], stdout=sp.PIPE,
                                          stdin=sp.PIPE)

    def detect_regions(self, frame):
        self.detection_process.stdout.flush()
        self.detection_process.stdin.write(frame.tobytes())
        cnts_string = self.detection_process.stdout.readline()
        parsed_json = json.loads(cnts_string)
        cnts = parsed_json['boundaries']

        return cnts
