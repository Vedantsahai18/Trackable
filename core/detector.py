import cv2
import numpy as np

from utils.helpers import print_info
from utils.constants import NMS_CONFIDENCE, NMS_THRESHOLD, DETECTION_CONFIDENCE

class Detector:
    '''
    Class for detecting objects.
    '''

    def __init__(self, cfg_path, weights_path):
        '''
        Load the model weights and classes.
        '''
        
        print_info("Loading YOLOv3 Model...")
        self.model = cv2.dnn.readNet(cfg_path, weights_path)
        self.output_layers = self.model.getUnconnectedOutLayersNames()
        with open('yolo/classes.txt', 'r') as f:
            self.classes = f.read().splitlines()
        print_info("Model Loaded!")

    def detect_person(self, image):
        '''
        Perform human detection.
        '''

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        outputs = self.model.forward(self.output_layers)
        
        labels = []
        confidences = []
        boxes = []
        conf_threshold = NMS_CONFIDENCE
        nms_threshold = NMS_THRESHOLD

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > DETECTION_CONFIDENCE and self.classes[class_id] == 'person':
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    w = int(detection[2] * image.shape[1])
                    h = int(detection[3] * image.shape[0])
                    x = center_x - w / 2
                    y = center_y - h / 2
                    labels.append(self.classes[class_id])
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Applying Non-Max Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        final_boxes = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            final_boxes.append((round(x), round(y), round(x+w), round(y+h)))

            self.__draw_bounding_box(image, labels[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        return image, final_boxes


    def __draw_bounding_box(self, image, label, confidence, start_x, start_y, end_x, end_y):
        '''
        Draw bounding box after detecting human.
        '''

        color = (0, 255, 0)

        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)
        cv2.putText(image, label + ' ' + str(confidence), (start_x - 10, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
