from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf


class TLClassifier(object):
    def __init__(self, is_site):
        self.IGNORE_THRESHOLD = 6
        self.confidence = 0.6
        graph_file = './sim/frozen_inference_graph.pb'
        if is_site:
            self.IGNORE_THRESHOLD = 3
            self.confidence = 0.5
            graph_file = './site/frozen_inference_graph.pb'
        graph = self.load_graph(graph_file)

        self.image_tensor = graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = graph.get_tensor_by_name('detection_classes:0')

        self.tf_session = tf.Session(graph=graph)
        self.light_state_dict = {0: TrafficLight.UNKNOWN,
                                 1: TrafficLight.GREEN,
                                 2: TrafficLight.RED,
                                 3: TrafficLight.YELLOW}
        self.ignore_cnt = 0
        self.light_state = TrafficLight.UNKNOWN


    def load_graph(self, graph_file):
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return graph


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # implement light color prediction
        self.ignore_cnt += 1
        if self.ignore_cnt < self.IGNORE_THRESHOLD:
            return self.light_state

        image_np = np.expand_dims(image, 0)
        (boxes, scores, classes) = self.tf_session.run([self.detection_boxes,
                                                        self.detection_scores,
                                                        self.detection_classes],
                                                       feed_dict={self.image_tensor: image_np})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        idxs = []
        for i in range(len(classes)):
            if scores[i] >= self.confidence:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]

        highest_score = 0.0
        light_id = 0
        for i in range(filtered_boxes.shape[0]):
            if filtered_scores[i] > highest_score:
                highest_score = filtered_scores[i]
                light_id = filtered_classes[i]

        self.ignore_cnt = 0

        self.light_state = self.light_state_dict[light_id]
        return self.light_state
