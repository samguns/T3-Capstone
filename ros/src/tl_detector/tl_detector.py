#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3
TL_DETCTION_LOOKAHEAD_WPS = 50

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.camera_image = None
        self.lights = []
        self.car_position = -1
        self.waypoints_2d = None
        self.stop_line_idxs = []

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.light_classifier = TLClassifier(self.config['is_site'])

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.has_image = False

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        rospy.Subscriber('/vehicle/car_position', Int32, self.car_position_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints]
            waypoint_tree = KDTree(self.waypoints_2d)

            stop_line_positions = self.config['stop_line_positions']
            for i, stop_line in enumerate(stop_line_positions):
                # Get stop line waypoint index
                idx = waypoint_tree.query([stop_line[0], stop_line[1]], 1)[1]
                self.stop_line_idxs.append(idx)

            self.stop_line_idxs.sort()

        if self.config['is_site']:
            self.stop_line_idxs[0] -= 3

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # cv2.imwrite("./images/" + '{:04}'.format(msg.header.seq) + ".jpg", cv_image)

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def car_position_cb(self, msg):
        self.car_position = msg.data
        if len(self.stop_line_idxs) == 0:
            return

        if self.car_position > self.stop_line_idxs[0]:
            if not self.config['is_site']:
                self.stop_line_idxs = self.stop_line_idxs[1:]

    def get_light_state(self):
        """Determines the current color of the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # return light.state
        if not self.has_image:
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        check_light_state = False
        light_wp = None

        if self.pose and len(self.stop_line_idxs) > 0:
            # find the closest visible traffic light (if one exists)
            closest_stop_line_index = self.stop_line_idxs[0]
            d = closest_stop_line_index - self.car_position
            if 0 <= d < TL_DETCTION_LOOKAHEAD_WPS:
                check_light_state = True
                light_wp = closest_stop_line_index

        if check_light_state:
            state = self.get_light_state()
            return light_wp, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
