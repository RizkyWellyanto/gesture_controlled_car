# demo_camera dependencies
import sys
import time
import signal
sys.path.insert(0, 'python')
import cv2
import model

import util
from body import Body
import matplotlib.pyplot as plt
import copy
import numpy as np

# rospy_control dependencies
import os, sys
from os.path import dirname, abspath

# likely_lib_path = os.path.join(dirname(dirname(dirname(abspath(__file__)))), 'lib', 'p27_site_packages')
likely_lib_path = os.path.join(os.getcwd(), 'p27_site_packages')
if os.path.exists(likely_lib_path):
    sys.path.insert(0, likely_lib_path)

import roslib
import rospy
from pacmod_msgs.msg import PacmodCmd, PositionWithSpeed
from std_msgs.msg import Bool, Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# -------------------------
# set up global parameters
# -------------------------
topic_prefix = None
pub_cmd_topic = {}
pub_enable = None
pub_brake = None
pub_accel = None
pub_turn = None
pub_steer = None
pub_gear = None
brake_time = 0
signal_time = 0
steer_time = 0
steer_angular_position = 0
steer_angular_velocity_limit = 0
accel_speed = 0
accel_time = 0

kb_gains = dict(w=.5, s=.5, a=.2, d=.2)

# Desire Forward Car Speed
target_speed = 2

# Desire Steering Angle
des_steer    = 0.

# Steering State
KEEP_STRAIGHT      = 0
KEEP_TURNING_LEFT  = 1
KEEP_TURNING_RIGHT = 2

# Speed State
SET_FORWARD_SPEED  = 3
SET_BREAK          = 4

# Current State
CURRENT_STEERING_STATE = 0
CURRENT_SPEED_STATE    = 3

# Current Car Speed
vehicle_speed = 0.

# raw image from the car camera
oriImg = None

prev_time = time.time()
new_time  = time.time()

def clip_f(v, min_val=0, max_val=100):
    return max(min(max_val, v), min_val)

def setup_rospy():
    topic_prefix = '/pacmod/as_rx/'
    pub_cmd_topic = {'brake': (topic_prefix+'brake_cmd', PacmodCmd),
                     'accel': (topic_prefix+'accel_cmd', PacmodCmd),
                     'turn': (topic_prefix+'turn_cmd', PacmodCmd),
                     'steer': (topic_prefix+'steer_cmd', PositionWithSpeed),
                     'gear': (topic_prefix+'shift_cmd', PacmodCmd),
                     'enable': (topic_prefix+'enable', Bool)}

    global pub_enable
    global pub_brake
    global pub_accel
    global pub_turn
    global pub_steer
    global pub_gear
    pub_enable = rospy.Publisher(*pub_cmd_topic['enable'], queue_size=1)
    pub_brake = rospy.Publisher(*pub_cmd_topic['brake'], queue_size=1) # queue_size not handled
    pub_accel = rospy.Publisher(*pub_cmd_topic['accel'], queue_size=1)
    pub_turn = rospy.Publisher(*pub_cmd_topic['turn'], queue_size=1)
    pub_steer = rospy.Publisher(*pub_cmd_topic['steer'], queue_size=1)
    pub_gear = rospy.Publisher(*pub_cmd_topic['gear'], queue_size=1)
    rospy.init_node('openpose_ctrl', anonymous=True)
    pub_enable.publish(True)


    rospy.Subscriber('/pacmod/as_tx/vehicle_speed', Float64, set_vehicle_speed)

    steer_time = .5
    accel_time = 2.0

    steer_angular_position = 0.2   # dangerous, don't change too much
    steer_angular_velocity_limit = 0.1 # dangerous, don't change too much
    accel_speed = 0.5 # dangerous, don't change too much

class image_converter:
   def __init__(self):
       self.bridge = CvBridge()
       self.image_sub = rospy.Subscriber('/mako_1/mako_1/image_raw', Image, self.callback)

   def callback(self, data):
       try:
           # TODO: downsample
           global oriImg
           oriImg = cv2.resize(self.bridge.imgmsg_to_cv2(data, 'bgr8'), (384, 288))
           # oriImg = self.bridge.imgmsg_to_cv2(data, 'bgr8')
       except CvBridgeError as e:
           print(e)
           oriImg = None

def set_vehicle_speed(data):
    global vehicle_speed
    vehicle_speed = data.data

def signal_handler(signal, frame):
    # ----------------
    # CTRL-C detected
    # ----------------
    pub_enable.publish(True)
    pub_brake.publish(f64_cmd=0., enable=True, ignore=False, clear=False)
    pub_accel.publish(f64_cmd=0., enable=True, ignore=False, clear=False)
    pub_steer.publish(angular_position=0., angular_velocity_limit=+0.5)
    pub_gear.publish(ui16_cmd=PacmodCmd.SHIFT_PARK, enable=True, ignore=False, clear=False)
    pub_turn.publish(ui16_cmd=PacmodCmd.TURN_NONE, enable=True, ignore=False, clear=False)
    time.sleep(.5)
    pub_enable.publish(False)
    sys.exit(0)

class PID:
    def __init__(self, kp, kd, ki):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.reset()

    def reset(self):
        self.last_e = None
        self.last_t = None
        self.integ_e = 0.

    def __call__(self, err, dt=None):
        now_t = time.time()
        self.last_t = self.last_t or now_t
        self.last_e = self.last_e or err
        mydt = dt or (now_t - self.last_t)
        mydt = max(mydt, 1e-6)
        err_dot = (err - self.last_e) / mydt
        self.integ_e = self.integ_e + (err - self.last_e) * mydt
        kp_term = self.kp * err
        kd_term = self.kd * err_dot
        ki_term = self.ki * self.integ_e
        actuation = kp_term + kd_term + ki_term
        self.last_t = now_t
        self.last_e = err
        return actuation, kp_term, kd_term, ki_term, self.integ_e

class Rectifier:
    def __init__(self, pts):
        self.pts = list(sorted(pts, key= lambda x_: x_[0]))
        assert len(pts) > 0
        self.pts = [(-1e20, self.pts[0][1])] + self.pts + [(1e20, self.pts[-1][1])]

    def __call__(self, f):
        for i in range(len(self.pts)-1):
            pt1_x = self.pts[i][0]
            pt1_y = self.pts[i][1]
            pt2_x = self.pts[i+1][0]
            pt2_y = self.pts[i+1][1]

            if (f >= pt1_x) and (f < pt2_x):
                h = pt1_y + (pt2_y - pt1_y) * float(f-pt1_x) / float(pt2_x - pt1_x)
                return h

def rospy_brake():
    pub_enable.publish(True)
    rospy.loginfo('sent brake\n')
    pub_brake.publish(f64_cmd=0.5, enable=True)
    time.sleep(brake_time)
    pub_brake.publish(f64_cmd=0.0, enable=True)
    rospy.loginfo('cancelled brake\n')

def car_actions(candidate, subset):
    global prev_time
    global new_time
    global des_steer
    global CURRENT_STEERING_STATE
    global pid_speed
    global rectifier
    global vehicle_speed
    
    # 6 arm limb, 6 leg limb, 5 neck head limb, 2 unknown things
    limbSeq = [[3, 4], [2, 3], [1, 2], [1, 5], [5, 6], [6, 7], \
               [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],  \
               [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], \
               [2, 16], [5, 17]]
    
    # for each object: for each point (total 18): draw position (x, y)
    for n in range(len(subset)):
        index0 = int(subset[n][0])
        index1 = int(subset[n][1])
        index2 = int(subset[n][2])
        index3 = int(subset[n][3])
        index4 = int(subset[n][4])
        index5 = int(subset[n][5])
        index6 = int(subset[n][6])
        index7 = int(subset[n][7])
        
        # if ((index0 == -1) or (index2 == -1) or (index3 == -1) or (index4 == -1) or (index5 == -1) or (index6 == -1) or (index7 == -1)):
        if ((index0 == -1) or (index4 == -1) or (index7 == -1)):
            print('[INFO]:\tMissing Joint Points')
            # rospy_brake()
            CURRENT_STEERING_STATE = KEEP_STRAIGHT
            return

        x0, y0 = candidate[index0][0:2]

        # x1, y1 = candidate[index1][0:2]
      
        # x2, y2 = candidate[index2][0:2]
        
        # x3, y3 = candidate[index3][0:2]
        
        x4, y4 = candidate[index4][0:2]
        
        # x5, y5 = candidate[index5][0:2]
        
        # x6, y6 = candidate[index6][0:2]
       
        x7, y7 = candidate[index7][0:2]

        new_time  = time.time()
        dt        = new_time - prev_time
        prev_time = new_time

        acc_pos   = 0.
        brake_pos = 0.
        appropriate_gear = PacmodCmd.SHIFT_FORWARD
        angular_velocity = 0.45

        # print('y4: {}, y3: {}, x3: {}, x2: {}'.format(y4, y3, x3, x2))

        alpha = 2.5

        # Right Turn Pose
        # old: if (y4 < y3) and (x3 < x2):
        if y4 < y0:
            # print('[INFO]:\t Detected Right Turn Pose')
            des_steer = des_steer + (alpha * dt * kb_gains['d'])
            CURRENT_STEERING_STATE = KEEP_TURNING_RIGHT
        # Left Turn Pose
        # old: elif (y7 < y6) and (x5 < x6):
        elif y7 < y0:
            # print('[INFO]:\t Detected Left Turn Pose')
            des_steer = des_steer - (alpha * dt * kb_gains['a'])
            CURRENT_STEERING_STATE = KEEP_TURNING_LEFT
            
        # Hard Brake Pose
        elif abs(x7 - x4) < 75.0 and abs(y7 - x4) < 75.0:
            print('Hard Brake Signal')
            acc_pos   = 0.
            brake_pos = 0.7
            # desire steering remains the same
            pid_speed.reset()
            appropriate_gear = PacmodCmd.SHIFT_NEUTRAL

        # Acceleration Pose
        elif abs(x7 - x4) > 200.0:
            print('Acceleration Pose...')
            speed_err = target_speed - vehicle_speed
            pid_actuation, kp_term, kd_term, ki_term, intege = pid_speed(speed_err)
            rectified_actuation = rectifier(pid_actuation)
            print('------------ vehicle_speed: {}'.format(vehicle_speed))
            print('------------ speed_err: {}'.format(speed_err))
            print('------------ pid_actuation: {}'.format(pid_actuation))
            print('------------ rectified_actuation: {}'.format(rectified_actuation))
            acc_pos = clip_f(rectified_actuation, 0., 1.)
            brake_pos = clip_f(-rectified_actuation, 0., 1.)
            CURRENT_STEERING_STATE = KEEP_STRAIGHT # maintain speed 2mph

        des_steer = clip_f(des_steer, min_val=-11, max_val=11)

        # avoid over speed
        acc_pos = min(0.32, acc_pos)

        if vehicle_speed > 5:
            appropriate_gear = PacmodCmd.SHIFT_NEUTRAL

        if vehicle_speed > 3:
            angular_velocity = 0.25
        elif vehicle_speed > 2:
            angular_velocity = 0.35
        elif vehicle_speed > 1:
            angular_velocity = 0.45

        print('final des_steer: {}'.format(des_steer))
        print("brake_pos: {}".format(brake_pos))
        print('acc_pos: {}'.format(acc_pos))
        print('brake_pos: {}'.format(brake_pos))

        pub_enable.publish(True)
        pub_gear.publish(ui16_cmd=appropriate_gear, enable=True, ignore=False, clear=False)
        pub_accel.publish(f64_cmd=acc_pos, enable=True, ignore=False, clear=False)
        pub_brake.publish(f64_cmd=brake_pos, enable=True, ignore=False, clear=False)
        if CURRENT_STEERING_STATE == KEEP_TURNING_LEFT:
            rospy.loginfo('Sending Left Steering Direction...')
            pub_turn.publish(ui16_cmd=PacmodCmd.TURN_LEFT, enable=True, ignore=False, clear=False)
        elif CURRENT_STEERING_STATE == KEEP_TURNING_RIGHT:
            rospy.loginfo('Sending Right Steering Direction...')
            pub_turn.publish(ui16_cmd=PacmodCmd.TURN_RIGHT, enable=True, ignore=False, clear=False)
        else:
            rospy.loginfo('Sending Neutral Steering Direction...')
            pub_turn.publish(ui16_cmd=PacmodCmd.TURN_NONE, enable=True, ignore=False, clear=False)
        # rospy.loginfo('Sending Steering Signal...')
        pub_steer.publish(angular_position=-des_steer, angular_velocity_limit=+angular_velocity)
        # rospy.loginfo('End Steering')

    return


# ---------------
# Initialization
# ---------------
setup_rospy()
print('[INFO]:\tROS is ready!')
ic = image_converter()
print('[INFO]:\tConnected cv2 camera source to a ROS topic.')
signal.signal(signal.SIGINT, signal_handler)
print('[INFO]:\tCtrl-C signal handler is ready!')

print('[INFO]:\tInitialize PID...')
pid_speed = PID(kp=4., kd=1e-3, ki=6e-3)
print('[INFO]:\tInitialize Rectifier...')
rectifier = Rectifier(pts = [( -1,   -1. ),
                             ( 0.35 ,    0.35),])

# main program: get openpose body/hand labels, perform rospy control
body_estimation = Body('model/body_pose_model.pth')

prev_time = time.time()
while not rospy.is_shutdown():
    # skip if cv2 is not getting any images
    if oriImg is None:
        print('[WARNING]:\tNo image from ROS')
        continue

    candidate, subset = body_estimation(oriImg)

    # only perform car_action if camera see exactly 1 person
    if (len(subset) != 1):
        continue

    # --------------
    # Visualization
    # --------------
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    # A window to show the original video
    cv2.imshow('live', canvas)

    # Send ROS signal
    car_actions(candidate, subset)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # NOTE: signal callback?
        break
