import math
import os
import random
import subprocess
import time
from os import path
import math, numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState, SetModelStateRequest, SpawnModel
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
GOAL_REACHED_DIST = 0.7
COLLISION_DIST = 0.45   #小范围需要改称0.15，原来是0.35  
TIME_DELTA = 0.1
ROBOT_MODEL_NAME = "lzh_run_robot"
MODEL_WAIT_TIMEOUT = 180.0
ODOM_WAIT_TIMEOUT = 90.0
SET_MODEL_STATE_RETRIES = 3
SPAWN_MODEL_RETRIES = 3
ROBOT_DESCRIPTION_PARAM = "/robot_description"


# Check if the random goal position is located on an obstacle and do not accept it if it is
# Check if the random goal position is located on an obstacle and do not accept it if it is

# Check if the random goal position is located on an obstacle and do not accept it if it is
# Person 模型的坐标
pperson_positions = [
    (-0.367907, 4.20599),    # person_standing
    (3.36913, 1.11529),      # person_standing_0
    (-2.40739, -1.43645)     # person_standing_clone
]

# Bookshelf 书柜模型的坐标
bookshelf_positions = [
    (-0.066691, 4.316188),     # bookshelf
    {3.833930,3.833930}
]

# Cube 模型的坐标
cube_positions = [
    (0.879655, -1.18307),  # cube_20k
    (-1.79869, 1.06001),    # cube_20k_0
    (3.33188, 4.31104),      # cube_20k_1
    (-4.86938,0.495988),   # cube_20k_2
    (-1.34271, -4.27865),       # cube_20k_6
    (-4.69208, -4.36245)     # cube_20k_0_clone
]

# Dumpster 垃圾箱模型的坐标
dumpster_positions = [
    (3.83278, -3.46434),     # Dumpster
    (-3.91946, 4.33831)      # Dumpster_0
]
#十字架的坐标
ten_number=[(3.827450,-3.842281)]
#桌子的坐标
table=[(3.049900,4.569090),
       (-4.371780,4.602530),
    #    (0.613912,-3.311370),
    #    (5.100920,-1.543050),
    #    (1.163820,1.528840)
    ]
# Cabinet 柜子模型的坐标
chair=[(3.813390,0.850337),
       (0.014060,4.195960)
       ]

def check_pos(x, y):
    goal_ok = True
    
    # 检查 Person 模型，禁止区域半径为 1 米
    # for px, py in person_positions:
    #     distance = math.sqrt((x - px) ** 2 + (y - py) ** 2)
    #     if distance <= 1:
    #         goal_ok = False
    #         break
    
    # # 检查 Bookshelf 书柜模型，禁止区域半径为 1 米
    # if goal_ok:
    #     for bx, by in bookshelf_positions:
    #         distance = math.sqrt((x - bx) ** 2 + (y - by) ** 2)
    #         if distance <= 1:
    #             goal_ok = False
    #             break
    
   # 检查 Cube 模型，禁止区域半径为 1.5 米
    if goal_ok:
        for cx, cy in cube_positions:
            distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if distance <= 2.5:
                goal_ok = False
                break
    if goal_ok:#检查椅子模型
        for cx, cy in chair:
            distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if distance <= 3:
                goal_ok = False
                break
    # # 检查 Dumpster 垃圾箱模型，禁止区域半径为 2 米
    # if goal_ok:
    #     for dx, dy in dumpster_positions:
    #         distance = math.sqrt((x - dx) ** 2 + (y - dy) ** 2)
    #         if distance <= 2:
    #             goal_ok = False
    #             break
                # 检查十字架模型，禁止区域半径为 2.2 米
    if goal_ok:
        for dx, dy in ten_number:
            distance = math.sqrt((x - dx) ** 2 + (y - dy) ** 2)
            if distance <= 2.5:
                goal_ok = False
                break

                            # 检查桌子架模型，禁止区域半径为 2.2 米
    if goal_ok:
        for dx, dy in table:
            distance = math.sqrt((x - dx) ** 2 + (y - dy) ** 2)
            if distance <= 2.2:
                goal_ok = False
                break
    # # 检查 Cabinet 模型，禁止区域半径为 2 米
    # if goal_ok:
    #     for cx, cy in cabinet_positions:
    #         distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    #         if distance <= 2:
    #             goal_ok = False
    #             break
    if x<-6.5 or x>6.5 or y<-6.5 or y>6.5:#检查边界
        goal_ok = False
    return goal_ok


        
class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim,roscoreip,gazebo_port,t):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0
        self.angle=0
        self.goal_x = 1
        self.goal_y = 0.0
        self.succes=0
        self.step_penalty=-1
        self.test=1
        self.choice=False
        self.collision_check=False
        self.end_check=False
        self.upper = 5
        self.lower = -5
        self.last_angular_velocity=0
        self.scale_to_goal=11
        self.obs_scale=0.5
        self.acceleration_scale=0
        self.last_speed=0
        self.last_distance=0
        self.smoothness_scale=0.1
        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None
        self.obs_points=[]
        self.project_root = path.abspath(path.join(path.dirname(__file__), ".."))
        self.robot_model_name = ROBOT_MODEL_NAME
        self.set_self_state = ModelState()
        self.set_self_state.model_name = self.robot_model_name#zzx
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = -4.2
        self.set_self_state.pose.position.z = 0.02
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        initial_quaternion = Quaternion.from_euler(0.0, 0.0, 1.57)
        self.set_self_state.pose.orientation.z = initial_quaternion.z
        self.set_self_state.pose.orientation.w = initial_quaternion.w

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        port = roscoreip
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        os.environ['ROS_MASTER_URI'] = f'http://localhost:{port}'
        os.environ['GAZEBO_MASTER_URI'] = f'http://localhost:{gazebo_port}'
        subprocess.Popen(["roscore", "-p", str(port)])
        time.sleep(10)
        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        # os.environ['ROS_MASTER_URI'] = f'http://localhost:{port}'
        # os.environ['GAZEBO_MASTER_URI'] = f'http://localhost:{gazebo_port}'
        print(port)
        print(gazebo_port)
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        setup_file = path.join(self.project_root, "catkin_ws", "devel", "setup.bash")
        command = f'source "{setup_file}" && roslaunch "{fullpath}"'
        subprocess.Popen(['bash', '-c', command])
        
        print("Gazebo launched!")
        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.spawn_urdf_model = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1#zzx
        )
        self.odom = rospy.Subscriber(
            "/odom", Odometry, self.odom_callback, queue_size=1#zzx
        )
        self._wait_for_gazebo_services(timeout=MODEL_WAIT_TIMEOUT)
        self._unpause_physics_for_startup()
        self.ensure_robot_spawned()
        self.wait_for_odom(timeout=ODOM_WAIT_TIMEOUT, log_label="first /odom")


    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def _wait_for_gazebo_services(self, timeout=MODEL_WAIT_TIMEOUT):
        services = [
            "/gazebo/get_model_state",
            "/gazebo/set_model_state",
            "/gazebo/spawn_urdf_model",
            "/gazebo/reset_world",
            "/gazebo/unpause_physics",
            "/gazebo/pause_physics",
        ]
        deadline = time.time() + timeout
        for service_name in services:
            remaining = max(0.1, deadline - time.time())
            try:
                rospy.wait_for_service(service_name, timeout=remaining)
            except rospy.ROSException as exc:
                raise RuntimeError(
                    "Timed out waiting for Gazebo service {} after {:.1f}s".format(
                        service_name, timeout
                    )
                ) from exc
        rospy.loginfo("Gazebo services ready.")

    def wait_for_model(self, timeout=MODEL_WAIT_TIMEOUT):
        deadline = time.time() + timeout
        last_status = ""
        next_unpause = 0.0
        while not rospy.is_shutdown() and time.time() < deadline:
            if time.time() >= next_unpause:
                self._unpause_physics_for_startup()
                next_unpause = time.time() + 5.0
            try:
                response = self.get_model_state(self.robot_model_name, "world")
                if response.success:
                    rospy.loginfo("Gazebo model [%s] ready.", self.robot_model_name)
                    return response
                last_status = response.status_message
            except rospy.ServiceException as exc:
                last_status = str(exc)
            time.sleep(0.5)
        raise RuntimeError(
            "Timed out waiting for Gazebo model [{}] after {:.1f}s. Last status: {}".format(
                self.robot_model_name, timeout, last_status
            )
        )

    def _unpause_physics_for_startup(self):
        try:
            self.unpause()
        except rospy.ServiceException as exc:
            rospy.logwarn("Could not unpause Gazebo while waiting for spawn: %s", exc)

    def ensure_robot_spawned(self):
        try:
            return self.wait_for_model(timeout=5.0)
        except RuntimeError:
            rospy.logwarn(
                "Gazebo model [%s] is not present after launch; spawning from %s.",
                self.robot_model_name,
                ROBOT_DESCRIPTION_PARAM,
            )

        robot_xml = self._wait_for_robot_description(timeout=30.0)
        last_status = ""
        for attempt in range(1, SPAWN_MODEL_RETRIES + 1):
            self._unpause_physics_for_startup()
            try:
                response = self.spawn_urdf_model(
                    self.robot_model_name,
                    robot_xml,
                    "",
                    self.set_self_state.pose,
                    "world",
                )
                last_status = response.status_message
                rospy.loginfo("spawn_urdf_model attempt %d: %s", attempt, last_status)
            except rospy.ServiceException as exc:
                last_status = str(exc)
                rospy.logwarn("spawn_urdf_model attempt %d failed: %s", attempt, exc)

            try:
                return self.wait_for_model(timeout=30.0)
            except RuntimeError as exc:
                last_status = str(exc)
                rospy.logwarn("Robot model still missing after spawn attempt %d.", attempt)

        raise RuntimeError(
            "Failed to spawn Gazebo model [{}] after {} attempts. Last status: {}".format(
                self.robot_model_name, SPAWN_MODEL_RETRIES, last_status
            )
        )

    def _wait_for_robot_description(self, timeout=30.0):
        deadline = time.time() + timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            if rospy.has_param(ROBOT_DESCRIPTION_PARAM):
                return rospy.get_param(ROBOT_DESCRIPTION_PARAM)
            time.sleep(0.2)
        raise RuntimeError(
            "Timed out waiting for ROS param {} after {:.1f}s".format(
                ROBOT_DESCRIPTION_PARAM, timeout
            )
        )

    def _set_odom_state(self, od_data):
        self.last_odom = od_data
        self.odom_x = od_data.pose.pose.position.x
        self.odom_y = od_data.pose.pose.position.y
        quaternion = Quaternion(
            od_data.pose.pose.orientation.w,
            od_data.pose.pose.orientation.x,
            od_data.pose.pose.orientation.y,
            od_data.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        self.angle = round(euler[2], 4)

    def wait_for_odom(self, timeout=ODOM_WAIT_TIMEOUT, log_label="/odom"):
        deadline = time.time() + timeout
        last_error = None
        next_unpause = 0.0
        while not rospy.is_shutdown() and time.time() < deadline:
            if time.time() >= next_unpause:
                self._unpause_physics_for_startup()
                next_unpause = time.time() + 5.0
            try:
                remaining = max(0.1, min(5.0, deadline - time.time()))
                odom_msg = rospy.wait_for_message("/odom", Odometry, timeout=remaining)
                break
            except rospy.ROSException as exc:
                last_error = exc
        else:
            raise RuntimeError(
                "Timed out waiting for {} after {:.1f}s".format(log_label, timeout)
            ) from last_error
        self._set_odom_state(odom_msg)
        rospy.loginfo("%s received.", log_label)
        return odom_msg

    def _set_model_state_with_retry(self, model_state, retries=SET_MODEL_STATE_RETRIES):
        request = SetModelStateRequest()
        request.model_state = model_state
        last_status = ""
        for attempt in range(1, retries + 1):
            try:
                response = self.set_model_state(request)
                if response.success:
                    return response
                last_status = response.status_message
            except rospy.ServiceException as exc:
                last_status = str(exc)
            time.sleep(0.1 * attempt)
        raise RuntimeError(
            "Failed to set Gazebo model [{}] state after {} attempts. Last status: {}".format(
                model_state.model_name, retries, last_status
            )
        )
       
    def velodyne_callback(self, v):
        if self.last_odom is None:
            return
        self.obs_points.clear()
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        self.angle = round(euler[2], 4)
      #  print(angle)
        for i in range(len(data)):         
            if data[i][2] > 0:
      #          print(data[i][0],data[i])
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 )   #

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)   #这个角度里面就保存一个最近dist的障碍物，这个距离是由障碍物的x,y,z算出来的,那xy一样的话肯定z=0最小，实际上也就是比的x,y算出来的dist，但是
                        #与单线的区别是，这个可以扫描较矮的障碍物
                  #      print(self.gaps[j][0],self.velodyne_data[j])
                        obs_angle=self.angle+self.gaps[j][0]#计算障碍物相对于小车的角度
                        if(obs_angle>3.14):
                            obs_angle=-3.14+(obs_angle-3.14)
                        if(obs_angle<-3.14):
                            obs_angle=3.14+(obs_angle+3.14)
                     #   print(obs_angle)
                        obs_position_x=self.last_odom.pose.pose.position.x+math.cos(obs_angle)*self.velodyne_data[j]#计算障碍物的全局坐标
                        obs_position_y=self.last_odom.pose.pose.position.y+math.sin(obs_angle)*self.velodyne_data[j]
                        self.obs_points.append((obs_position_x,obs_position_y,dist))
                #        print(obs_position_x,obs_position_y)
                        
                        break
     
    def calculate_total_repulsion_force(self,x_robot, y_robot,force_magnitude_attract):
        total_force_x = 0
        total_force_y = 0
        Rho_att=math.sqrt((self.goal_x-x_robot)**2+(self.goal_y-y_robot)**2)
        for obs in self.obs_points:
          dx = obs[0] - x_robot
          dy = obs[1] - y_robot
          dr = math.sqrt(dx**2 + dy**2)
          
          if dr > 0:  # 避免除以零，理论上障碍物不应该和机器人重合
            k=10 #0.001
            Rho_obs=dr
            Rho=1
            
            if(Rho_obs<Rho):
                force_magnitude = k * (1 / Rho_obs-1.0/Rho ) *Rho_att / math.sqrt(Rho_obs)
                # if(force_magnitude>force_magnitude_attract):
                #     force_magnitude=force_magnitude_attract
                force_x = -force_magnitude * dx / dr
                force_y = -force_magnitude * dy / dr
            else:
                force_x = 0
                force_y = 0
            
            total_force_x += force_x
            total_force_y += force_y
        #    print(total_force_x,total_force_y)
        return (total_force_x, total_force_y)
    
    def calculate_attraction_force(self,x_robot, y_robot, x_target, y_target, k_a):
        dx = x_target - x_robot
        dy = y_target - y_robot
        da = math.sqrt(dx**2 + dy**2)
        force_magnitude = k_a * (da )
        return (force_magnitude * dx / da, force_magnitude * dy / da)
    
    def odom_callback(self, od_data):
        self._set_odom_state(od_data)

    # Perform an action and read a new state
    def step(self, last_state,action,omega):
        target = False
     
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)

        self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        sigma = 0.01  # 噪声强度，根据实际调整
        # 原始数据 v_state是list
        v_state = np.array(self.velodyne_data)  # 转换为numpy数组

# 加入高斯噪声
        sigma = 0.001  # 噪声强度
        noisy_v_state = v_state 

        # Min-Max归一化（假设测距范围是0~10米）
        laser_min, laser_max = 0.0, 10.0
        normalized_v_state = (noisy_v_state - laser_min) / (laser_max - laser_min)

        # 限制在[0,1]之间
        normalized_v_state = np.clip(normalized_v_state, 0.0, 1.0)

        # 如果后续需要放回列表中：
        laser_state = [noisy_v_state.tolist()]

        # Calculate robot heading from odometry data
        if self.last_odom is None:
            self.wait_for_odom(timeout=ODOM_WAIT_TIMEOUT, log_label="/odom before step")
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
     #   print("odom:!!!",self.odom_x)
        #for j in range(len(self.gaps)):
           # if(self.velodyne_data[j]<10):
          #   print(self.gaps[j][0],self.gaps[j][1])
           # print(self.velodyne_data[j])#这里保存当前时刻的扫描情况
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        self.angle = round(euler[2], 4)
       # print(angle)
        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - self.angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            print("goal reached! !  !")
            
            self.succes+=1
            self.end_check=True
            done = True
        if collision:
            self.collision_check=True
        robot_state = [distance, theta, action[0], action[1]]
        
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(self,target, collision, action, min_laser,last_state,omega,distance)
        return state, reward, done, target

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        self.angle = np.random.uniform(-np.pi, np.pi)

        quaternion = Quaternion.from_euler(0.0, 0.0, self.angle)
        object_state = self.set_self_state





        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-6, 6)
            y = np.random.uniform(-6, 6)
            position_ok = check_pos(x, y)

            if  x<2 and x>-8 and y>-3.5 and y<-1.5:
                    box_ok=False
            if  x>-4 and x<-2 and y>-7 and y<3:
                    box_ok=False
        object_state.pose.position.x =x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        object_state.reference_frame = "world"
        self.wait_for_model(timeout=MODEL_WAIT_TIMEOUT)
        self._set_model_state_with_retry(object_state)
        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y


        self.change_goal()

                # randomly scatter boxes in the environment
        cylinder=[]
        for i in range(6):
            name = "drc_practice_blue_cylinder_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-7, 7)
                y = np.random.uniform(-7, 7)
                box_ok = check_pos(x, y)
                for j in cylinder:
                    dx=abs(x-j[0])
                    dy=abs(y-j[1])
                    if  np.linalg.norm([dx, dy]) < 4:                                                                                                                             
                        box_ok=False
                        break
                if  x<2 and x>-8 and y>-3.5 and y<-1.5:
                    box_ok=False
                if  x>-4 and x<-2 and y>-7 and y<3:
                    box_ok=False
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 2.2 or distance_to_goal < 2.2:
                    box_ok = False
            cylinder.append((x,y))
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

        # set a random goal in empty space in environment
        
        self.distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        
        # randomly scatter boxes in the environment
        
        self.publish_markers([0.0, 0.0])
      #  self.random_target()
        
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)
        self.wait_for_odom(timeout=ODOM_WAIT_TIMEOUT, log_label="/odom after reset")
       # time.sleep(1)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - self.angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
       # self.random_target()
        return state

    def change_goal(self):
        # Place a new goal asnd check if its location is not on one of the obstacles
        if self.upper < 18:
            self.upper += 0.008
        if self.lower > -18:
            self.lower -= 0.008

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)
        # self.goal_x = 2
        # self.goal_y = 2
        print("upper:",self.upper)
    #    print("odom:",self.odom_x,self.odom_y)
        print("goal:",self.goal_x,self.goal_y)

    def random_target(self):
        # 要求：self.goal_x, self.goal_y 先被设置好（例如在 change_goal() 里）
        import rospy
        from gazebo_msgs.srv import SetModelState, SetModelStateRequest
        from gazebo_msgs.msg import ModelState
        from geometry_msgs.msg import Twist

        # 等待并获取服务代理
        rospy.wait_for_service("/gazebo/set_model_state")
        set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        # 组包
        req = SetModelStateRequest()
        req.model_state = ModelState()
        req.model_state.model_name = "Target_0"         # 确保与世界中的名字完全一致
        req.model_state.pose.position.x = float(self.goal_x)
        req.model_state.pose.position.y = float(self.goal_y)
        req.model_state.pose.position.z = 0.05          # 稍抬起，避免埋地
        req.model_state.pose.orientation.w = 1.0
        req.model_state.twist = Twist()
        req.model_state.reference_frame = "world"       # 关键：用世界坐标系

        # 调用并可选重试（应对 reset 后被覆盖）
        ok = False
        for _ in range(3):
            resp = set_model_state(req)
            ok = resp.success
            if ok:
                break
            rospy.sleep(0.05)

        if not ok:
            rospy.logwarn("[random_target] set_model_state failed; check model name and timing.")

            
    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        cylinder=[]
        for i in range(2):
            name = "drc_practice_blue_cylinder_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-7, 7)
                y = np.random.uniform(-7, 7)
                box_ok = check_pos(x, y)
                for j in cylinder:
                    dx=abs(x-j[0])
                    dy=abs(y-j[1])
                    if  np.linalg.norm([dx,dy]) <3:                                                                                                                             
                        box_ok=False
                        break
                if  x<1 and x>-8 and y>-3 and y<-1:
                 box_ok=False
                if  x>-4 and x<-2 and y>-6 and y<3.5:
                 box_ok=False
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 2.5 or distance_to_goal < 2.5:
                    box_ok = False
            cylinder.append((x,y))
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        
        
        min_laser = min(laser_data)
        #print(min_laser)
        if min_laser < COLLISION_DIST:
            print("collsion!")
            return True, True, min_laser
        return False, False, min_laser

    # @staticmethod
    # def get_reward(self, target, collision, action, min_laser,last_state,omega,distance_to_goal):
    #     if target:
    #         return 100.0
    #     elif collision:
    #         return -100.0
    #     else:
    #         r3 = lambda x: 1 - x if x < 1 else 0.0
    #         return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2





    @staticmethod
    def get_reward(self, target, collision, action, min_laser,last_state,omega,distance_to_goal):
        # Calculate the reward for the current state    35
            reward=0.0 
            if target:
                reward= 0  
            current_distance = distance_to_goal  # 获取当前与目标的距离
            distance_diff = self.last_distance - current_distance  # 距离变化
            reward += distance_diff * self.scale_to_goal  # 放大系数，可根据需要调整
            self.last_distance = current_distance  # 更新上一次的距离
         #   print("reward_goal:",reward)
            base = 1
            k=250
          #  print("goal_reard:",reward)
            threshold = 0.45  # 阈值，可根据需要调整
           # reward=0.0
            distance=10
            for obs in self.obs_points:
                #print(obs[2])
                distance = min(distance,obs[2])  # 假设obs[2]是障碍物的距离
            #    print(distance)
            # if(distance< threshold):
            #     reward -= self.obs_scale * math.exp(-(distance - threshold) ** 2)
            reward-=self.obs_scale* 1.0/ (1 + np.exp(k * (distance - threshold)))
            # if  distance<0.7:
                
            #     print("distance:",distance)
            #     print("reward_obs:",self.obs_scale* 1/ (1 + np.exp(k * (distance - threshold))))
            # if 1 / (1 + np.exp(k * (distance - threshold)))>0.01:
            #     print("distance:",distance)
            #     print("distance_reard:", -self.obs_scale*1 / (1 + np.exp(k * (distance - threshold))))
          #  print("reward_all:",rewWard)
            
            reward+=self.step_penalty

                        # 计算真实的角度差，范围在 (-π, π)
            angular_velocity_diff = abs(action[1] - self.last_angular_velocity)
            # 使用余弦函数计算平滑度惩罚，使得范围在 [0, -1] 之间
            smoothness_penalty = math.tanh(angular_velocity_diff) * self.smoothness_scale
            reward -= smoothness_penalty  # 更新奖励值
            # 更新角度
            self.last_angular_velocity = action[1]


            # self.last_angle = self.angle  # 更新上一时刻的角速度


                        # 计算当前速度和加速度的变化（考虑机器人的稳定性）
            speed_diff = abs(action[0] - self.last_speed)  # 假设action[0]是当前速度
            acceleration_penalty = math.tanh(speed_diff) * self.acceleration_scale  # 加速度惩罚系数
            reward -= acceleration_penalty
            self.last_speed = action[0]  # 更新上一个时间步的速度


            #reward/=2
         #   print("obs_reard:",reward)
            return  reward

    #     ""
        
    #     计算奖励值

    #     参数:
    #     target (bool): 是否到达目标
    #     collision (bool): 是否发生碰撞
    #     action (list): 当前动作
    #     min_laser (float): 最小激光雷达距离

    #     返回:
    #     float: 奖励值
    #     """
    #  #   print(last_state[22],last_state[23])
    #     # 到达目标给予100的正向奖励
    #     # reward=0.0
    #     # if target:
    #     #    reward=600.0
    #     # if collision:
    #     #     reward=-300.0    
    #     # else:
    #     #     # 根据障碍物距离计算负向奖励
    #     #     base = 1
    #     #     threshold = 1.5
    #     #     scale = 0.0001
    #     #     for obs in self.obs_points:
    #     #         distance = obs[2]  # 假设obs[2]是障碍物的距离
    #     #     #    print(distance)
    #     #         if(distance< 1):
    #     #             reward -= scale * math.exp(-(distance - threshold) ** 2)
    #     #     #print(reward)
            
    #     # reward2=0.0
    #     # angele_offset=omega
    #     # #print("angel_coffset %d",angele_offset)
    #     # reward2=0.1*math.cos(angele_offset)-0.2
    #     # #print("calculate angel reward:",reward2)
    #     # #print(reward)
    #     # #print(angele_offset,reward2)
    #     # reward+=reward2
    #     # #print(reward2)
    #     # #print(reward)
    #     # #print("\n")
    #     # return reward
    #     #print(reward)
    
