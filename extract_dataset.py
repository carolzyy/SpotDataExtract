import os
import pickle
import glob
from dataclasses import dataclass, field
import numpy as np
from typing import List
import bosdyn.client
@dataclass
class jointstate:
    name: str = field(default='')
    position: float = field(default=0.0)
    velocity: float = field(default=0.0)
    acceleration: float = field(default=0.0)
    load: float = field(default=0.0)

def load_arm_states(arm_folder):
    arm_states = {}
    for filename in os.listdir(arm_folder):
        with open(os.path.join(arm_folder, filename), 'rb') as file:
            timestamp = float(filename.split('.pkl')[0])
            state = pickle.load(file)
            arm_states[timestamp] = state
    return arm_states

def generate_states(state_log_folder):
    with open(state_log_folder, 'r') as file:
        log_contents = file.readlines()
    joint_num = 19
    num = 0
    state = jointstate()
    jointstate_list = []
    start = False
    for ind, line in enumerate(log_contents):
        if 'kinematic_state' in line:
            start =True
            num = num +1
            output_log_file = f'jointstate_{num}.log'

        if start:
            if 'joint_states' in line:
                state.name = log_contents[ind + 1].split(':')[-1][2:-1]
                if state.name == 'arm0.hr0':
                    continue
                state.position = log_contents[ind + 3].split(':')[-1]
                state.velocity = log_contents[ind + 6].split(':')[-1]
                state.acceleration = log_contents[ind + 9].split(':')[-1]
                state.load = log_contents[ind + 12].split(':')[-1]
                jointstate_list.append(state)
            if len(jointstate_list) == joint_num:
                start = False
                with open(output_log_file, 'w') as log_file:
                    log_file.write(jointstate_list)



char_to_num = {
    'W': 0, 'A': 1, 'S': 2, 'D': 3,
    'w':4,'a':5,'s':6,'d':7,
    'R':8,'F':9,'I':10,'K':11,
    'U':12,'O':13,'J':14,'L':15,
     'q':16,'e':17,
     'M':18,'N':19}

action_list = char_to_num.keys()
def load_actions(action_log_file):
    with open(action_log_file, 'r') as file:
        lines = file.readlines()
    actions = [(float(line.split('-')[0]), line.split('-')[1].strip()) for line in lines]

    filter_acts = []

    record = False
    finished = False
    for a in actions:
        #print(a)
        if a[1]=='i': #f
            record = True

        if record and (a[1]=='h' or a[1]=='p'):
            finished = True
            break

        if record and (a[1] in action_list):
            filter_acts.append(a)
    return filter_acts, finished

def closest_timestamp(timestamps, target_timestamp):
    """Find the closest timestamp to target_timestamp but less than target_timestamp."""
    closest = None
    for ts in timestamps:
        if ts < target_timestamp and (closest is None or (target_timestamp - ts < target_timestamp - closest)):
            closest = ts
    return closest

def synchronize(arm_folder, action_log_file, camera_folders):
    arm_states = load_arm_states(arm_folder)
    actions, finished = load_actions(action_log_file)
    
    synchronized_data = []
    
    for action_timestamp, action in actions:

        # Find closest arm state timestamp
        closest_arm_ts = closest_timestamp(arm_states.keys(), action_timestamp)
        
        # Find closest image timestamps for each camera
        camera_images = []#{}
        for camera_folder in camera_folders:
            image_files = os.listdir(camera_folder)
            image_timestamps = [float(filename.split('.png')[0]) for filename in image_files]
            image_names = [filename for filename in image_files]
           

            closest_image_ts = closest_timestamp(image_timestamps, action_timestamp)
            if closest_image_ts:
                camera_images.append(os.path.join(camera_folder, image_names[image_timestamps.index(closest_image_ts)] ))

        if len(camera_images)<4:
            print(f'img is not 4, only got:{len(camera_images)} in',camera_images[0])
            continue

        arm_joint_positions = []
        for joint in arm_states.get(closest_arm_ts):
            arm_joint_positions.append(joint.position.value)

        synchronized_data.append({
            'action_timestamp': action_timestamp,
            'action': action,
            'arm_state_timestamp': closest_arm_ts,
            'arm_state': arm_joint_positions,
            'camera_images': camera_images
        })
    
    return synchronized_data, finished
###########################################
##input: datafolderpath
##output:
#####demostration: tra_num,action_sq,final_reward
##############action_timestamp:float
##############action:str[f,i,WSAD，wsad] R,F,
##############arm_state_time:float
##############arm_state:list[float]
##############cameraimage:list[str]
#########################################
def extract_demonstrations(base_path= '/home/carol/Project/off-lineRL/spot_data/real/data/log-v6/'):
    sub_folder_list = glob.glob(base_path+'*')

    demonstrations = []
    for sub_path in sub_folder_list:
        try:
            arm_folder = sub_path + '/arms'
            #state_log_file = sub_path + '/robot_state_log.log'
            action_log_file = sub_path + '/action_log.log'
            camera_folders = [
                            sub_path+'/image/frontleft_depth_in_visual_frame',
                            sub_path+'/image/frontleft_fisheye_image',
                            sub_path+'/image/frontright_depth_in_visual_frame',
                            sub_path+'/image/frontright_fisheye_image']

            synchronized_data, finished = synchronize(arm_folder, action_log_file, camera_folders)
            if len(synchronized_data) ==0:
                print(action_log_file)

            demonstrations.append([synchronized_data, finished])
            #generate_states(state_log_file)
        except:
            print(sub_path, 'does not have proper data')
            continue
        # Print or save the synchronized data as needed

    return demonstrations
