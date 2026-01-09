
import numpy as np
from PIL import Image

def process_vision_info(colorlist):
    """处理消息中的图像和视频信息"""
    image_inputs = []
    video_inputs = []
    
    for image in colorlist:
        # for pixel_values in image['image']:    
            image = Image.fromarray(image)
            image = image.resize((720,640))
            image_inputs.append(image)
                        
    
    return image_inputs, video_inputs
import numpy as np
from scipy.spatial.transform import Rotation as R
import re
import json

def parse_response(response_text):
    """
    解析一个字符串，根据特定规则返回结果。

    规则:
    1. 如果字符串中包含 "found"，则返回 (None, True)。
    2. 如果不包含 "found"，则查找并解析第一个出现的 "<coordinate>[...]" 模式。
       - 如果找到，返回 (解析出的坐标列表, False)。
       - 如果找不到，返回 (None, False)。

    Args:
        response_text (str): 需要解析的输入字符串。

    Returns:
        tuple[list | None, bool]: 一个元组，包含两个值：
                                 - 第一个是坐标列表或None。
                                 - 第二个是布尔值，表示是否找到了 "found"。
    """
    # 规则 1: 检查是否包含 "found"
    # 使用 'in' 关键字进行简单的子字符串检查，高效且直接。


    # 规则 2: 如果没有 "found"，则匹配坐标
    # 定义正则表达式模式
    # - <coordinate>: 匹配字面量
    # - (\[.*?\]): 这是一个捕获组 (用圆括号包围)。
    #   - \[ 和 \]: 匹配字面量的方括号。
    #   - .*?: 非贪婪匹配，匹配方括号内的任意字符，直到第一个 ']' 出现。
    # - </coordinate>: 匹配字面量
    pattern = re.compile(r"<coordinate>\s*(\[.*?\])\s*</coordinate>")
    
    match = pattern.search(response_text)
    is_found = "found" in response_text
    
    if match:
        coord_str = match.group(1)
        try:
            # json.loads 可以完美处理 [-1.11, 1.11] 这样的字符串
            coord_list = json.loads(coord_str)
            
            if isinstance(coord_list, list):
                return coord_list, is_found
        except json.JSONDecodeError:
            print(f"警告: 无法解析坐标内容 '{coord_str}'")
            return None, is_found
            
    # 如果没匹配到坐标，但包含 found，则返回 (None, True)
    # 或者根据你的原始逻辑处理
    return None, is_found


import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R
def pose_to_matrix(pose):
    position = np.array(pose['position'])
    rotation = np.array(pose['rotation'])

    rotation_matrix = R.from_quat(rotation[[1, 2, 3, 0]]).as_matrix()

    rot_normal_raw = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rotation_matrix = rotation_matrix @ rot_normal_raw
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix
    pose[:3, 3] = position
    return pose

def transform_poses(world_coord, world_quat, current_world_coord, current_world_quat):
    world_quat={
        'x': world_quat.x, 'y': world_quat.y,
        'z': world_quat.z, 'w': world_quat.w
    }
    current_pose = {'position': current_world_coord, 'rotation': [current_world_quat['w'], current_world_quat['x'], current_world_quat['y'],
                                                          current_world_quat['z']]}
    input_poses = [{'position': world_coord, 'rotation': [world_quat['w'], world_quat['x'],
                                                          world_quat['y'], world_quat['z']]}]
    current_pose = pose_to_matrix(current_pose)
    poses = [pose_to_matrix(pose) for pose in input_poses]
    poses = np.array(poses)
    rotation_matrix = np.array([[1,0,0],[0,0,-1],[0,1,0]])

    current_pose_array = current_pose.copy()
    poses = poses.copy()

    current_pose_array[:3,:3] = current_pose_array[:3,:3] @ rotation_matrix
    poses[:,:3,:3] = poses[:,:3,:3] @ rotation_matrix

    current_pose_matrix = current_pose_array
    current_pose_inv = np.linalg.inv(current_pose_matrix)
    pose_matrices = poses
    transformed_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], pose_matrices)

    rotation = transformed_matrices[:,:3,:3]
    positions = transformed_matrices[:, :3, 3].tolist()
    yaw = np.arctan2(-rotation[:, 0, 1], rotation[:, 1, 1])

    positions = [positions[0][0], positions[0][1]]

    return {'local_coord': positions, 'rotation':rotation.tolist(), 'yaw':yaw.tolist()[0]}


def transform_to_local_frame(world_point, agent_world_coord, agent_world_quat):
    world_point = np.array(world_point)
    agent_world_coord = np.array(agent_world_coord)
    relative_point = world_point - agent_world_coord
    quat_xyzw = [agent_world_quat['x'], agent_world_quat['y'], agent_world_quat['z'], agent_world_quat['w']]
    agent_rotation = R.from_quat(quat_xyzw)
    local_point = agent_rotation.apply(relative_point, inverse=True)
    local_point = local_point.round(2).tolist()
    res = [local_point[0] ,local_point[1]* -1, local_point[2] * -1]
    return res

def transform_from_local_frame(local_point, agent_world_coord, agent_world_quat):

    local_point = np.array([
        local_point[0] ,
        local_point[1]* -1,
        local_point[2] * -1
    ])
    agent_world_coord = np.array(agent_world_coord)

    quat_xyzw = [agent_world_quat['x'], agent_world_quat['y'], agent_world_quat['z'], agent_world_quat['w']]
    agent_rotation = R.from_quat(quat_xyzw)
    relative_point_in_world = agent_rotation.apply(local_point)
    reconstructed_world_point = relative_point_in_world + agent_world_coord

    return reconstructed_world_point



obj_goal_template = [
"Find a {} in your immediate surroundings and stop when you see one.", 
"Explore the area until you locate a {}. Stop when you've reached its location.",
"Move through the environment to discover a {}. Your task is complete when you're directly facing it.",
"Navigate to any visible {}. Stop immediately upon successful discovery.",
"Search for an instance of {} within this space. Terminate navigation once you've positioned yourself within arm's reach of it.",
"Survey the surroundings until you identify a {}. Stop navigating as soon as you are positioned directly in front of it",
"Roam through the space until a {} is spotted. Terminate navigation the moment you’re certain you’re facing it.",
"Go to the {}, then stop at the front of it.",
"Move to the nearst {}, then stop",
"Navigate to a nearst {}, then stop over there.",
"Get close to the {}, then stop",
"Could you help me find a {}? Show me the way"]
# obj_goal_template = [
#     "Find a {} in your immediate surroundings and stop when you see one.", "Explore the area until you locate a {}. Stop when you've reached its location.",
#     "Move through the environment to discover a {}. Your task is complete when you're directly facing it.", "Navigate to any visible {}. Stop immediately upon successful discovery.",
#     "Search for an instance of {} within this space. Terminate navigation once you've positioned yourself within arm's reach of it.",
#     "Survey the surroundings until you identify a {}. Stop navigating as soon as you are positioned directly in front of it", "Roam through the space until a {} is spotted. Terminate navigation the moment you’re certain you’re facing it.",
#     "Go to the {}, then stop at the front of it.", "Move to the nearst {}, then stop", "Navigate to a nearst {}, then stop over there.",
#     "Get close to the {}, then stop", "Could you help me find a {}? Show me the way"
# ]
import random


import datetime

def getresult(qwen,processor,bank,log_file_path, current_frontiers, decision_agent_state, object_category, decision_num,
               visited_frontier_set, use_compression=True, compression_times=2, dino_size='vitb16', image_length=50):
    def log_message(message):
        with open(log_file_path, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n") 
    ref_coord = decision_agent_state.position
    ref_quat_dict = {
        'x': decision_agent_state.rotation.x, 'y': decision_agent_state.rotation.y,
        'z': decision_agent_state.rotation.z, 'w': decision_agent_state.rotation.w
    }
    all_prompt_images = []

    historical,_=bank.get_spin_data()
    if  len(historical)<5:
            historical_spin_images,_=bank.get_spin_data()
            part1_content = "This is the first observation point, there are no historical images."
    else:
        # transform_poses
        historical_spin_images,historical_spin_states=bank.get_spin_data()
        historical_spin_states_45=historical_spin_states[-image_length+1:-4]
        history_image_coordinate=[transform_poses(item.position,item.rotation,ref_coord, ref_quat_dict) for item in historical_spin_states_45]
        history_image_placeholders="".join([f"[{item['local_coord'][0]:.2f}, {item['local_coord'][1]:.2f},yaw:{item['yaw']:.2f}]<image>" for item in history_image_coordinate])
        part1_content = f"Firstly, we provide all the historical memory images: {history_image_placeholders}"


    all_prompt_images.extend(historical_spin_images[-image_length+1:])
    part2_content = (f"Then, I have four images that form a 360-degree panoramic view around a central location.\n"
                        f"Each image has a field of view (FOV) of 110 degrees and is taken at 90-degree intervals. "
                        f"The fourth image defines the reference orientation (its forward direction is 0 degrees)."
                        f"The first three images are:<image><image><image>\n"
                        f"The fourth reference image is:<image>")




    local_to_global_map = {}
    if current_frontiers:
        frontier_parts = ['The coordinates of the explorable frontiers are: ']
    else:
        frontier_parts = ['No explorable frontiers found. Directly identify the target object or location from the user instruction and output the target']
    for frontier_coord in current_frontiers:
        local_coord = transform_to_local_frame(frontier_coord, ref_coord, ref_quat_dict)
        x, z = local_coord[0], local_coord[2]
        local_to_global_map[(x, z)] = frontier_coord
        frontier_parts.append(f"[{x:.2f}, {z:.2f}]")

    frontier_content = ''.join(frontier_parts)

    task_sentence = "instruction: "+object_category

    user_content = "\n".join([part1_content,part2_content,frontier_content,task_sentence])
    
    current_messages=[]
 
    msg_copy={}
    log_message(user_content)
    if len(current_messages) == 0: 
        content = []
        content.append({"type": "image", "image": all_prompt_images})
        content.append({"type": "text", "text":user_content})
        msg_copy['content'] = content
        current_messages.append(msg_copy)
    current_messages[0]['role']='user'
    text = processor.apply_chat_template(
        current_messages, tokenize=False, add_generation_prompt=True
    )
    text=text.replace('<|vision_start|><|image_pad|><|vision_end|>','')
    text=text.replace('<image>','<|vision_start|><|image_pad|><|vision_end|>')

    image_inputs, video_inputs = process_vision_info(all_prompt_images)

    from transformers import AutoTokenizer


    inputs = processor(
        text=text,
        images=image_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
        use_compression=use_compression,
        compression_times=compression_times,
        dino_size=dino_size
    )
    device = qwen.device 
    inputs = inputs.to(device)
    # print(frontier_content)
    print(text.count('image_pad'))
    with torch.no_grad():
        generated_ids = qwen.generate(**inputs, max_new_tokens=512)
        # test=qwen_nav.forward(**inputs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        parsed_local_target,is_final_decision=parse_response(output_texts[0])
    print(output_texts)
    if parsed_local_target is None:
        if  local_to_global_map is None and current_frontiers is not None:
            print("警告：没有可用的候选点或模型未能解析出坐标。")
            return current_frontiers[0] if current_frontiers else None, False
    model_choice_coord = np.array(parsed_local_target)
    
    min_dist = float('inf')
    best_match_global_coord = None

    if is_final_decision:
        model_choice_coord= np.insert(model_choice_coord, 1, 0)
        best_match_global_coord=transform_from_local_frame(model_choice_coord,ref_coord, ref_quat_dict)
    else:
        for local_key, global_coord_val in local_to_global_map.items():
            dist = np.linalg.norm(model_choice_coord - np.array(local_key))
            
            if dist < min_dist:
                min_dist = dist
                best_match_global_coord = global_coord_val

    target_position0 = best_match_global_coord

    return target_position0, is_final_decision,output_texts



import torch

class Bank:
    def __init__(self, goto_sample_interval=3):
        self.images_spin = []
        self.agent_states_spin = []
        self.images_goto = []
        self.agent_states_goto = []
        self.unsampled_goto_images = []
        self.unsampled_goto_agent_states = []

        self.goto_sample_interval = goto_sample_interval

    def add(self, images, agent_states, data_type='spin'):
        """
        向记忆库中添加数据。

        Args:
            images (list): 图像列表。
            agent_states (list): 对应的智能体状态列表。
            data_type (str): 'spin' 或 'goto'。
        """
        if not images:
            return

        if data_type == 'goto':
            # --- 【核心修改】处理 goto 数据 ---
            sampled_indices = set(self._get_sampled_indices(len(images)))

            # 遍历所有传入的数据，进行分流
            for i, (img, state) in enumerate(zip(images, agent_states)):
                if i in sampled_indices:
                    # 如果索引在采样集中，存入主历史记录
                    self.images_goto.append(img)
                    self.agent_states_goto.append(state)
                else:
                    # 否则，存入“未采样”的备份区
                    self.unsampled_goto_images.append(img)
                    self.unsampled_goto_agent_states.append(state)
        
        else: # data_type == 'spin'
            # spin 数据逻辑保持不变，全部保留在主历史记录中
            self.images_spin.extend(images)
            self.agent_states_spin.extend(agent_states)


    def get_spin_data(self):
        """返回当前存储的所有图像和状态。"""
        return self.images_spin, self.agent_states_spin
    def get_goto_data(self):
        """返回当前存储的所有图像和状态。"""
        return self.images_goto, self.agent_states_goto

    def get_unsampled_goto_data(self):
        """
        【新增】获取在 goto 过程中被丢弃的未采样数据。
        
        Returns:
            tuple: (list_of_unsampled_images, list_of_unsampled_agent_states)
        """
        return self.unsampled_goto_images, self.unsampled_goto_agent_states

    def clear(self):
        """
        【更新】清空记忆库中的所有数据。
        """
        self.images_spin.clear()
        self.agent_states_spin.clear()
        self.images_goto.clear()
        self.agent_states_goto.clear()
        self.unsampled_goto_images.clear()
        self.unsampled_goto_agent_states.clear()
        print("Bank memory cleared.")

    def _get_sampled_indices(self, num_images):
        """
        计算采样索引的辅助函数。
        保留首尾，中间间隔 self.goto_sample_interval 个。
        """
        if self.goto_sample_interval==1:
            return list(range(num_images))
        if num_images <= 2:
            return list(range(num_images))

        sampled_indices = {0, num_images - 1}
        step = self.goto_sample_interval + 1
        for i in range(step, num_images - 1, step):
            sampled_indices.add(i)
        
        return sorted(list(sampled_indices))

    def __len__(self):
        return len(self.images)

