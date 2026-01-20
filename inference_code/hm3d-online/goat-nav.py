from collections import defaultdict
import gzip
import os
import habitat
from habitat.utils.visualizations import maps
from habitat_sim import Simulator as Sim
import json
import habitat_sim
import numpy as np
from habitat.tasks.nav.nav import TopDownMap

from frontier_utils import convert_meters_to_pixel, detect_frontier_waypoints, get_closest_waypoint, get_polar_angle, map_coors_to_pixel, pixel_to_map_coors, reveal_fog_of_war
from sim_utils import get_simulator
import cv2
import random
from utils import resize_with_padding,draw_dashed_line,log_message,add_arguments,load_navigation_data_goat,get_info_goat,make_log_file,get_sim_agent,make_video_goat
from qwen_utils_goat import getresult,Bank
import torch

import numpy as np
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
# hyperparameter
data_set_path = "./MTU3D/dataset/embodied_bench_data/embodied_bench_data/our-set/goat_full_set.json"
navigation_data_path = "./MTU3D/dataset/embodied_bench_data/embodied_bench_data/goat"
hm3d_data_base_path = "./MTU3D/dataset/hm3d/val"
embodied_scan_dir = "./MTU3D/dataset/embodied_scan"
output_path =  "./output_dirs/goat-full-finetune-num-1.json"
log_dir="./mturesult/log"
sim_setting_path='configs/habitat/goat_sim_config.yaml'
goat_agent_setting_path='configs/habitat/goat_agent_config.yaml'
video_path='./mtuqwen/MTU3D/mturesult/video'
val_split=["val_unseen"]

decision_num_min = 3
visible_radius = 10
generated_avi_files = []
min_pixels = 56 * 56
max_pixels = 4480 * 4480
map_resolution = 512
REPRODUCIBLE_SEED = 42

# args = add_arguments()
# pattern = args.pattern
# model_path=args.model_path
# name=args.name
enable_visualization = True
pattern = 'notall'
model_path='./checkpoint-20000'
name='args_name'

# load navigation data
split_list,navigation_data_dict=load_navigation_data_goat(embodied_scan_dir,navigation_data_path)
qwen=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",attn_implementation="flash_attention_2"
    )
processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
  
# record result
if os.path.exists(output_path):
    result_dict = json.load(open(output_path, "r"))
else:
    result_dict = {'val_seen': [], 'val_seen_synonyms': [], 'val_unseen': []}
data_set = json.load(open(data_set_path, "r"))
for split in split_list:
    existing_episodes = {(result['scan_id'], result['episode_index']) for result in result_dict[split]}
    data_set[split] = [episode for episode in data_set[split] if (episode['scan_id'], episode['episode_index']) not in existing_episodes]


for split in split_list:
    if split not in val_split:
        continue
    random.seed(REPRODUCIBLE_SEED)
    if pattern =='all':
        split_list=[]
        for key,scene in navigation_data_dict[split].items():
            num=0
            for item in scene['episodes']:
                split_list.append(
                    {"scan_id":key,
                    "episode_index":num,
                    "object_category":item['object_category'],
                    }
                )
                num=num+1
        episode_list_to_process = list(split_list)
    else:    
        episode_list_to_process = list(data_set[split])

    random.shuffle(episode_list_to_process)
    for cur_data in episode_list_to_process:

        scene_id,scene_path,episode_index,cur_episode=get_info_goat(cur_data,hm3d_data_base_path,navigation_data_dict,split)

        # get simulator
        sim,agent,path_finder,start_position,start_rotation=get_sim_agent(sim_setting_path,goat_agent_setting_path,scene_path,cur_episode)
        # get fronier param

        top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=map_resolution,  draw_border=False)
        fog_of_war_mask = np.zeros_like(top_down_map)
        area_thres_in_pixels =  convert_meters_to_pixel(9, map_resolution, sim)
        visibility_dist_in_pixels = convert_meters_to_pixel(visible_radius, map_resolution, sim)
        
        # start decision
        decision_num = 0
        global_color_list,global_frontier_list,global_state_list = [],[],[]



        visited_frontier_set = set()


        # visited frontier
        bank=Bank(goto_sample_interval=4)
        color_past=None
        agent_past=None
        name_list=[]
        # start for loop
        for sub_episode_index in range(len(cur_episode['tasks'])):
            # build goal
            cur_sub_episode = cur_episode['tasks'][sub_episode_index]
            if (len(cur_sub_episode) == 3):
                goal_category, goal_type, goal_object_id = cur_sub_episode
            else:
                goal_category, goal_type, goal_object_id, goal_image_id = cur_sub_episode
            if goal_type == 'object':
                sentence = goal_category
                goals = [g for g in navigation_data_dict[split][scene_id]['goals_by_category'][goal_category]]
            elif goal_type == 'description':
                goals = [g for g in navigation_data_dict[split][scene_id]['goals_by_category'][goal_category] if g['object_id'] == goal_object_id]
                sentence = goals[0]['lang_desc']
                assert len(goals) == 1
            elif goal_type == 'image':
                sentence = goal_category
                goals = [g for g in navigation_data_dict[split][scene_id]['goals_by_category'][goal_category] if g['object_id'] == goal_object_id]
                # goal_image_feat = image_feat_dict[split][scene_id][int(goal_object_id.split('_')[1])][goal_image_id]
                assert len(goals) == 1
            goto_color_list,goto_agent_state_list  = [],[]
            all_view_points = [
                vp["agent_state"]["position"]
                for goal in goals
                for vp in goal["view_points"]
            ]
            episode_cum_distance = 0
            prev_agent_state = agent.get_state()
            sub_episode_start_position = prev_agent_state.position
            total_steps = 0
            log_file_path,timestamp=make_log_file(log_dir,split,name,scene_id,cur_episode['episode_id'],sentence)

            while total_steps < 2000:

                if goto_color_list: # 确保不是第一次循环
                    bank.add(goto_color_list, goto_agent_state_list, data_type='goto')
                    # 清空 goto 列表，为下一个 goto 做准备
                    goto_color_list, goto_depth_list, goto_agent_state_list = [], [], []                
                # spin around
                spin_color_list = []
                spin_agent_state_list = []
                action_list = ['turn_left'] * 12
                for action in action_list:
                    obervations = sim.step(action=action)
                    agent_state = agent.get_state()
                    color = obervations['color_sensor'][:, :, :3]
                    color= cv2.cvtColor(color, cv2.COLOR_BGR2RGB)  # (h,w,4) 0-255
                    spin_color_list.append(color)
                    fog_of_war_mask,vis_obstacles_mask = reveal_fog_of_war(top_down_map=top_down_map, current_fog_of_war_mask=fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=110, max_line_len=visibility_dist_in_pixels, enable_debug_visualization=enable_visualization)
                    frontier_waypoints,frontier = detect_frontier_waypoints(top_down_map, fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=110, max_line_len=visibility_dist_in_pixels,area_thresh=area_thres_in_pixels, xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1], enable_visualization=enable_visualization,tag='spin')
                    tag = 'all' 
                    OTHER_VIEWPOINT_COLOR = (255, 165, 0)
                    OTHER_VIEWPOINT_RADIUS = 3
                    for vp in all_view_points:
                        map_y, map_x = map_coors_to_pixel(vp, top_down_map, sim)
                        cv2.circle(frontier, (map_x, map_y), radius=OTHER_VIEWPOINT_RADIUS, color=OTHER_VIEWPOINT_COLOR, thickness=-1)
                    global_frontier_list.append((frontier, 'spin',sentence))
                    spin_agent_state_list.append(agent_state)
                    if enable_visualization:
                        cv2.imwrite('color.png', color)
                    total_steps += 1
                bank.add(spin_color_list[2::3], spin_agent_state_list[2::3], data_type='spin')    
                global_color_list.extend(spin_color_list)
                global_state_list.extend(spin_agent_state_list)
                agent_state = agent.get_state()
                fog_of_war_mask,vis_obstacles_mask = reveal_fog_of_war(top_down_map=top_down_map, current_fog_of_war_mask=fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=110, max_line_len=visibility_dist_in_pixels, enable_debug_visualization=enable_visualization)
                frontier_waypoints,frontier = detect_frontier_waypoints(top_down_map, fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=110, max_line_len=visibility_dist_in_pixels,area_thresh=area_thres_in_pixels, xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1], enable_visualization=enable_visualization,tag='spin')
                
                if len(frontier_waypoints) == 0:
                    frontier_waypoints = []
                else:
                    frontier_waypoints = frontier_waypoints[:, ::-1]
                    frontier_waypoints = pixel_to_map_coors(frontier_waypoints, agent_state.position, top_down_map, sim)
                frontier_waypoints = [waypoint for waypoint in frontier_waypoints if tuple(np.round(waypoint, 1)) not in visited_frontier_set] 
                # decision
                log_message(f"total_steps:{total_steps}",log_file_path,timestamp)
                log_message(sentence,log_file_path,timestamp)
                target_position, is_final_decision,output_texts = getresult(
                                                                    qwen,
                                                                    processor,
                                                                    bank,
                                                                    log_file_path,
                                                                    current_frontiers=frontier_waypoints,
                                                                    decision_agent_state=agent_state, # 传入决策时的状态作为参考系
                                                                    object_category=sentence,
                                                                    decision_num=decision_num,
                                                                    visited_frontier_set=visited_frontier_set,
                                                                    use_compression=True,
                                                                    compression_times=2,
                                                                    dino_size='vitb16',
                                                                    image_length=50,
                                                                ) 
            
                print(f"Model decision: target_position={target_position}, is_final_decision={is_final_decision}")            
                log_message(f"Model decision: target_position={target_position}, is_final_decision={is_final_decision},\n output:{output_texts}",log_file_path,timestamp)
                decision_num += 1
                try:
                    if not is_final_decision:
                        visited_frontier_set.add(tuple(np.round(target_position, 1)))
                    agent_island = path_finder.get_island(agent_state.position)
                    target_on_navmesh = path_finder.snap_point(point=target_position, island_index=agent_island)
                    follower = habitat_sim.GreedyGeodesicFollower(path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right")
                    action_list = follower.find_path(target_on_navmesh)
                except Exception as e:
                    log_message(f"start backed path...",log_file_path,timestamp)

                    path_found = False 
                    action_list = []    
                    alternative_waypoints = list(frontier_waypoints)
                    random.shuffle(alternative_waypoints)

                    for alternative_target in alternative_waypoints:
                        if (alternative_target == target_position).all():
                            continue    
                        print(f"alternative_target {alternative_target}")         
                        try:
                            alt_target_on_navmesh = path_finder.snap_point(point=alternative_target, island_index=agent_island)
                            action_list = follower.find_path(alt_target_on_navmesh)
                            log_message(f"sucess backed path: {alternative_target} ",log_file_path,timestamp)
                            

                            target_position = alternative_target
                            target_on_navmesh = alt_target_on_navmesh
                            
                            path_found = True 
                            break 

                        except Exception as fallback_e:
                            log_message(f"fail alternative_target:{alternative_target} : {fallback_e}",log_file_path,timestamp)
                            continue

                    if not path_found:
                        break 

                goto_color_list ,goto_agent_state_list= [],[]
                if action_list==None:
                    break
                for action in action_list:
                    if action:
                        obervations = sim.step(action=action)
                        global_color_list.append(obervations['color_sensor'][:, :, :3])
                        agent_state = agent.get_state()
                        color = obervations['color_sensor'][:, :, :3]
                        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) # (h,w,4) 0-255
                        goto_color_list.append(color)
                        goto_agent_state_list.append(agent_state)
                        fog_of_war_mask,vis_obstacles_mask = reveal_fog_of_war(top_down_map=top_down_map, current_fog_of_war_mask=fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=110, max_line_len=visibility_dist_in_pixels, enable_debug_visualization=enable_visualization)
                        total_steps += 1
                        episode_cum_distance += np.linalg.norm(agent_state.position - prev_agent_state.position)
                        prev_agent_state = agent_state
                        frontier_waypoints,frontier = detect_frontier_waypoints(top_down_map, fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=110, max_line_len=visibility_dist_in_pixels,area_thresh=area_thres_in_pixels, xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1], enable_visualization=enable_visualization,tag='goto')
                        OTHER_VIEWPOINT_COLOR = (255, 165, 0)
                        OTHER_VIEWPOINT_RADIUS = 3
                        for vp in all_view_points:
                            map_y, map_x = map_coors_to_pixel(vp, top_down_map, sim)
                            cv2.circle(frontier, (map_x, map_y), radius=OTHER_VIEWPOINT_RADIUS, color=OTHER_VIEWPOINT_COLOR, thickness=-1)
                        global_frontier_list.append((frontier, 'goto',sentence))
                global_state_list.extend(goto_agent_state_list)
            # break on final decision
                if is_final_decision:
                    log_message('subgoal complete___________________________________________________________\n',log_file_path,timestamp)
                    break   
                if goto_color_list==[]:
                    print(' no goto_color_list')
                    # log_message(' no goto_color_list')
                    break 
                color_past=goto_color_list[-1]
                agent_past=goto_agent_state_list[-1]
            agent_state = agent.get_state()
            view_points = [
                        view_point["agent_state"]["position"]
                        for goal in goals
                        for view_point in goal["view_points"]
            ]
            # computer start end geodesic distance
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = start_position
            path.requested_ends = view_points
            # log_message(view_points)
            if path_finder.find_path(path):
                start_end_geo_distance = path.geodesic_distance
            else:
                log_message("goal is not navigatable",log_file_path,timestamp)
                # print("goal is not navigatable")
                start_end_geo_distance = np.inf
            # compute agent current distance
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = agent_state.position
            path.requested_ends = view_points
            if path_finder.find_path(path):
                agent_end_geo_distance = path.geodesic_distance
            else:
                log_message("agent_end_geo_distance is inf",log_file_path,timestamp)
                agent_end_geo_distance = np.inf
            # compute success rate
            if start_end_geo_distance == np.inf:
                sr = 1
                spl = 1
            elif agent_end_geo_distance == np.inf:
                sr = 0
                spl = 0
            else:
                log_message(f'agent_end_geo_distance:{agent_end_geo_distance} 1',log_file_path,timestamp)
                sr = agent_end_geo_distance <= 1
                spl = sr * start_end_geo_distance / max(start_end_geo_distance, episode_cum_distance)
            result_dict[split].append({'scan_id': scene_id, 'episode_index': episode_index, 'sr': sr, 'spl': spl})
            name_list.append(agent_end_geo_distance <= 1)
            log_message(f"Episode finished. SR: {sr}, SPL: {spl}, Steps: {total_steps}, Decision Num: {decision_num}",log_file_path,timestamp)   
        if enable_visualization:   
             make_video_goat(name_list,start_position,path_finder,global_color_list, global_frontier_list,video_path,split,sr,scene_id,cur_episode,log_file_path,global_state_list,top_down_map,sim)

        with open(output_path, 'w') as f:
            json.dump(result_dict, f)

for split in split_list:
    total_sr = 0
    total_spl = 0
    category_sr_spl = defaultdict(lambda: {'sr': 0, 'spl': 0, 'count': 0})
    for result in result_dict[split]:
        total_sr += result['sr']
        total_spl += result['spl']
        category = result['object_category']
        # category_sr_spl[category]['sr'] += result['sr']
        # category_sr_spl[category]['spl'] += result['spl']
        # category_sr_spl[category]['count'] += 1

    avg_sr = total_sr / len(result_dict[split])
    avg_spl = total_spl / len(result_dict[split])
    print(f"Split: {split}, Average SR: {avg_sr}, Average SPL: {avg_spl}")

    for category, metrics in category_sr_spl.items():
        avg_category_sr = metrics['sr'] / metrics['count']
        avg_category_spl = metrics['spl'] / metrics['count']
        print(f"Category: {category}, Average SR: {avg_category_sr}, Average SPL: {avg_category_spl}")
        