import enum
import open3d as o3d
import numpy as np
import pathlib
import copy
import yaml
import time
from PIL import Image
from matplotlib import  pyplot as plt  
import pandas as pd


def load_binary_cloud(file_path):
    """ Loads a binary point cloud with intensity values.
    Intensity values are dropped and the point cloud is stored 
    as an open3d point cloud object

    Args:
        file_path (string): [Absolute path to the '.bin' point cloud]

    Returns:
        [open3d.geometry.PontCloud]: [Returns the pointcloud data in an open3d format]
    """
    raw_data = np.fromfile(file_path, dtype=np.float32)
    # Remove pont cloud intensity values
    raw_points = raw_data.reshape((-1, 4))[:, 0:3]
    point_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(raw_points))
    return point_cloud


def binary_clouds_to_pcd(input_path, output_path):
    """ Converts all '.bin' point cloud files in a folder to the
    pcd format

    Args:
        input_path (string): Input folder path
        output_path (string): Output folder path
    """
    for point_cloud_file in pathlib.Path(input_path).glob('*.bin'):
        point_cloud = load_binary_cloud(point_cloud_file)
        pcd_filename = pathlib.Path(
            output_path + '/' + point_cloud_file.name).with_suffix('.pcd')
        o3d.io.write_point_cloud(str(pcd_filename), point_cloud)


def draw_icp_result(source, target, transformation):
    """Displays the source and the target clouds in same
    frame. The source is shown in orange and the target is
    shown in blue

    Args:
        source (open3d.geometry.PontCloud): [source  (model) point cloud for icp]
        target (open3d.geometry.PontCloud): [target (data) point cloud for icp]
        transformation (np.array((4,4))): [4x4 transformation matrix between the two clouds]
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    target_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])



def custom_draw_geometry(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("/home/drew/Desktop/capstone_spring/global_icp/images/testimage6.png")
    vis.destroy_window()
 

def custom_draw_geometry_with_key_callback(pcd):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "../../TestData/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        print("saving image!!!")
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.savefig("/home/drew/Desktop/capstone_spring/global_icp/images/test_image.png")
        # plt.show()
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

def load_config(config_file):
    """[Loads a configuration file]

    Args:
        config_file ([string]): [Config file location]

    Returns:
        [dictonary]: [Returns a dictionary of configuration parameters]
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        # try:
        #     parameter_bounds = tuple(config['bayes_opt']['parameter_bounds'])
        #     for key in parameter_bounds:
        #         parameter_bounds[key] = tuple(parameter_bounds[key])
        #     config['bayes_opt']['parameter_bounds'] = parameter_bounds
        # except KeyError as e:
        #     print("Config valid is invalid please make sure there is valid entry for", e)

        return config


def get_parameter(params, type, key):
    """[summary]

    Args:
        params ([dictionary]): [dictionary of parameters]
        type ([string]): [type of parameter]
        key ([string]): [key for parameter]

    Returns:
        [string]: [the value of the parameter]
    """
    value = None
    try:
        value = params[type][key]
        return value
    except KeyError as e:
        print("Config valid is invalid please make sure there is valid entry for", e)
        return value

def get_translation_params(params):
    """[Extracts the x,y, and z paramters]

    Args:
        params ([dictionar]): [parameter bounds]

    Returns:
        [dictionary]: [dictonary containing the paramter bounds for x,y,z]
    """
    translation_vals = ["x", "y", "z"]
    return {key: params[key] for key in translation_vals}

def get_roatation_params(params):
    """[Extracts the rotation parameters (roll, pitch, yaw)]

    Args:
        params ([dictionary]): [paramter bounds]

    Returns:
        [dictonary]: [dictionary containg the rotation bounds (roll, pitch, yaw)]
    """
    rotation_vals = ["roll", "pitch", "yaw"]
    return {key: params[key] for key in rotation_vals}

def random_downsample(pointcloud, num_points):
    """[Randomly Downsamples a pointlcoud to a specific number of points]

    Args:
        pointcloud ([o3d.pointcloud]): [pointcloud to downsample]
        num_points ([float]): [description]
    Returns:
        [o3d.pointlcoud]: [downsampled pointcloud]
    """
    current_points = np.asarray(pointcloud.points).shape[0]
    ratio = float(num_points)/float(current_points)
    downsampled_cloud = copy.deepcopy(pointcloud)
    return downsampled_cloud.random_down_sample(ratio)

def scale_point_cloud(pointcloud, box_size):
    """[summary]

    Args:
        pointcloud ([o3d.geometry.pointcloud]): [pointcloud to scale]
        box_size ([float]): [size of the box to scale the cloud into]

    Returns:
        [o3d.geometry.pointcloud]: [scaled pointcloud]
    """
    max_dim = np.max(pointcloud.get_max_bound())
    print(max_dim)
    scale = float(box_size)/float(max_dim)
    scaled_cloud = copy.deepcopy(pointcloud).scale(scale, pointcloud.get_center())
    return scaled_cloud

def export_icp_image(source_cloud, target_cloud, transformation, path):
    """[exports a frame of icp alignment]

    Args:
        source_cloud ([o3d.geometry.pointcloud]): [source cloud]
        target_cloud ([o3d.geometry.pointcloud]): [target cloud]
        transformation ([np.array(4x4)]): [transform between source and target cloud]
        path ([string]): [file path to save image]
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    source_temp = copy.deepcopy(source_cloud)
    target_temp = copy.deepcopy(target_cloud)
    source_temp.paint_uniform_color([1,0.706,0])
    target_temp.paint_uniform_color([0,0.651,0.929])
    target_temp.transform(transformation)
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.update_geometry(source_temp)
    vis.update_geometry(target_temp)
    vis.poll_events()
    vis.capture_screen_image(path, do_render=True)
    vis.destroy_window()


def parse_go_icp_result(path):
    """[parse the .txt output of goicp to a time and transform value]

    Args:
        path ([string]): [filepath of GO-ICP output.txt]
S
    Returns:
        [np.array(4x4), float]: [transfrom, error]
    """
    transform = np.zeros((4,4), dtype=np.float)
    file = open(path, 'r')
    lines = file.readlines()
    run_time = float(lines[0])
    transform[0,:3] = np.asarray([float (x) for x in lines[1].split()])
    transform[1,:3] = np.asarray([float (x) for x in lines[2].split()])
    transform[2,:3] = np.asarray([float (x) for x in lines[3].split()])
    transform[0,3] = float(lines[4])
    transform[1,3] = float(lines[5])
    transform[2,3] = float(lines[6])
    transform[3,3] = 1.0
    return transform,run_time

def load_tum_dataset(path):
    """Load TUM dataset image

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    depth_raw = o3d.io.read_image(path)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
    depth_raw,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


def load_testing_sequence(csv_path):
    csv = pd.read_csv(csv_path)
    return csv

def get_scaling_factor_go_icp(source_cloud, target_cloud):
    """Returns the scaling factor for GO ICP

    Args:
        source_cloud (open3d.geometry.poincloud): source cloud
        target_cloud (open3d.geometry.poincloud): target cloud

    Returns:
        _type_: _description_
    """
    source_np = np.asarray(source_cloud.points)
    target_np = np.asarray(target_cloud.points)
    source_np = source_np - np.mean(source_np, axis = 0)
    target_np = target_np - np.mean(target_np, axis = 0)
    combo = np.concatenate((target_np, source_np), axis = 0)
    return max(np.ptp(combo, axis = 0)) / 2.

# Transform matrix and scaling factor and scale the matrix 
def scale_matrix_tf(scaling_factor, matrix):
    to_return = copy.deepcopy(matrix)
    to_return[0,3] = to_return[0,3] *scaling_factor
    to_return[1,3] = to_return[1,3] *scaling_factor
    to_return[2,3] = to_return[2,3] *scaling_factor

    return to_return

    # scale_tf = np.zeros((4,4))
    # scale_tf[0,0] = scale_tf[1,1] = scale_tf[2,2] = scaling_factor
    # scale_tf[3,3] = 1
    # return np.matmul(scale_tf, matrix)


def render_camera_image(source_cloud, target_cloud, transformation, path, x_rot, y_rot, x_trans, y_trans, up, zoom, pt_size): 
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible = True)
    source_temp = copy.deepcopy(source_cloud)
    target_temp = copy.deepcopy(target_cloud)
    # calculate centers of pointclouds
    source_center = source_temp.get_center()
    target_center = target_temp.get_center() 
    overall_center = np.mean([source_center, target_center], axis = 0)
    # test sphere to visualize center point
    # sphere = o3d.geometry.TriangleMesh.create_sphere(.25,20)
    # sphere.translate(overall_center)
    # print("source center:", source_center, 'target center', target_center)
    # print("overall center:", overall_center)
    # sphere.paint_uniform_color([.4,.2,.6])
    source_temp.paint_uniform_color([1,0.706,0])
    target_temp.paint_uniform_color([0,0.651,0.929])
    target_temp.transform(transformation)
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    # vis.add_geometry(sphere)
    vis.update_geometry(source_temp)
    vis.update_geometry(target_temp)
    # vis.update_geometry(sphere)
    vis.poll_events()
    option = vis.get_render_option()
    option.point_size = pt_size
    ctr = vis.get_view_control()
    ctr.set_lookat(overall_center)
    ctr.rotate(x_rot, y_rot)
    # ctr.rotate(0,tilt, 0, 200)
    ctr.translate(x_trans, y_trans)
    ctr.set_zoom(zoom)
    ctr.set_up(up)
    vis.capture_screen_image(path, do_render=True)
    vis.destroy_window()




    

