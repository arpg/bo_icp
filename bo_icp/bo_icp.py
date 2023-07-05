import copy
import math
import bayes_opt
import open3d as o3d
import numpy as np
import math
import scipy.spatial.distance
import time


from scipy import stats
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from bo_icp import bo_icp, utils


class BayesICP:
    """Bayes ICP Class
    """
    def __init__(self, config='config/config.yaml', source_cloud = None, target_cloud = None, format = None):
        """Constructor for Bayes ICP 

        Args:
            config (string, optional): [Yaml config file location]. Defaults config.yaml.
        """
        self.go_icp_ready = False
        self.optimal_transform = np.zeros((4,4))
        self.optimal_rotation = np.zeros((4,4))
        self.load_config(config)
        self.optimal_param = {}
        if source_cloud != None and target_cloud != None:
            self.load_point_clouds(source_cloud, target_cloud, format)
        else:
            self.load_point_clouds(utils.get_parameter(self.params, 'data', 'source_cloud_path'), 
                utils.get_parameter(self.params, 'data', 'target_cloud_path'),
                format=utils.get_parameter(self.params, 'data', 'format'))


    def load_config(self,config_path):
        self.params = utils.load_config(config_path)


    def load_point_clouds(self, source_path, target_path, format='auto'):
        """[Loads the target and source clouds for ICP]

        Args:
            source_path (string): [Path for the model or source point cloud (target will be aligned to this)]
            target_path (string): [Path for the target or data cloud (will be aligned to source)]
            format (str, optional): [Method to parse the point cloud file (see open3d.io for more)]. Defaults to 'auto'.
        """
        if format == 'tum':
            self.source_cloud = utils.load_tum_dataset(source_path)
            self.target_cloud = utils.load_tum_dataset(target_path)
        elif format == 'KITTI':
            self.source_cloud = utils.load_binary_cloud(source_path)
            self.target_cloud = utils.load_binary_cloud(target_path)
        else:
            self.source_cloud = o3d.io.read_point_cloud(source_path, format=format)
            self.target_cloud = o3d.io.read_point_cloud(target_path, format=format)
        
        self.source_cloud_viz = copy.deepcopy(self.source_cloud)
        self.target_cloud_viz = copy.deepcopy(self.target_cloud)

    def p2p_icp(self, initial_transform):
        """performs of point to point icp using an initial guess

        Args:
            initial_transform (np.array): [4x4 transformation matrix for icp initialization]
            iterations (int, optional): [Maximum number of iterations for icp]. Defaults to 50.
            relative_rmse ([type], optional): [rsme termination criteria. See open3d for more]. Defaults to 1e-06.
            relative_fitness ([type], optional): [fitness termination criteria see open3d for more]. Defaults to 1e-6.

        Returns:
            [open3d.geometry.Transformation]: [Transform obtained by running point to point icp]
        """
        threshold = utils.get_parameter(self.params, 'icp', 'threshold')
        iterations = utils.get_parameter(self.params, 'icp', 'max_iterations')
        relative_rmse = utils.get_parameter(self.params, 'icp', 'relative_rmse')
        relative_fitness = utils.get_parameter(self.params, 'icp', 'relative_fitness')

        p2p_icp_transform = o3d.pipelines.registration.registration_icp(
            self.source_cloud, copy.deepcopy(self.target_cloud), threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations,  relative_rmse=relative_rmse, relative_fitness=relative_fitness))
        return p2p_icp_transform

    
    def generate_transform(self,x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        """[summary]
        Generates a 3D transform from x,z,y,roll,pitch, and yaw values.

        Args:
            x (float, optional): [translation in x]. Defaults to 0.0.
            y (float, optional): [translation in y]. Defaults to 0.0.
            z (float, optional): [translation in z]. Defaults to 0.0.
            roll (float, optional): [Roll]. Defaults to 0.0.
            pitch (float, optional): []. Defaults to 0.0.
            yaw (float, optional): [description]. Defaults to 0.0.

        Returns:
            [np.array(4x4)]: [Transformation matrix]
        """



        transform = np.zeros((4,4))
        transform[3,3] = 1
        rotation_matrix = o3d.geometry.Geometry3D.get_rotation_matrix_from_xyz([roll, pitch, yaw])
        transform[:3,:3] = rotation_matrix
        transform[0,3] = x
        transform[1,3] = y
        transform[2,3] = z
        transform[np.isnan(transform)] = 0 
        return transform


    def evaluate_icp(self, transform, metric='distance'):
        cloned_target = copy.copy(self.target_cloud)
        transformed_target_cloud = o3d.geometry.PointCloud.transform(cloned_target, transform)
        if metric == 'final_evaluation':
            final_evaluation = o3d.pipelines.registration.evaluate_registration(self.source_cloud, 
            self.target_cloud, self.threshold, transform)
            return -1*final_evaluation.fitness
        elif metric == 'hausdorff':
            hausdorff_distance = scipy.spatial.distance.directed_hausdorff(self.source_cloud.points,
            transformed_target_cloud.points)
            return np.mean(hausdorff_distance)
        elif metric == 'distance':
            point_to_point_distance = self.source_cloud.compute_point_cloud_distance(transformed_target_cloud)
            return  np.mean(point_to_point_distance)
        elif metric == 'trimmed_mean':
            point_to_point_distance = self.source_cloud.compute_point_cloud_distance(transformed_target_cloud)
            return stats.trim_mean(point_to_point_distance, 0.25)
        elif metric == 'inlier_rmse':
            final_evaluation = o3d.pipelines.registration.evaluate_registration(self.source_cloud, 
            self.target_cloud, self.threshold, transform)
            inlier_rmse = final_evaluation.inlier_rmse
            if math.isclose(inlier_rmse, 0.0):
                inlier_rmse = 10000 
            return inlier_rmse


    def bayes_opt_to_transform(self, params):
        x = params.get('x')
        y = params.get('y')
        z = params.get('z')
        roll = params.get('roll')
        pitch = params.get('pitch')
        yaw = params.get('yaw')
        final_transform = self.generate_transform(x,y,z,roll,pitch,yaw)
        return final_transform

    def get_optimal_rotation_values(self):
        roll = 0
        pitch = 0
        yaw = 0
        if 'roll' in self.optimal_param:
            roll = self.optimal_param.get('roll')
        if 'pitch' in self.optimal_param:
            pitch = self.optimal_param.get('pitch')
        if 'yaw' in self.optimal_param:
            yaw = self.optimal_param.get('yaw')
        return roll, pitch, yaw


    def process_point_cloud(self):
        if utils.get_parameter(self.params, 'data', 'voxel_downsample'):
            voxel_size = utils.get_parameter(self.params, 'data', 'voxel_size')
            self.source_cloud = self.source_cloud.voxel_down_sample(voxel_size)
            self.target_cloud = self.target_cloud.voxel_down_sample(voxel_size)


    def bayes_opt_icp_func(self,x=0,y=0,z=0,roll=0,pitch=0,yaw=0):
        if roll == 0 and pitch == 0 and yaw == 0:
            roll, pitch, yaw = self.get_optimal_rotation_values()
        initial_transform = self.generate_transform(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)
        icp_transform = self.p2p_icp(initial_transform)
        metric = utils.get_parameter(self.params, 'bayes_opt', 'error_metric')
        fitness = self.evaluate_icp(icp_transform.transformation, metric=metric)
        # For maximization (default for byes opt library)
        return -1.0*fitness

    def bayes_opt_icp_init(self, type='rotation'):
        parameter_bounds = utils.get_parameter(self.params, 'bayes_opt', 'parameter_bounds')
        if type == 'all':
            pass
        elif type == 'rotation':
            parameter_bounds = utils.get_roatation_params(parameter_bounds)
        elif type == 'translation':
            parameter_bounds = utils.get_translation_params(parameter_bounds)

        func = self.bayes_opt_icp_func

        self.process_point_cloud()
        
        print(parameter_bounds)
        self.optimizer = BayesianOptimization(
            f = func,
            pbounds = parameter_bounds,
            random_state=1
            )

    def bayes_opt_icp(self):
        # Get Parameters
        init_count = utils.get_parameter(self.params, 'bayes_opt', 'init_count')
        num_iter = utils.get_parameter(self.params, 'bayes_opt', 'num_iter')
        alpha = utils.get_parameter(self.params, 'bayes_opt', 'alpha')
        acq = utils.get_parameter(self.params, 'bayes_opt', 'acq')
        kappa = utils.get_parameter(self.params, 'bayes_opt', 'kappa')
        xi = utils.get_parameter(self.params, 'bayes_opt', 'xi')
        
        self.optimizer.maximize(
        init_points = init_count,
        n_iter = num_iter,
        alpha=alpha,
        acq=acq,
        xi=xi,
        kappa=kappa,
        )
        self.optimal_param.update(self.optimizer.max.get('params'))
        self.optimal_transform_no_icp = self.bayes_opt_to_transform(self.optimal_param)
        self.optimal_transform = self.p2p_icp(self.bayes_opt_to_transform(self.optimal_param)).transformation

    def run_bayes_opt_icp(self):
        type = utils.get_parameter(self.params, 'bayes_opt', 'type')
        tic = time.time()
        if type == 'separate':
            self.bayes_opt_icp_init(type="rotation")
            self.bayes_opt_icp()
            self.bayes_opt_icp_init(type="translation")
            self.bayes_opt_icp()
        else:
            self.bayes_opt_icp_init(type=type)
            self.bayes_opt_icp()
        toc = time.time()
        self.run_time = toc - tic

    

    




