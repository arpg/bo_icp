# BO-ICP

To get started:

```
git clone git@github.com:arpg/bo_icp.git
```

Install the package:

```
cd bo_icp
pip install .
```
Note it's recommended to do this in virtual environment such as conda.

Experiments from the papers were performed on the [KITTI](https://www.cvlibs.net/datasets/kitti/) and [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) Datasets.

For the KITTI dataset use `format: 'KITTI'` in the config file and for TUM use `format: 'tum'`. For all other file types the format tag gets passed into the `open3d.io.read_point_cloud()` function. The *auto* specifier will work for PCD files. For other file types please refer to the library [documentation](http://www.open3d.org/docs/release/) for more information.

The `BO_ICP_example.ipynb` shows an example of how to use this code. Point cloud locations can either be specified in the config file or by passing in a `source_cloud` and `target_cloud` path to the BO_ICP constructor (as done in the notebook).

Configs from the paper are provided as config_a, config_b, and config_c. It's recommended to start with b for a new dataset. If you need more accuracy try C and for a faster runtime try A. The generic `config.yaml` has comments describing each of the parameters.

We make use of the [BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization.git) library by Fernando Nogueira and [Open3d](https://github.com/isl-org/Open3D.git).

If you find this work useful please consider citing:

```
@INPROCEEDINGS{10160570,
  author={Biggie, Harel and Beathard, Andrew and Heckman, Christoffer},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={BO-ICP: Initialization of Iterative Closest Point Based on Bayesian Optimization}, 
  year={2023},
  volume={},
  number={},
  pages={6944-6950},
  doi={10.1109/ICRA48891.2023.10160570}}

```



