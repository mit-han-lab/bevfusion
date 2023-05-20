import argparse

from data_converter import nuscenes_converter as nuscenes_converter
from data_converter.create_gt_database import create_groundtruth_database


def nuscenes_data_prep(
    root_path,
    info_prefix,
    version,
    dataset_name,
    out_dir,
    capture_likelihood=False,
    cluster_info_path=None,
    max_sweeps=10,
    load_augmented=None,
):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        capture_likelihood (bool, optional): Whether to capture sample likelihood from nuScenes. Defaults to False.
        cluster_info_path (str, optional): Path of cluster info file.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    # if load_augmented is None:
    #     print("start nuscenes_converter.create_nuscenes_infos")
    #     nuscenes_converter.create_nuscenes_infos(
    #             root_path=root_path, 
    #             info_prefix=info_prefix, 
    #             version=version, 
    #             max_sweeps=max_sweeps, 
    #             capture_likelihood=capture_likelihood, 
    #             cluster_info_path=cluster_info_path,
    #         )
        # if version == "v1.0-test":
        #     info_test_path = osp.join(root_path, f"{info_prefix}_infos_test.pkl")
        #     nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)
        #     return

        # info_train_path = osp.join(root_path, f"{info_prefix}_infos_train.pkl")
        # info_val_path = osp.join(root_path, f"{info_prefix}_infos_val.pkl")
        # nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
        # nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)
    
    # nuscenes_converter.create_nuscenes_infos(
    #             root_path=root_path, 
    #             info_prefix=info_prefix, 
    #             version=version, 
    #             max_sweeps=max_sweeps, 
    #             out_dir=out_dir,
    #             capture_likelihood=capture_likelihood, 
    #             cluster_info_path=cluster_info_path,
    #         )

    # create_groundtruth_database(
    #     dataset_name=dataset_name,
    #     data_path=out_dir,
    #     info_prefix=info_prefix,
    #     info_path=f"{out_dir}/{info_prefix}_infos_train.pkl",
    #     load_augmented=load_augmented,
    #     #capture_likelihood=capture_likelihood,
    #     #cluster_info_path=cluster_info_path, 
    # )
    create_groundtruth_database(
        dataset_name,
        out_dir, #root_path
        info_prefix,
        f"{out_dir}/{info_prefix}_infos_train.pkl",
        load_augmented=load_augmented,
    )


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="kitti", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    default="./data/kitti",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--info-path",
    type=str,
    default="./data/kitti_info",
    help="specify the info path of dataset",
)
parser.add_argument(
    "--version",
    type=str,
    default="v1.0",
    required=False,
    help="specify the dataset version, no need for kitti",
)
parser.add_argument(
    '--capture-likelihood',
    default=False,
    action='store_true',
    help='specify whether to use capture sample likelihood for the dataset')
parser.add_argument(
    '--cluster-info-path',
    type=str,
    default='./data/nuscenes/nuscenes_training_cluster_info.pkl',
    help='specify the path of cluster info file')
parser.add_argument(
    "--max-sweeps",
    type=int,
    default=10,
    required=False,
    help="specify sweeps of lidar per example",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="./data/kitti",
    required=False,
    help="name of info pkl",
)
# parser.add_argument(
#     "--database-save-path",
#     type=str,
#     default="./data/kitti",
#     required=False,
#     help="name of info pkl",
# )
parser.add_argument("--extra-tag", type=str, default="kitti")
parser.add_argument("--painted", default=False, action="store_true")
parser.add_argument("--virtual", default=False, action="store_true")
parser.add_argument(
    "--workers", type=int, default=4, help="number of threads to be used"
)
args = parser.parse_args()

if __name__ == "__main__":
    load_augmented = None
    if args.virtual:
        if args.painted:
            load_augmented = "mvp"
        else:
            load_augmented = "pointpainting"

    if args.dataset == "nuscenes" and args.version != "v1.0-mini":
        train_version = f"{args.version}-trainval"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            capture_likelihood=args.capture_likelihood,
            cluster_info_path=args.cluster_info_path,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
        test_version = f"{args.version}-test"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            capture_likelihood=args.capture_likelihood,
            cluster_info_path=args.cluster_info_path,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
    elif args.dataset == "nuscenes" and args.version == "v1.0-mini":
        train_version = f"{args.version}"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            capture_likelihood=args.capture_likelihood,
            cluster_info_path=args.cluster_info_path,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )