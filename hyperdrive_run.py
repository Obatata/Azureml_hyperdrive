from azureml.core import Workspace, Experiment, Environment
from azureml.core.environment import CondaDependencies
from azureml.core.compute import AmlCompute
# connect to workspace
ws = Workspace.from_config("./config")

# get the dataset from workspace
input_ds = ws.datasets.get("Dafaults")

# create environment
my_env =Environment(name="my_env")
#install packages
conda_dependencies = CondaDependencies()
conda_dependencies.add_pip_package("azureml-sdk")
conda_dependencies.add_pip_package("sklearn")
#add package to the my_env
my_env.python.conda_dependencies  = conda_dependencies
# set user_manager_dependences to False
my_env.python.user_managed_dependencies = False
# regiter my_env on workspace (ws)
my_env.register(ws)

# create compute cluster
cluster_name = "Cluster-auto-ml"
if cluster_name not in ws.compute_targets:
    compute_config = AmlCompute.provisioning_configuration(
                                vm_size="STANDARD_D11_V2",
                                max_nodes=2
                                )
    cluster = AmlCompute.create(ws, cluster_name, compute_config)
    cluster.wait_for_completion()
else:
    print("target compute was created before")
    cluster = ws.compute_targets[cluster_name]
