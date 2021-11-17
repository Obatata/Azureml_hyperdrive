from azureml.core import Workspace, Experiment, Environment
from azureml.core.environment import CondaDependencies
from azureml.core.compute import AmlCompute
from azureml.core import ScriptRunConfig, Experiment
from azureml.train.hyperdrive import GridParameterSampling, choice, HyperDriveConfig, PrimaryMetricGoal



# connect to workspace
ws = Workspace.from_config("./config")

# get the dataset from workspace
input_ds = ws.datasets.get('Defaults')

# create environment
my_env =Environment(name="my_env")
#install packages
conda_dependencies = CondaDependencies()
conda_dependencies.add_pip_package("azureml-sdk")
conda_dependencies.add_pip_package("pandas")
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

# create a script configuration for custom environment of my_env
script_config = ScriptRunConfig(
                                script="hyperdrive_script_to_submit.py",
                                arguments=["--input_data", input_ds.as_named_input('raw_data')],
                                environment=my_env,
                                source_directory=".",
                                compute_target=cluster
                               )

# create hyper drive prameters
hyper_params = GridParameterSampling(
                                     {"--n_estimators":choice(10, 20, 50, 100),
                                      "--min_samples_leaf":choice(1, 2, 5)
                                     }
                                    )

# Configure the Hyperdrive class
hyper_config =  HyperDriveConfig(
                                 run_config=script_config,
                                 hyperparameter_sampling=hyper_params,
                                 policy=None,
                                 primary_metric_name="accuracy",
                                 primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                 max_total_runs=20,
                                 max_concurrent_runs=2,
                                )

# create new experiment
new_experiment = Experiment(
                            workspace=ws,
                            name="Hyperdrive_exp_01",
                           )
new_run = new_experiment.submit(config=hyper_config)
new_run.wait_for_completion(show_output=True)

