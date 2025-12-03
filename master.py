import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"


import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
if not (jnp.array(1.).dtype is jnp.dtype("float64")):
    raise RuntimeError("jax.config.jax_enable_x64 must be set for numerical stability. "
                      "Please set this environment variable before using NQS.")
import a_scale.archiving_utils
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import a_scale
from a_scale import nn 


@hydra.main(version_base=None, 
            config_path="conf", 
            config_name="master")
def main(cfg: DictConfig):

    

    # ---- Set up logging ---- #
    logger = logging.getLogger(__name__)
    logger.info("starting the pipeline")

    # ---- Create the job directory and save the config ---- #
    job_dir_str = cfg.job_dir
    logger.info(f"job directory: {job_dir_str}")
    if not os.path.exists(cfg.job_dir):
        os.makedirs(cfg.job_dir)
    OmegaConf.save(cfg, cfg.job_dir + "/config.yaml")

    # save this file in the job directory
    with open(cfg.job_dir + "/master.py", "w") as f:
        f.write("# Path: scale/master.py\n")
        with open(__file__, "r") as master_file:
            f.write(master_file.read())

    
    # --------------- Create Archives --------------- #
    a_scale.archiving_utils.create_archive(cfg.nn_archive_file)
    a_scale.archiving_utils.create_archive(cfg.nqs_archive_file)

    fit_nqs = cfg.fit_nqs
    
    if fit_nqs:  
    # --------------- Fit NQS Model --------------- #
        # instantiate the model
        logger.info("fitting the NQS model")
        scaling_model = hydra.utils.instantiate(cfg.a_scaling_model)
        
        fitted_nqs, train_eval_metric = scaling_model() 
        logger.info("fitted NQS model: \n" + OmegaConf.to_yaml(fitted_nqs))
        logger.info("train eval metric: " + str(train_eval_metric))
    else:
        logger.info("loading the NQS model...")
        fitted_nqs = OmegaConf.load(cfg.fitted_nqs)
        logger.info("loaded NQS model: \n" + OmegaConf.to_yaml(fitted_nqs))

    # --------------- Fit Chinchilla Models --------------- #
    fit_chinchilla_2 = cfg.fit_chin2
    if fit_chinchilla_2: 
        logger.info("fitting the chinchilla models...")
        chinchilla_2 = hydra.utils.instantiate(cfg.chinchilla_2)
        fitted_chin2, train_eval_metric_chin2 = chinchilla_2()
        logger.info("fitted chinchilla 2 model: \n" + OmegaConf.to_yaml(fitted_chin2))
        logger.info("train eval metric chinchilla 2: " + str(train_eval_metric_chin2))
    else:
        logger.info("loading chinchilla 2 model...")
        fitted_chin2 = OmegaConf.load(cfg.fitted_chin2)
        logger.info("loaded chinchilla 2 model: \n" + OmegaConf.to_yaml(fitted_chin2))


    # --------------- Run Loss Estimation --------------- #
    # instantiate the procedure
    run_loss_estimation = cfg.run_loss_estimation
    if run_loss_estimation:
        logger.info("running the loss estimation procedure")
        loss_estimation = hydra.utils.instantiate(cfg.loss_estimation)
        eval_metrics_df = loss_estimation(fitted_nqs = fitted_nqs, 
                                            fitted_baselines =  {"chinchilla_2": fitted_chin2}) #"chinchilla_1": fitted_chin1 
                                                               # })
                                                                #,
                                                              #  "chinchilla_2": fitted_chin2})
        logger.info("eval metrics: \n" + str(eval_metrics_df))
        # log the effective model size factor
        #logger.info("effective model size factors tested: ")
        #logger.info(str(cfg.effective_model_size_factor_multiplier), str(cfg.effective_model_size_factor_power))
    else:
        logger.info("skipping the loss estimation procedure!")


    # --------------- Resource Allocation --------------- #
    run_resource_allocation = cfg.run_resource_allocation #True
    if run_resource_allocation:
        logger.info("running the resource allocation procedure")
        resource_allocation = hydra.utils.instantiate(cfg.resource_allocation)
        resource_allocation(fitted_nqs = fitted_nqs, 
                                fitted_baselines = {"chinchilla_2": fitted_chin2})#,
                                                    #"chinchilla_2": fitted_chin2})
            

        logger.info("resource allocation complete")



    # --------------- Critical Batch Size --------------- #

    run_critical_batch_size = cfg.run_critical_batch_size #True #True #True
    if run_critical_batch_size:
        logger.info("running the critical batch size procedure")
        critical_batch_size = hydra.utils.instantiate(cfg.critical_batch_size)
        critical_batch_size(fitted_nqs = fitted_nqs)
        logger.info("critical batch size complete")




    # -------------- Var Analysis --------------- #
    run_variance_analysis = cfg.run_variance_analysis #True
    if run_resource_allocation and run_critical_batch_size and run_variance_analysis:
        # run the script var_analysis_in_pipe.py with the command line argument of the job directory
        logger.info("running the variance analysis procedure")
        import subprocess
        # run var_analysis_in_pipe.py with the job directory as argument
        subprocess.run(["python", "var_analysis_in_pipe.py", cfg.job_dir])
        logger.info("variance analysis complete")
    elif run_variance_analysis:
        logger.info("skipping the variance analysis procedure because resource allocation or critical batch size was not run.")




if __name__ == "__main__":
    main()