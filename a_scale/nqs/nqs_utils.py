# fit NQS: h + neural_net
# look up/update NN archive
from jax import config
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import zeta
import hydra
import pandas as pd
from jax.scipy.special import gamma, gammainc

from a_scale.nqs.nqs import risk
from a_scale.nqs.nqs import NQS, SGD, fit_multiple, to_x, to_nqs
from a_scale.nqs.nqs import StepLR, SequentialLR, MultiStepLR, ConstantLR

import json

from pyDOE import lhs  # For Latin Hypercube Sampling
import numpy as np

import copy 




def latin_hypercube_initializations(seed, num_inits, param_ranges):
        """Create initialization points using Latin Hypercube Sampling."""
        # Parameter names in order
        param_names = ['a', 'b', 'ma', 'mb', 'c', 'sigma']
        dims_param = len(param_names)

        # convert the param_ranges onto the normalized space, using to_x
        # to_x is a function that takes in an NQS object and converts it to dim 6 array
        lower_bound_NQS = NQS(a=param_ranges['a'][0],
                              b=param_ranges['b'][0],
                              ma=param_ranges['ma'][0],
                              mb=param_ranges['mb'][0],
                              c=param_ranges['c'][0],
                              sigma=param_ranges['sigma'][0])
        upper_bound_NQS = NQS(a=param_ranges['a'][1],
                                b=param_ranges['b'][1],
                                ma=param_ranges['ma'][1],
                                mb=param_ranges['mb'][1],
                                c=param_ranges['c'][1],
                                sigma=param_ranges['sigma'][1])
        # Convert to normalized space   
        lower_bound = to_x(lower_bound_NQS)
        upper_bound = to_x(upper_bound_NQS)

        # convert back to lower and upper bound params dict
        param_ranges_normalized = {'a': (lower_bound[0], upper_bound[0]),
                                   'b': (lower_bound[1], upper_bound[1]),
                                   'ma': (lower_bound[2], upper_bound[2]),
                                   'mb': (lower_bound[3], upper_bound[3]),
                                   'c': (lower_bound[4], upper_bound[4]),
                                   'sigma': (lower_bound[5], upper_bound[5])}
        
        # Generate Latin Hypercube samples in [0, 1] range
        # set seed
        np.random.seed(seed)
        samples = lhs(len(param_names), samples=num_inits)
        
        # Scale samples to the parameter ranges
        init_nqs_list = []
        for i in range(num_inits):
            x_normalized = jnp.array([0.0] * len(param_names))
            
            for j, param_name in enumerate(param_names):
                # Get parameter range
                lower, upper = param_ranges_normalized[param_name]
                # Scale from [0, 1] to [lower, upper]
                #x_normalized[j] = lower + samples[i, j] * (upper - lower)
                x_normalized = x_normalized.at[j].set(lower + samples[i, j] * (upper - lower))
            
            # Convert to NQS space
            init_nqs = to_nqs(x_normalized)
            init_nqs_list.append(init_nqs)
            
        return init_nqs_list

def create_scheduler(step_decay_schedule, K):
        
        
    # use json to convert the string to a dictionary
    #step_decay_schedule = json.loads(step_decay_schedule_str)
    decay_at = step_decay_schedule["decay_at"]
    decay_amt = step_decay_schedule["decay_amt"]

    # check if the first element of decay_at is <= 1.0
    #raise ValueError("decay_at[0] is", decay_at[0], "K is", K)
    if decay_at[0] > 1.0: # decay_at should be intepreted as steps
        milestones = [decay_at[0]]
        for i in range(1, len(decay_at)):
            milestones.append(milestones[-1] + decay_at[i])
        
        #raise ValueError("milestones are", milestones)
        # Create schedulers
        schedulers = [ConstantLR(1.0, milestones[0])]
        for i in range(1, len(decay_at)):
            schedulers.append(ConstantLR(1.0 * decay_amt[i-1], 
                                        milestones[i] - milestones[i-1]))
        schedulers.append(ConstantLR(1.0 * decay_amt[-1],
                                    K - milestones[-1]))
        
        # if the last milestone is less than K, keep adding t the milestones and schedulers, 
        # by repeat the last decay_at and decay_amt
        while milestones[-1] < K:
            milestones.append(milestones[-1] + decay_at[-1])
            schedulers.append(ConstantLR(1.0 * decay_amt[-1],
                                        milestones[-1] - milestones[-2]))

    else: # decay_at should be interpreted as fractions of K
            # decay at is a list of the iteration at which the learning rate is decayed
            #milestones = [int(h.decay_at_1 * K), 
            #              int(h.decay_at_2 * K)]
            # write the above defn of milestones in a more general way, loop over decay_at
            milestones = [int(decay_at[0] * K)]
            for i in range(1, len(decay_at)):
                milestones.append(int(decay_at[i] * K))
            # decay_amt is the amount by which the learning rate is decayed
            #schedulers = [ConstantLR(init_lr, milestones[0]),
            #            ConstantLR(init_lr * h.decay_amt_1, 
            #                        milestones[1] - milestones[0]), 
            #            ConstantLR(init_lr * h.decay_amt_1 * h.decay_amt_2, 
            #                        K - milestones[1])]
            schedulers = [ConstantLR(1.0, milestones[0])]
            for i in range(1, len(decay_at)):
                schedulers.append(ConstantLR(1.0 * decay_amt[i-1], 
                                            milestones[i] - milestones[i-1]))
            schedulers.append(ConstantLR(1.0 * decay_amt[-1],
                                        K - milestones[-1]))
            
    scheduler = SequentialLR(milestones=milestones, schedulers=schedulers)

    return scheduler


# ------------------ Fitting
from a_scale.nqs.nqs_sgd import fit_nqs

def fit_nqs_old(h_dicts, nn_losses, seed, number_of_initializations, param_ranges_raw,
            gtol, max_steps):


    nqs_init_list = latin_hypercube_initializations(seed, number_of_initializations, param_ranges_raw)
    # h_dicts is a list of dictionaries, each dictionary is a hyperparameter dictionary
    # ls is a list of losses, each loss is a number

    # convert the list of dictionaries into a list of (N,K,B, folds) tuples, by
    # traverse the list of dictionaries
    NKBfoldsarrs = []
    for h_dict in h_dicts:
        N = h_dict["N"]
        K = h_dict["K"]
        B = h_dict["B"]
        #raise ValueError("step_decay_schedule: ", h_dict["step_decay_schedule"])
        if h_dict["lr_schedule"] == "step":
            step_decay_schedule = h_dict["step_decay_schedule"] #json.loads(h_dict["step_decay_schedule"])
            scheduler = create_scheduler(step_decay_schedule, K)
        elif h_dict["lr_schedule"] in ["constant","cosine"]: # Treat cosine as constant in fitting
            scheduler = ConstantLR(1.0, K)
        else:
            raise ValueError("for fast nqs, lr_schedule must be one of 'step' or 'constant/cosine'")
        folds = SGD(lr=h_dict["lr"], momentum=h_dict["momentum"], nesterov=False,
                    scheduler=scheduler)
        NKB = (N, K, B)
        NKBfoldsarrs.append(NKB + (folds,))

    # fit takes a list of (NKB, folds) tuples
    fitted_nqs_formatted, trajectories_formatted = fit_multiple(
    #fitted_nqs_formatted, best_loss 
       
        NKBfoldsarrs, lossarr = nn_losses, nqs0_list = nqs_init_list, return_traj = True,
        steps = max_steps, gtol = gtol)
    nqs_fitted = fitted_nqs_formatted 

    # trajectories_formatted is a list of dictionaries,
    # where each dictionary has a key "loss".
    # find the minimum loss in the list of dictionaries, excl. nans
    # if trajectories_formatted is None, set it to an empty list
    lowest_loss = float("inf")
    #if trajectories_formatted is None:
    # if the variable does not exist, set it to an empty list
    if "trajectories_formatted" not in locals():
        raise ValueError("trajectories_formatted is not defined, please check the fit_multiple function.")
       # trajectories_formatted = []
       # lowest_loss = float(best_loss)
    else:
        for i in range(len(trajectories_formatted)):
            loss_at_i = trajectories_formatted[i]["loss"]
            # if not nan, and less than lowest_loss, update
            if not jnp.isnan(loss_at_i) and loss_at_i < lowest_loss:
                lowest_loss = loss_at_i

    fit_metric_value = lowest_loss

    #nqs_fitted, res = fit(NKBfoldsarrs, lossarr = nn_losses, nqs0 = nqs0, return_res = True)
    #fit_metric_value = res.fun
    # turn fit_metric_value into a jnp scalar
    fit_metric_value = jnp.array(fit_metric_value)

    # convert the fitted nqs into a dictionary
    fitted_nqs_dict = {'a': nqs_fitted.a, 
                       'b': nqs_fitted.b, 
                       'm_a': nqs_fitted.ma, 
                       'm_b': nqs_fitted.mb, 
                       'eps': nqs_fitted.c, 
                       'sigma': nqs_fitted.sigma}
    # convert the fitted_nqs_dict values into floats
    fitted_nqs_dict = {k: float(v) for k, v in fitted_nqs_dict.items()}
    

    eval_metric_value = jnp.sqrt(2 * fit_metric_value) * 100

    return fitted_nqs_dict, eval_metric_value, trajectories_formatted


def fit_nqs_apr_old(h_dicts, nn_losses, nqs_init):
    # h_dicts is a list of dictionaries, each dictionary is a hyperparameter dictionary
    # ls is a list of losses, each loss is a number

    # convert the nqs_init dictionary into an NQS object
    nqs0 = NQS(a=nqs_init["a"], 
        b=nqs_init["b"], 
        ma=nqs_init["m_a"], 
        mb=nqs_init["m_b"], 
        c=nqs_init["eps"], 
        sigma=nqs_init["sigma"])
    # convert the list of dictionaries into a list of (N,K,B, folds) tuples, by
    # traverse the list of dictionaries
    NKBfoldsarrs = []
    for h_dict in h_dicts:
        N = h_dict["N"]
        K = h_dict["K"]
        B = h_dict["B"]
        #raise ValueError("step_decay_schedule: ", h_dict["step_decay_schedule"])
        if h_dict["lr_schedule"] == "step":
            step_decay_schedule = json.loads(h_dict["step_decay_schedule"])
            scheduler = create_scheduler(step_decay_schedule, K)
        elif h_dict["lr_schedule"] == "constant":
            scheduler = ConstantLR(1.0, K)
        else:
            raise ValueError("for fast nqs, lr_schedule must be one of 'step' or 'constant'")
        folds = SGD(lr=h_dict["lr"], momentum=h_dict["momentum"], nesterov=False,
                    scheduler=scheduler)
        NKB = (N, K, B)
        NKBfoldsarrs.append(NKB + (folds,))

    # fit takes a list of (NKB, folds) tuples
    fit_output_dict = fit(NKBfoldsarrs, lossarr = nn_losses, nqs0 = nqs0, 
                          return_res= True)
    nqs_fitted = fit_output_dict["nqs"]
    fit_metric_value = fit_output_dict["fit_metric_value"]
    #nqs_fitted, res = fit(NKBfoldsarrs, lossarr = nn_losses, nqs0 = nqs0, return_res = True)
    #fit_metric_value = res.fun
    # turn fit_metric_value into a jnp scalar
    fit_metric_value = jnp.array(fit_metric_value)

    # convert the fitted nqs into a dictionary
    fitted_nqs_dict = {'a': nqs_fitted.a, 
                       'b': nqs_fitted.b, 
                       'm_a': nqs_fitted.ma, 
                       'm_b': nqs_fitted.mb, 
                       'eps': nqs_fitted.c, 
                       'sigma': nqs_fitted.sigma}
    # convert the fitted_nqs_dict values into floats
    fitted_nqs_dict = {k: float(v) for k, v in fitted_nqs_dict.items()}
    

    eval_metric_value = jnp.sqrt(2 * fit_metric_value) * 100

    return fitted_nqs_dict, eval_metric_value


# -------------- Fast Risk Approximation ----

from a_scale.nqs.nqs_sgd import EM_nqs_from_cfg_six_optimized, EM_nqs_from_cfg_six_standard
def EM_nqs_from_cfg_six(nqs_cfg, working_file_path = None, LRA_tol= 0.001):
    #raise ValueError("The LRA_tol is set to:", LRA_tol)
    if nqs_cfg.h.lr_schedule in ["optimized"]:
        #raise ValueError("doing optimized LRA cosine, cfg is:", nqs_cfg)
        return EM_nqs_from_cfg_six_optimized(nqs_cfg, working_file_path, LRA_tol=LRA_tol)
    elif nqs_cfg.h.lr_schedule in ["step", "constant"]:
        return EM_nqs_from_cfg_six_standard(nqs_cfg, working_file_path)
    else:
        raise ValueError("Invalid NQS configuration lr_schedule")



def EM_nqs_from_cfg_six_standard_old(nqs_cfg, working_file_path = None):

    a = nqs_cfg.a
    b = nqs_cfg.b 
    sigma = nqs_cfg.sigma
   # sigma = nqs_cfg.sigma => to be calculated later
    m_b = nqs_cfg.m_b
    m_a = nqs_cfg.m_a #m_b #1 #nqs_cfg.m_b => to be calculated later
    c = nqs_cfg.eps
   # eps = nqs_cfg.eps

    h = nqs_cfg.h
    init_lr = h.lr 
    beta = h.momentum
    B = h.B
    K = h.K
    N = h.N
    step_decay_schedule = h.step_decay_schedule
    
    if h.lr_schedule == "step":
        scheduler = create_scheduler(step_decay_schedule, K)
        # raise an error that prints the schedule
        #raise ValueError("scheduler: ", scheduler)
    elif h.lr_schedule in ["constant"]:
        scheduler = ConstantLR(1.0, K)
    else:
        raise ValueError("for fast nqs, lr_schedule must be one of 'step' or 'constant'")

    #learning_rate_schedule = hydra.utils.instantiate(h.optimizer.learning_rate)
    #def lr_schedule(t,K):
    #    return learning_rate_schedule(t)/learning_rate_schedule(0)

   # bayes_risk, biases, variances = nqs(a, b, sigma, init_lr, lr_schedule, beta, B, K, N,
   #                                     m_a = m_a, m_b = m_b)
    
    #iters_to_calc = [64, 1000, 8000, 64000, 512000, 4096000, 32768000]
    # keep only the iterations that are less than K
    #iters_to_calc = [k for k in iters_to_calc if k <= K]
    iters_to_calc = [K]
    # add K
    #iters_to_calc.append(K)

    # temporary use x64 (for jax and np)
    #with config.update("jax_enable_x64", True): 
    if True:
       # bnbs = []
       # vars = []
        nqs_risks = []
        optimizer = SGD(lr=init_lr, momentum=beta, nesterov=False, 
                            scheduler=scheduler)
        for k in iters_to_calc:
            # raise ValueError("calc risk at iter k: ", k)  
            # compute bias & approx error 
            nqs_params_bnb = NQS(a=a, b=b, ma=m_a, mb=m_b, c=c, sigma=sigma)
            nqs_risk = risk(N, k, B, nqs_params_bnb, optimizer, kind="fast")
            # compute variance
            #nqs_params_tot = NQS(a=a, b=b, m_a=m_a, m_b=m_b, c=0.0, sigma=sigma)
           # tot = risk(N, k, B, nqs_params_tot, optimizer, kind="fast")
            #variance = tot - bayes_and_bias
    
            #bnbs.append(bayes_and_bias)
           #vars.append(variance)
            nqs_risks.append(nqs_risk)

    # convert the list of biases and variances into jnp arrays
    #bnbs = jnp.array(bnbs)
    #vars = jnp.array(vars)
    nqs_risks = jnp.array(nqs_risks)
    #print(a, b, sigma, init_lr, lr_schedule, beta, B, K, N, ma, mb)
    # convert into a dataframe, with index iteration (starting from 0)
    nqs_df = pd.DataFrame({"nqs_iter": iters_to_calc, 
                           "nqs_risk": nqs_risks})
    
    #nqs_df["bayes_risk"] = jnp.zeros(len(bnbs)) # bayes risk is rolled into bias

    return {"nqs_df": nqs_df}#, "lros_df": lros_df}





def compute_risk_trajectory_opt_lr(B, K_target, learning_rate, 
                           N, nqs_params=None,
                           stair_width = 1000, verbose = False,
                           LRA_tol = 0.001):
    """
    Compute risk trajectory for a given configuration.
    
    Args:
        B: Batch size
        K_target: Target number of steps
        scheduler_config: 'with_schedule' or 'no_schedule'
        learning_rate: Learning rate to use
        N: Number of data points
        n_trajectory_points: Number of points in trajectory
        nqs_params: NQS parameters
    
    Returns:
        tuple: (K_values, risks) for the trajectory
    """

    # if k_target is large, also increase stair_width
    # stair_width = 1000 * (K_target // 150000)
    #if K_target > 150000:
     #   stair_width = stair_width * (K_target // 150000 + 1)

    print("running compute_risk_trajectory_opt_lr with K_target:", K_target, " N:", N, "B:", B, "learning_rate:", learning_rate)

    if K_target > 10:
        stair_width = max(K_target //100, min(1000, K_target // 10))
    else:
        stair_width = 1

    print("stair_width is: ", stair_width)

    if nqs_params is None:
        raise ValueError("nqs_params must be provided")

    #raise ValueError("N:", N)
    
    # Create NQS object
    nqs0 = NQS(**nqs_params)

    original_schedule = {
        'decay_at': [], #*100,
        'decay_amt': [] #*100 #if scheduler_config == 'with_schedule' else [1.0]*100
    }

    K_values = []
    risks = []
    # using K_target, compute how many 1000 steps are required
    n_stages = K_target // stair_width + 1
    last_stage_steps = K_target % stair_width


    start_lr = learning_rate
    completed_schedule = original_schedule
    # greedy algorithm to find the best learning rate for each stage
    steps_in_prev_stage = 0
    for stage in range(n_stages):
        old_risk_at_attempt = float('inf')
        # use current_base_lr as the learning rate for this stage
        attempt_schedule = completed_schedule.copy()
        steps_in_stage = stair_width if stage < n_stages - 1 else last_stage_steps
        if steps_in_stage <= 0:
            if verbose:
                print(f"Skipping stage {stage+1}/{n_stages} due to no steps in stage")
            continue
        if steps_in_prev_stage == 0:
            # use constant learning rate for the first stage
            sgd = SGD(momentum=0.0, nesterov=False, lr=start_lr, 
                         #weight_decay=0.0, 
                         scheduler=ConstantLR(1.0, steps_in_stage))
            # compute risk for this stage
            K_current = steps_in_stage
            risk_best = risk(N=N, K=K_current, B=B, nqs=nqs0, folds=sgd)

            #raise ValueError("N: ", N, "K_current:", K_current, "B:", B, "risk_best:", risk_best,
            #                 "nqs:", nqs0, "folds:", sgd)

            schedule_best = {
                'decay_at': [steps_in_stage],
                'decay_amt': [1.0] # keep the same learning rate
            }
            steps_in_prev_stage = steps_in_stage
            prev_decay_amt = 1.0

        elif steps_in_prev_stage > 0:
            
            attempt_schedule['decay_at'] = completed_schedule['decay_at'] + [steps_in_prev_stage] # add steps for this stage
            attempt_schedule['decay_amt'] = completed_schedule['decay_amt'] + [prev_decay_amt] # keep the same learning rate
            if verbose:
                print("attempt_schedule", attempt_schedule)
            # compute risk for this stage
            K_current = sum(attempt_schedule['decay_at']) + steps_in_stage
            if verbose:
                print(f"K_current for stage {stage+1} is {K_current}")
            scheduler_config = create_scheduler(attempt_schedule, K_current)
            #print(f"attempt_schedule: {attempt_schedule}, K_current: {K_current}")
            sgd = SGD(momentum=0.0, nesterov=False, lr=start_lr, 
                         #weight_decay=0.0, 
                         scheduler=scheduler_config)
        
            new_risk_at_attempt = risk(N=N, K=K_current, B=B, nqs=nqs0, folds=sgd)

            # now try to find the optimal decay for this stage
            # i.e. decay by 50%, and check if risk is lower, if so, keep it and
            #  try a lower learning rate (i.e. decay by 50% more), until we reach a point where risk is not decreasing
            trys = -1
            while new_risk_at_attempt < old_risk_at_attempt - LRA_tol: #- 0.001: #- 0.05: #0.001: !!!
                trys += 1
                if verbose:
                    print('trying stage', stage+1, 'attempt', trys,)
                if trys > 10:
                    raise ValueError("Too many attempts to find optimal learning rate for stage", stage)
                
                old_risk_at_attempt = new_risk_at_attempt
                risk_best = old_risk_at_attempt
                schedule_best = attempt_schedule
                if verbose:
                    print("before schedule", schedule_best)
                # add a decay of 50%
                attempt_schedule = copy.deepcopy(schedule_best)
                attempt_schedule['decay_amt'][-1] *= 0.5
                #print("after schedule", schedule_best)
                # compute risk for this stage
                scheduler_config = create_scheduler(attempt_schedule, K_current)
                if verbose:
                    print(f"Using scheduler config: {scheduler_config}")
                sgd = SGD(momentum=0.0, nesterov=False, lr=start_lr, 
                            #weight_decay=0.0, 
                            scheduler=scheduler_config)
                new_risk_at_attempt = risk(N=N, K=K_current, B=B, nqs=nqs0, folds=sgd)
                # print old and new risk

                #print(f"Old risk: {old_risk_at_attempt:.4f}, New risk: {new_risk_at_attempt:.4f}, ")
            completed_schedule = copy.deepcopy(schedule_best)

            steps_in_prev_stage = steps_in_stage
            #print("steps_in_prev_stage", steps_in_prev_stage)
            prev_decay_amt = schedule_best['decay_amt'][-1]


        K_values.append(K_current)
        risks.append(risk_best)
        if verbose:
            print(f"Stage {stage+1}/{n_stages}: K={K_current:6d}, Risk={risk_best:.4f}, "
              f"Learning Rate Decay ={schedule_best['decay_amt'][-1]:.4f}, ")
        
    return K_values, risks, schedule_best




def EM_nqs_from_cfg_six_optimized_old(nqs_cfg, working_file_path = None, lr_adapt_tolerance=0.05):

    a = nqs_cfg.a
    b = nqs_cfg.b
    sigma = nqs_cfg.sigma
   # sigma = nqs_cfg.sigma => to be calculated later
    m_b = nqs_cfg.m_b
    m_a = nqs_cfg.m_a #m_b #1 #nqs_cfg.m_b => to be calculated later
    c = nqs_cfg.eps
   # eps = nqs_cfg.eps

    h = nqs_cfg.h
    init_lr = h.lr 
    #beta = h.momentum
    B = h.B
    K = h.K
    N = h.N
    #step_decay_schedule = h.step_decay_schedule
    
    if not h.lr_schedule == "optimized":
        raise ValueError("optimized not requested for EM lr")

    nqs_params = {
                'a': a,
                'b': b,
                'ma': m_a,
                'mb': m_b,
                'c': c,
                'sigma':sigma
            }
    
    K_values, risks, schedule_best = compute_risk_trajectory_opt_lr(B = B, 
        K_target = K, 
        learning_rate = init_lr, 
        N = N,
        nqs_params= nqs_params,
        stair_width = 1000, verbose = True, 
        LRA_tol = LRA_tol)
    
    #nqs_risks = jnp.array(risks)
    #print(a, b, sigma, init_lr, lr_schedule, beta, B, K, N, ma, mb)
    # convert into a dataframe, with index iteration (starting from 0)
    nqs_df = pd.DataFrame({"nqs_iter": K_values, 
                           "nqs_risk": risks})
    #raise ValueError('nqs_df'+ str(nqs_df))
    
    #nqs_df["bayes_risk"] = jnp.zeros(len(bnbs)) # bayes risk is rolled into bias

    return {"nqs_df": nqs_df}#, "lros_df": lros_df}



def EM_nqs_from_cfg(nqs_cfg, working_file_path = None):

    a = nqs_cfg.a
    b = nqs_cfg.b
    sigma = 1.
   # sigma = nqs_cfg.sigma => to be calculated later
    m_b = nqs_cfg.m_b
    m_a = m_b #1 #nqs_cfg.m_b => to be calculated later
    eps = 0.
   # eps = nqs_cfg.eps

    h = nqs_cfg.h
    init_lr = h.lr 
    beta = h.momentum
    B = h.B
    K = h.K
    N = h.N
    step_decay_schedule = h.step_decay_schedule

    def create_scheduler(step_decay_schedule, K):
            # use json to convert the string to a dictionary
            #step_decay_schedule = json.loads(step_decay_schedule_str)
            decay_at = step_decay_schedule["decay_at"]
            decay_amt = step_decay_schedule["decay_amt"]
            # decay at is a list of the iteration at which the learning rate is decayed
            #milestones = [int(h.decay_at_1 * K), 
            #              int(h.decay_at_2 * K)]
            # write the above defn of milestones in a more general way, loop over decay_at
            milestones = [int(decay_at[0] * K)]
            for i in range(1, len(decay_at)):
                milestones.append(int(decay_at[i] * K))
            # decay_amt is the amount by which the learning rate is decayed
            #schedulers = [ConstantLR(init_lr, milestones[0]),
            #            ConstantLR(init_lr * h.decay_amt_1, 
            #                        milestones[1] - milestones[0]), 
            #            ConstantLR(init_lr * h.decay_amt_1 * h.decay_amt_2, 
            #                        K - milestones[1])]
            schedulers = [ConstantLR(1.0, milestones[0])]
            for i in range(1, len(decay_at)):
                schedulers.append(ConstantLR(1.0 * decay_amt[i-1], 
                                            milestones[i] - milestones[i-1]))
            schedulers.append(ConstantLR(1.0 * decay_amt[-1],
                                        K - milestones[-1]))
            
            scheduler = SequentialLR(milestones=milestones, schedulers=schedulers)

            return scheduler
    
    if h.lr_schedule == "step":
        scheduler = create_scheduler(step_decay_schedule, K)
        # raise an error that prints the schedule
        #raise ValueError("scheduler: ", scheduler)
    elif h.lr_schedule == "constant":
        scheduler = ConstantLR(1.0, K)
    else:
        raise ValueError("for fast nqs, lr_schedule must be one of 'step' or 'constant'")

    #learning_rate_schedule = hydra.utils.instantiate(h.optimizer.learning_rate)
    #def lr_schedule(t,K):
    #    return learning_rate_schedule(t)/learning_rate_schedule(0)

   # bayes_risk, biases, variances = nqs(a, b, sigma, init_lr, lr_schedule, beta, B, K, N,
   #                                     m_a = m_a, m_b = m_b)
    
    #iters_to_calc = [64, 1000, 8000, 64000, 512000, 4096000, 32768000]
    # keep only the iterations that are less than K
    #iters_to_calc = [k for k in iters_to_calc if k <= K]
    iters_to_calc = [K]
    # add K
    #iters_to_calc.append(K)

    # temporary use x64 (for jax and np)
    #with config.update("jax_enable_x64", True): 
    if True:
        bnbs = []
        vars = []
        optimizer = SGD(lr=init_lr, momentum=beta, nesterov=False, 
                            scheduler=scheduler)
        for k in iters_to_calc:
            # compute bias & approx error 
            nqs_params_bnb = NQS(a=a, b=b, ma=m_a, mb=m_b, c=0.0, sigma=0)
            bayes_and_bias = risk(N, k, B, nqs_params_bnb, optimizer, kind="fast")
            # compute variance
            nqs_params_tot = NQS(a=a, b=b, ma=m_a, mb=m_b, c=0.0, sigma=sigma)
            tot = risk(N, k, B, nqs_params_tot, optimizer, kind="fast")
            variance = tot - bayes_and_bias
    
            bnbs.append(bayes_and_bias)
            vars.append(variance)

    # convert the list of biases and variances into jnp arrays
    bnbs = jnp.array(bnbs)
    vars = jnp.array(vars)
    #print(a, b, sigma, init_lr, lr_schedule, beta, B, K, N, ma, mb)
    # convert into a dataframe, with index iteration (starting from 0)
    nqs_df = pd.DataFrame({"nqs_iter": iters_to_calc, 
                           "nqs_bias": bnbs,
                           "nqs_var": vars})
    
    nqs_df["bayes_risk"] = jnp.zeros(len(bnbs)) # bayes risk is rolled into bias

    return {"nqs_df": nqs_df}#, "lros_df": lros_df}



# --------------- Gamma Approximation

def approx_expected_nqs_from_cfg(nqs_cfg, working_file_path = None):
    a = nqs_cfg.a
    b = nqs_cfg.b
    sigma = 1 
   # sigma = nqs_cfg.sigma
    ma = nqs_cfg.m_a
    mb = nqs_cfg.m_b
   # eps = nqs_cfg.eps

    h = nqs_cfg.h
    init_lr = h.lr
    beta = h.momentum
    B = h.B
    K = h.K
    N = h.N
    learning_rate_schedule = hydra.utils.instantiate(h.learning_rate_schedule)

    def lr_schedule(t,K):
        return learning_rate_schedule(t)/learning_rate_schedule(0)

    bayes_risk, bias, variance = nqs_approx(a, b, sigma, init_lr, lr_schedule, beta, B, K, N,
                                            m_a = ma, m_b = mb)
    # convert into a dataframe, with index iteration (starting from 0)
    nqs_df = pd.DataFrame({"nqs_iter": [K], "nqs_bias": [bias], "nqs_var": [variance]})
    nqs_df["bayes_risk"] = bayes_risk
    return {"nqs_df": nqs_df}



# ------------- Exact Expected Risk

def expected_nqs_from_cfg(nqs_cfg, working_file_path = None):
    a = nqs_cfg.a
    b = nqs_cfg.b
    sigma = 1 
   # sigma = nqs_cfg.sigma => to be calculated later
    mb = nqs_cfg.m_b
    ma = mb #1 #nqs_cfg.m_b => to be calculated later
   # eps = nqs_cfg.eps

    h = nqs_cfg.h
    init_lr = h.lr 
    beta = h.momentum
    B = h.B
    K = h.K
    N = h.N
    learning_rate_schedule = hydra.utils.instantiate(h.optimizer.learning_rate)

    def lr_schedule(t,K):
        return learning_rate_schedule(t)/learning_rate_schedule(0)

    bayes_risk, biases, variances = nqs(a, b, sigma, init_lr, lr_schedule, beta, B, K, N,
                                        m_a = ma, m_b = mb)
    #print(a, b, sigma, init_lr, lr_schedule, beta, B, K, N, ma, mb)
    # convert into a dataframe, with index iteration (starting from 0)
    nqs_df = pd.DataFrame({"nqs_iter": range(len(biases)), "nqs_bias": biases, "nqs_var": variances})
    nqs_df["bayes_risk"] = jnp.ones(len(biases)) * bayes_risk

    # apply the learning rate schedule to range K
    #xss = jnp.arange(K)
    # apply the learning rate schedule to xss, with K = K 
    # (lr_schedule takes t, K)
    # vmap over t (the xss)
    #vmap_lr_schedule = vmap(lr_schedule, in_axes=(0, None))
    #learning_rates_on_schedule = vmap_lr_schedule(xss, K)
    #lros_df = pd.DataFrame({"iteration": xss, "learning_rate": learning_rates_on_schedule})
    return {"nqs_df": nqs_df}#, "lros_df": lros_df}


# ------------------ Lower Level Functions

def nqs_approx(a, b, sigma, init_lr, lr_schedule, beta, B, K, N, m_a = 1, m_b = 1, exact = False):#, i_0 = 1):
    # use incomplete gamma function to approximate the bayes risk, bias and variance
    bayes_risk = 0.5 * (zeta(a, N+1)) * m_a 

    def Gamma_left(s,x):
        return gamma(s) - gammainc(s,x)
    # let eta be applying lr_schedule to the x_i's * K
    def f_eta(x):
        return lr_schedule((x*K), K)*init_lr/(1-beta)

    def Pi_fac(a,b,m_b,N,K,f_eta, steps = 1000):
        # let x_i be evenly spaced from 0 to N, there are steps number of x_i's
        x_i = jnp.linspace(0, 1, steps)
        c_ab = a/b - 1/b 
        f_etas = vmap(f_eta)(x_i)
        c_eta = 2*m_b*jnp.mean(f_etas)
        pi_factor = m_a*c_eta**(-c_ab)/b * K**(-c_ab) *(Gamma_left(c_ab, c_eta*K*N**(-b)) - Gamma_left(c_ab, c_eta*K))
        
        if exact:
            # pi factor is sum_{i=1}^N (i^{-a} (prod_{k=1}^K (1-eta_k * m_b * i^{-b}))^2)
            pi_factor_array = jnp.zeros(N)
            for i in range(N):
                pi_factor_array = pi_factor_array.at[i].set((i+1)**(-a))
                for k in range(K):
                    pi_factor_array = pi_factor_array.at[i].set(pi_factor_array[i] * (1 - f_eta(k/K) * m_b * (i+1)**(-b))**2)
            pi_factor = jnp.sum(pi_factor_array)
        
        return pi_factor 

    def Sigma_fac(b,m_b,N,K,f_eta,steps=1000):
        x_i = jnp.linspace(0, 1, steps)
        # add an element to x_i, so that the last element is 1
        etas = vmap(f_eta)(x_i)
        c_b = 2 - 1/b
        def integrand(r): # 0<r<1
            mask = jnp.where(x_i >= r, 1, 0)  # Create a mask
            eta_bar_r = jnp.sum(etas * mask) / steps
            gamma_facs = Gamma_left(c_b, 2*m_b*eta_bar_r*K*N**(-b)) - Gamma_left(c_b, 2*m_b*eta_bar_r*K)
            return f_eta(r)**2 * (2*eta_bar_r)**(-c_b)* gamma_facs
        # integrate the integrand from 0 to 1
        integrands = vmap(integrand)(x_i)
        integrated =0.5*(integrands[1:].mean() + integrands[:-1].mean())
        sigma_factor = integrated /K/(m_b**2) *(1/b *(K*m_b)**(1/b))

        if exact:
            # let sigma factor be sum_{i=1}^N (sum_{k=1}^K i^{-2b} eta_k^2 (prod_{l=k+1}^K (1-eta_l m_b i^{-b})^2))
            # first get an array of dim NxK, where each element is (i,k) is  i^{-2b} eta_k^2 (prod_{l=k+1}^K (1-eta_l * m_b * i^{-b})^2))
            sig_fac_array = jnp.zeros((N,K))
            for i in range(N):
                for k in range(K):
                    sig_fac_array = sig_fac_array.at[i,k].set((i+1)**(-2*b) * f_eta(k/K)**2)
                    for l in range(k+1, K):
                        sig_fac_array = sig_fac_array.at[i,k].set(sig_fac_array[i,k] * (1 - f_eta(l/K) * m_b * (i+1)**(-b))**2)
            # get sigma factor by summing over the array
            sigma_factor = jnp.sum(sig_fac_array) 
        return sigma_factor

    bias = 0.5 * m_a * Pi_fac(a,b,m_b,N,K,f_eta)
    var = sigma**2 * 1/B * 0.5* (m_b)**2 * Sigma_fac(b,m_b,N,K,f_eta)
    return bayes_risk, bias, var


def nqs(a, b, sigma, init_lr, lr_schedule, beta, B, K, N, m_a = 1, m_b = 1, i_0 = 1):
    # outputs the bayes_risk, bias, variance, and total error of the noisy quadratic system
    # a, b: floats, the exponents of the quadratic system
    # lr_schedule: function, the learning rate schedule
    # B: float, the batch size
    # K: int, the number of iterations
    # N: int, the dimension of the parameter space

    # initialise nqs
    theta_init, bias, var, bayes_risk, v, d = nqs_init(a, b, N, m_a, m_b, i_0)
    v_bias = v
    # run nqs optimization
    num_iters = K

    biases = jnp.zeros(num_iters+1)
    variances = jnp.zeros(num_iters+1)
    biases = biases.at[0].set(bias)
    variances = variances.at[0].set(var)

    for i in range(num_iters):
        lr_i = init_lr * lr_schedule(i,K)
        # hard code B for numerical stability - will adjust back for variance
        B_for_calcs = 1
        T_i, n_i = nqs_opt_values(lr_i, beta, B_for_calcs, sigma, d)
        bias, var, v_bias, v = nqs_opt_step(v_bias, v, T_i, n_i, d, lr_i)
        biases = biases.at[i+1].set(bias)
        variances = variances.at[i+1].set(var/B*B_for_calcs)
    
    #print(bias, var, bayes_risk)
    return bayes_risk, biases, variances





def nqs_init(a, b, N, m_a, m_b, i_0 = 1):
    # vector of length N, where theta_i = i^{0.5*(b-a)}
    power = 0.5 * (b - a)
    theta_init = jnp.power(jnp.arange(i_0, N+1), power) * jnp.sqrt(m_a/m_b)
    bias_init = 0.5 * m_a * jnp.sum(jnp.power(jnp.arange(i_0, N+1), -a))
    var_init = 0
    bayes_risk_init = 0.5 * (zeta(a, N+1)) * m_a 

    v_0_init = jnp.square(theta_init)
    v_init = [v_0_init, jnp.zeros(N-i_0+1), jnp.zeros(N-i_0+1)]
    # convert v_init to shape (N, 3) by trasposing
    v_init = jnp.array(v_init).T
    d = m_b * jnp.power(jnp.arange(i_0, N+1), -b) # shape (N,)
    return theta_init, bias_init, var_init, bayes_risk_init, v_init, d

@jit
def nqs_opt_values(lr, beta, B, sigma, d):
    c = sigma**2 * d

    def Td(d, lr, beta):

        Td = jnp.array([[jnp.square(1-lr*d), jnp.square(beta), 2*(1-lr*d)*beta],
                        [jnp.square(lr*d), jnp.square(beta), -2*lr*beta*d],
                        [-(1-lr*d)*lr*d, jnp.square(beta), (1-2*lr*d)*beta]])
        return Td
    
    T = vmap(Td, in_axes=(0, None, None))(d, lr, beta) # (N, 3, 3)
    
    # the first dimension of n is of size same as d
    # where the d-th element is d * jnp.array([lr**2 * c / B, lr**2 * c / B, lr**2 * c / B]) # (3,)
    n = c[:,None] * jnp.array([lr**2/ B, lr**2/ B, lr**2/ B])[None,:] # (N, 3)

    return T, n


@jit
def nqs_opt_step(v_bias, v, T, n, d, lr):
    # v, v_bias have shape (N,3)
    # T has shape (N, 3, 3)
    # n has shape (3,)
    # multiply the 2nd column of v_bias by lr^2
    # multiply the 3rd column of v_bias by lr
    v_bias = v_bias.at[:,1].set(v_bias[:,1] * lr**2) # here v_bias = [A(m), A(theta), -C(t)]
    v_bias = v_bias.at[:,2].set(v_bias[:,2] * lr)
    v = v.at[:,1].set(v[:,1] * lr**2)
    v = v.at[:,2].set(v[:,2] * lr)

    v_bias = jnp.einsum('ijk,ik->ij', T, v_bias) # progression without noise
    v = jnp.einsum('ijk,ik->ij', T, v) + n # progression with noise

    # undo the scaling of the 2nd and 3rd columns of v_bias
    v_bias = v_bias.at[:,1].set(v_bias[:,1] / lr**2)
    v_bias = v_bias.at[:,2].set(v_bias[:,2] / lr)
    v = v.at[:,1].set(v[:,1] / lr**2)
    v = v.at[:,2].set(v[:,2] / lr)

    bias = 0.5 * (d*v_bias[:,0]).sum()
    excess_risk = 0.5 * (d*v[:,0]).sum()
    var = excess_risk - bias
    return bias, var, v_bias, v

