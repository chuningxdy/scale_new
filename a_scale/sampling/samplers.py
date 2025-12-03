import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
import json
import scipy.stats as stats
#from a_scale.design_architecture import ModelArchitecture
from a_scale.run_nn import train_nn, build_nn

# ------- helpers -----------
def to_float_and_round(x, round_to):
            x_float = float(x)
            x_round = round(x_float, round_to)
            return x_round

def custom_round(x, log_base = 10, good_multiples = [1, 3, 10]):
            # round x such that there is only one significant digit
            # do this by rounding down to the nearest power of log_base
            # then multiply by the integer multiple of lr/nearest power of log_base
            nearest_power_of_log_base = log_base**jnp.floor(jnp.log(x)/jnp.log(log_base))
            multiple = x/nearest_power_of_log_base
            # among the integers 1, 3, 7, decide which is closest to multiple
            #good_multiples = [1, 3, 10]
            nearest_good_multiple = min(good_multiples, key = lambda x: jnp.abs(jnp.log(x) - jnp.log(multiple)))
            return nearest_power_of_log_base * nearest_good_multiple

def sample_integer_from_simplex(sum, key):
    # sample from the dirichlet distribution
    x = jax.random.dirichlet(key, jnp.ones(3))
    x = x * sum
    # get the non-integer parts
    y = x - jnp.floor(x)
    index_ymax = jnp.argmax(y)
    x = x.at[index_ymax].set(jnp.ceil(x[index_ymax]))
    index_ymin = jnp.argmin(y)
    x = x.at[index_ymin].set(jnp.floor(x[index_ymin]))
    index_ymid = 3 - index_ymax - index_ymin
    x = x.at[index_ymid].set(sum - x[index_ymax] - x[index_ymin])
    return x

def sample_integer_from_line(sum, key):
    x = jax.random.dirichlet(key, jnp.ones(2))
    x = x * sum
    y = x - jnp.floor(x)
    index_ymax = jnp.argmax(y)
    x = x.at[index_ymax].set(jnp.ceil(x[index_ymax]))
    index_ymin = jnp.argmin(y)
    x = x.at[index_ymin].set(sum - x[index_ymax])
    return x



# ------- samplers -----------

def h_sampler_from_csv(file_path, seq_len):
    '''
    read a csv file with hyper values
    '''
    df = pd.read_csv(file_path)
    print("read csv file: "+file_path)
    #raise ValueError("STOPPED")

    # check if the columns are present and no missing values
    required_columns = ["N", "B", "K", "lr", "end_lr", "momentum", "lr_schedule", "optimizer", "step_decay_schedule"]
    df_has_C = "C" in df.columns
    for col in required_columns:
        if col not in df.columns:
            raise ValueError("Missing column: "+col)
        if df[col].isnull().values.any():
            raise ValueError("Missing values in column: "+col)
    # convert N, B, K to integers
    df["N"] = df["N"].astype(int)
    df["B"] = df["B"].astype(int)
    df["K"] = df["K"].astype(int)
    # check lr, end_lr, momentum are floats
    if not df["lr"].apply(lambda x: isinstance(x, float)).all():
        raise ValueError("lr is not a float")
    if not df["end_lr"].apply(lambda x: isinstance(x, float)).all():
        raise ValueError("end_lr is not a float")
    if not df["momentum"].apply(lambda x: isinstance(x, float)).all():
        raise ValueError("momentum is not a float")
    #raise ValueError("momentum:" + str(df["momentum"].unique()))
    # check lr_schedule, optimizer are strings
    if not df["lr_schedule"].apply(lambda x: isinstance(x, str)).all():
        raise ValueError("lr_schedule is not a string")
    if not df["optimizer"].apply(lambda x: isinstance(x, str)).all():
        raise ValueError("optimizer is not a string")
    # check step_decay_schedule is a string
    if not df["step_decay_schedule"].apply(lambda x: isinstance(x, str)).all():
        raise ValueError("step_decay_schedule is not a string")
    
    # remove duplicates
    df = df.drop_duplicates()
    

    
    if not df_has_C:
        df["C"] = 6 * df["N"]/1000 * df["B"]/1000 * df["K"]/1000 * seq_len
    df["T"] = df["K"]
    df["memory"] = df["N"]/1000 * df["B"]/1000
    ct_summary = pd.DataFrame({"C": [df["C"].sum()], "T": [df["T"].sum()], "i": [df.shape[0] ]})
    memory_df = pd.DataFrame({"memory": df["memory"]})

    # drop T and memory from df
    df = df.drop(columns = ["T", "memory"])
    # if df does not have column C from the csv file, delete C
    if not df_has_C:
        df = df.drop(columns = ["C"])

    # only keep the required columns
    #df = df[required_columns]

    

    return {"samples": df, "ct_summary": ct_summary, "memory": memory_df}


def h_sampler_semigrid(
        rand_key,
        max_N, min_N, lvls_N,
        max_B, min_B, lvls_B,
        max_K, min_K, lvls_K,
        min_lr, max_lr, lvls_lr,
        end_lr,
        momentum,
        lr_schedule, 
        optimizer, 
        step_decay_schedule,
        seq_len,
        traj = False):
    '''
    produce a csv file with hyper values
    for end_lr, lr_schedule, optimizer, momentum, step_decay_schedule, simply use the input values
    for N, B, K, sample from a semi-grid:
        if min == max, use that value
        for the dimensions where min < max, use a semi-grid
        - incl. a row for (min_N, min_B, min_K) 
        - incl. rows with (min_N, min_B, K) where K is from a grid between min_K and max_K, with lvls_K levels
        - incl. rows with (min_N, B, min_K) where B is from a grid between min_B and max_B, with lvls_B levels
        - incl. rows with (N, min_B, min_K) where N is from a grid between min_N and max_N, with lvls_N levels
        - remove duplicates
    count how many samples were generated for N, B, K
    then populate lr randomly from a grid using lvls_lr levels
    check how many unique values were generated for lr, make sure it is at least 3
    if not, raise an error

    '''

    # start with an empty dataframe
    df = pd.DataFrame()
    
    # get values for N, B, K, incl. start and end points
    # linspace on the log scale
    Ns = jnp.logspace(jnp.log2(min_N), jnp.log2(max_N), lvls_N, base = 2)
    Bs = jnp.logspace(jnp.log2(min_B), jnp.log2(max_B), lvls_B, base = 2)
    Ks = jnp.logspace(jnp.log2(min_K), jnp.log2(max_K), lvls_K, base = 2)

    # make sure N, B, K are integers
    Ns = [int(x) for x in Ns]
    Bs = [int(x) for x in Bs]
    Ks = [int(x) for x in Ks]
    
    # generate the semi-grid
    new_row = pd.DataFrame({"N": [min_N], "B": [max_B], "K": [min_K]})
    df = pd.concat([df, new_row], axis = 0)
    if min_K < max_K:
        for K in Ks:
            new_row = pd.DataFrame({"N": [min_N], "B": [max_B], "K": [K]})
            df = pd.concat([df, new_row], axis = 0)
    if min_B < max_B:
        for B in Bs:
            new_row = pd.DataFrame({"N": [min_N], "B": [B], "K": [min_K]})
            df = pd.concat([df, new_row], axis = 0)
    if min_N < max_N:
        for N in Ns:
            new_row = pd.DataFrame({"N": [N], "B": [max_B], "K": [min_K]})
            df = pd.concat([df, new_row], axis = 0)

    # remove duplicates
    df = df.drop_duplicates()
    # reset index
    df = df.reset_index(drop = True)

    # compare each pair of rows
    # if two rows have similar N, B, K (less than 50% diff in each), keep the first one
    similar_pairs = []
    for i in range(df.shape[0]):
        for j in range(i+1, df.shape[0]):
            N_diff = abs(df.iloc[i]["N"] - df.iloc[j]["N"])/df.iloc[i]["N"]
            B_diff = abs(df.iloc[i]["B"] - df.iloc[j]["B"])/df.iloc[i]["B"]
            K_diff = abs(df.iloc[i]["K"] - df.iloc[j]["K"])/df.iloc[i]["K"]
            if N_diff < 0.5 and B_diff < 0.5 and K_diff < 0.5:
                similar_pairs.append(j)
   
    # remove duplicates in similar_pairs
   # raise ValueError(df.index)
    similar_pairs = list(set(similar_pairs))
    df = df.drop(similar_pairs)

    # group rows by N, B
    # take the largest K for each group

    if traj:
        df = df.groupby(["N", "B"]).agg({"K": "max"}).reset_index()
    
    # for each row, generate a random lr
    key = jax.random.PRNGKey(rand_key)
    if min_lr < max_lr:
        lrs = jnp.logspace(jnp.log10(min_lr), jnp.log10(max_lr), lvls_lr)
    elif min_lr == max_lr:
        lrs = [min_lr]
        # convert to array
        lrs = jnp.array(lrs)
    else:
        raise ValueError("Invalid lr bounds")
    
    lr_col = []
    for i in range(df.shape[0]):
        lr_key, key = jax.random.split(key)
        lr = jax.random.choice(lr_key, lrs)
        # to float and round
        lr = to_float_and_round(lr, 5)
        lr_col.append(lr)
    
    df["lr"] = lr_col

    # add the other hyperparameters
    df["end_lr"] = end_lr
    df["momentum"] = momentum
    df["lr_schedule"] = lr_schedule
    df["optimizer"] = optimizer
    # convert step_decay_schedule to a dictionary of lists
    step_decay_schedule_dict = {key: list(value) for key, value in step_decay_schedule.items()}
    df["step_decay_schedule"] = json.dumps(step_decay_schedule_dict)

    # get the ct_summary (total C, T, i)
    C = 6 * df["N"]/1000 * df["B"]/1000 * df["K"]/1000 * seq_len
    T = df["K"]
    i = df.shape[0]
    ct_summary = pd.DataFrame({"C": [C.sum()], "T": [T.sum()], "i": [i]})

    # get the memory values
    Ms = df["N"]/1000 * df["B"]/1000
    memory_df = pd.DataFrame({"memory": Ms})

    return {"samples": df, "ct_summary": ct_summary, "memory": memory_df}


    
def h_sampler_single_point(
        N, B, K,
        lr, end_lr, momentum,
        lr_schedule, optimizer, step_decay_schedule, seq_len):
    '''
    produce a csv file with hyper values
    for a single point
    '''
    # start with an empty dataframe
    df = pd.DataFrame()

    step_decay_schedule = {key: list(value) for key, value in step_decay_schedule.items()}

    new_row = pd.DataFrame({"N": [N], "B": [B], "K": [K], 
                            "lr": [lr], "end_lr": [end_lr], "momentum": [momentum], 
                            "lr_schedule": [lr_schedule], "optimizer": [optimizer], 
                            "step_decay_schedule": [json.dumps(step_decay_schedule)]})
    df = pd.concat([df, new_row], axis = 0)

    # get the ct_summary (total C, T, i)
    C = 6 * N/1000 * B/1000 * K/1000 * seq_len
    T = K
    i = 1
    ct_summary = pd.DataFrame({"C": [C], "T": [T], "i": [i]})
    # get the memory values
    Ms = N/1000 * B/1000
    memory_df = pd.DataFrame({"memory": [Ms]})
    return {"samples": df, "ct_summary": ct_summary, "memory": memory_df}





def h_sampler_time_compute_budget(
        compute_budget, 
        memory_budget,  
        time_budget,
        max_frac,
        max_time_frac,
        max_n_runs,
        rand_key,
        max_N, min_N,
        max_B, min_B,
        max_K, min_K,
        min_lr, max_lr,
        min_end_lr, max_end_lr,
        min_momentum, max_momentum,
        lr_schedule_options, optimizer_options,
        permitted_ckpts,
        max_attempts,
        step_decay_schedule):
    '''
    produce a csv file with hyper values
    random sampling, subject to compute and time budget
    '''

    # ---- helpers ----
    def random_choice_from_list(inlist, key):
                # randomly sample from a list
                index = jax.random.randint(key, minval = 0, maxval = len(inlist), shape = ())
                index = int(index)
                inlist = [item for item in inlist]
                return inlist[index]
    
    def reject(N, B, K):

        N_reject = N < min_N or N > max_N
        B_reject = B < min_B or B > max_B
        K_reject = K < min_K or K > max_K
        memory_reject = N/1000 * B/1000 > memory_budget
        time_reject = K/time_budget > max_time_frac
        return N_reject or B_reject or K_reject or memory_reject or time_reject
    
    # convert step_decay_schedule to a dictionary of lists
    step_decay_schedule = {key: list(value) for key, value in step_decay_schedule.items()}

    # remove -1 from permitted_ckpts
    permitted_ckpts = [x for x in permitted_ckpts if x != -1]

    # start with an empty dataframe
    df = pd.DataFrame()
    # the ranges are tuples of the form (min, max)
    # sample a from a_range, use scipy uniform distribution
    # jax random key:

    n_samples = max_n_runs
    T = 0
    C = 0
    samples_collected = 0
    # let Ms be an empty jnp array
    Ms = jnp.array([])

    # generate a random key from the seed rand_key
    key = jax.random.PRNGKey(rand_key)

    rejects_count = 0

    while samples_collected < n_samples and T < time_budget and C < compute_budget:
        compute_budget_i_key, key = jax.random.split(key)
        # randomly sample a fraction between 1/10 and 0.5 of the compute budget
        compute_budget_i_frac = jax.random.uniform(compute_budget_i_key, minval = 0.1, maxval = max_frac)
        NBK_budget_i = compute_budget_i_frac * compute_budget/6

        if min_B == max_B and min_N == max_N and min_K == max_K:
            N = min_N
            B = min_B
            K = min_K
        elif min_B == max_B and min_N < max_N and min_K < max_K:
            B = min_B
            NK_budget_i = NBK_budget_i / B
            log_NK = jnp.log(NK_budget_i)/jnp.log(2) + jnp.log(1000)/jnp.log(2)*3
            key_ink, key = jax.random.split(key)
            logNK = sample_integer_from_line(log_NK, key_ink)
            N, K = 2**logNK[0], 2**logNK[1]
        elif min_B < max_B and min_N < max_N and min_K < max_K:
            log_NBK = jnp.log(NBK_budget_i)/jnp.log(2) + jnp.log(1000)/jnp.log(2)*3
            # round to the nearest integer
            log_NBK = jnp.round(log_NBK)
            
            # sample N, B, K
            key_inbk, key = jax.random.split(key)
            logNBK = sample_integer_from_simplex(log_NBK, key_inbk)
            N, B, K = 2**logNBK[0], 2**logNBK[1], 2**logNBK[2]
        else:
             raise ValueError("Invalid NKB bounds for random isoflop samples")
        # round N, B, K
        N = int(N)
        B = int(B)
        K = int(K)

        # reject if N, B, K are not in the range
        if reject(N, B, K):
            rejects_count += 1
            if rejects_count > max_attempts:
                 raise ValueError("Too many attempts to sample from time and compute budget:"+
                                  "\nlast N, B, K: "+str(N)+", "+str(B)+", "+str(K))
            continue
        else:
            # sample lr, momentum, lr_schedule
            if min_lr == max_lr:
                lr = min_lr
            else:
                key_lr, key = jax.random.split(key)
                lr = jnp.exp(jax.random.uniform(key_lr, minval = jnp.log(min_lr), maxval = jnp.log(max_lr)))
                lr = custom_round(lr, log_base = 10)

            if min_end_lr == max_end_lr:
                end_lr = min_end_lr
            else:
                key_end_lr, key = jax.random.split(key)
                end_lr = jnp.exp(jax.random.uniform(key_end_lr, minval = jnp.log(min_end_lr), maxval = jnp.log(max_end_lr)))
                end_lr = custom_round(end_lr, log_base = 10)
            
            if min_momentum == max_momentum:
                momentum = min_momentum
            else:
                key_momentum, key = jax.random.split(key)
                oneminusmomentum = jnp.exp(jax.random.uniform(key_momentum, minval = jnp.log(1-min_momentum), maxval = jnp.log(1-max_momentum)))
                oneminusmomentum = custom_round(oneminusmomentum, log_base = 10)
                momentum = 1 - oneminusmomentum
            

            key_lr_schedule, key = jax.random.split(key)
            lr_schedule = random_choice_from_list(lr_schedule_options, key_lr_schedule)
                
            key_optimizer, key = jax.random.split(key)
            optimizer = random_choice_from_list(optimizer_options, key_optimizer)

            permitted_ckpts_in_run = [x for x in permitted_ckpts if x < K]

         #   for K_j in permitted_ckpts_in_run:
         #       new_row = pd.DataFrame({"K": [K_j], 
         #                               "N": [N],
         #                               "B": [B],
         #                               "lr": [lr],
         #                               "end_lr": [end_lr],
         #                               "momentum": [momentum],
         #                               "lr_schedule": [lr_schedule],
         #                               "optimizer": [optimizer]})
         #       df = pd.concat([df, new_row], axis = 0)
            
            # if lr_schedule is constant or cosine,
            # let step_decay_schedule be "na"
            if lr_schedule == "constant" or lr_schedule == "cosine":
                step_decay_schedule_val = "na"
            elif lr_schedule == "step":
                step_decay_schedule_val = json.dumps(step_decay_schedule)
            else:
                raise ValueError("Invalid lr_schedule")
            new_row = pd.DataFrame({"K": [K], 
                                    "N": [N],
                                    "B": [B],
                                    "lr": [lr],
                                    "end_lr": [end_lr],
                                    "momentum": [momentum],
                                    "lr_schedule": [lr_schedule],
                                    "optimizer": [optimizer],
                                    "step_decay_schedule": [step_decay_schedule_val]})
            df = pd.concat([df, new_row], axis = 0)

            #df = df.drop_duplicates()

            # update T, C, samples_collected
            T += K
            C += 6*(N/1000)*(B/1000)*(K/1000)
            samples_collected += 1
            # add the memory to Ms
            Ms = jnp.append(Ms, N/1000 * B/1000)
            

    df["type"] = "c_t_budget"

    # save a csv file with the total C, T, i
    summary_df = pd.DataFrame({"C": [C], "T": [T], "i": [samples_collected]})
    # save a csv file with the memory values
    memory_df = pd.DataFrame({"memory": Ms})
    # degbug stop
    #raise ValueError("STOPPED")
    return {"samples": df, "ct_summary": summary_df, "memory": memory_df}




def h_sampler_grid_and_random_isoflop_old(
                                    compute_budget, time_budget,
                                    use_grid,use_random_isoflops,
                                    isoflop_fracs,
                                    n_samples,
                                    rand_key,
                                    space_N,space_B,space_K,
                                    space_lr,space_end_lr,space_momentum,
                                    max_N,min_N,
                                    max_B,min_B,
                                    max_K,min_K,
                                    min_lr,max_lr,
                                    min_end_lr,max_end_lr,
                                    min_momentum,max_momentum,
                                    lr_schedule_options, optimizer_options,
                                    path = None):
    '''
    produce a csv file with hyper values
    random sampling, first from a series of isoflops, then from a grid
    '''
    def random_sample_from_simplex(key):
        # dirichlet distribution
        x = jax.random.dirichlet(key, jnp.ones(3))
        return x
    
    def random_sample_from_line_segment(key):
        # uniform distribution
        x = jax.random.uniform(key, minval = 0, maxval = 1)
        return x
    
    def array_with_max_min_space(X_max, X_min, X_space, logbase):
        # randomly sample from an array with max, min, and space, incl. the endpoints
        choices = jnp.arange(jnp.log(X_min)/jnp.log(logbase), 
                             jnp.log(X_max)/jnp.log(logbase) + X_space, 
                             X_space)
        choices = logbase**choices
        return choices
    
    def random_choice_from_list(inlist, key):
        # randomly sample from a list
        index = jax.random.randint(key, minval = 0, maxval = len(inlist), shape = ())
        index = int(index)
        inlist = [item for item in inlist]
        return inlist[index]

    def random_NKB_from_isoflop(iso_flop_budget, key):
        if max_B == min_B and max_N > min_N and max_K > min_K:
            B = max_B
            NK_budget = iso_flop_budget/6/B
            log_NK_budget = jnp.log2(NK_budget) + jnp.log2(1000)*3
            log_N = random_sample_from_line_segment(key) * log_NK_budget
            N = 2**log_N
            K = 2**(log_NK_budget - log_N)
        
        elif max_N == min_N and max_B > min_B and max_K > min_K:
            N = max_N
            BK_budget = iso_flop_budget/6/N
            log_BK_budget = jnp.log2(BK_budget) + jnp.log2(1000)*3
            log_B = random_sample_from_line_segment(key) * log_BK_budget
            B = 2**log_B
            K = 2**(log_BK_budget - log_B)

        elif max_N > min_N and max_B > min_B and max_K > min_K:
            NKB_budget = iso_flop_budget/6
            log_NKB_budget = jnp.log2(NKB_budget) + jnp.log2(1000)*3
            log_NKB = random_sample_from_simplex(key) * log_NKB_budget
            N, K, B = 2**log_NKB[0], 2**log_NKB[1], 2**log_NKB[2]

        else:
            raise ValueError("Invalid NKB bounds for random isoflop samples" +
                            "\nN: "+str(min_N)+", "+str(max_N)+
                            "\nB: "+str(min_B)+", "+str(max_B)+
                            "\nK: "+str(min_K)+", "+str(max_K))

        return N, K, B
    
    def reject(N, B, K):
        N_reject = N < min_N or N > max_N
        B_reject = B < min_B or B > max_B
        K_reject = K < min_K or K > max_K
        T_reject = K > time_budget
        return N_reject or B_reject or K_reject or T_reject
    
    def random_sample_from_isoflop(iso_flop_budget, n_samples, key):
        samples_collected = 0
        df = pd.DataFrame()
        attempts = 0
        while samples_collected < n_samples:
            attempts += 1
            if attempts > 100000:
                raise ValueError("Too many attempts to sample from isoflop")
            key_NKB, key = jax.random.split(key)
            N,K,B = random_NKB_from_isoflop(iso_flop_budget, key_NKB)
            if reject(N, B, K):
                continue
            else:
                # randomly sample lr, momentum, lr_schedule, optimizer
                key_lr, key = jax.random.split(key)
                lr = jnp.exp(jax.random.uniform(key_lr, minval = jnp.log(min_lr), maxval = jnp.log(max_lr)))
                lr = custom_round(lr, log_base = 10)

                key_end_lr, key = jax.random.split(key)
                end_lr = jnp.exp(jax.random.uniform(key_end_lr, minval = jnp.log(min_end_lr), maxval = jnp.log(max_end_lr)))
                end_lr = custom_round(end_lr, log_base = 10)
                
                key_momentum, key = jax.random.split(key)
                oneminusmomentum = jnp.exp(jax.random.uniform(key_momentum, minval = jnp.log(1-min_momentum), maxval = jnp.log(1-max_momentum)))
                oneminusmomentum = custom_round(oneminusmomentum, log_base = 10)
                momentum = 1 - oneminusmomentum
                
                key_lr_schedule, key = jax.random.split(key)
                lr_schedule = random_choice_from_list(lr_schedule_options, key_lr_schedule)
                
                key_optimizer, key = jax.random.split(key)
                optimizer = random_choice_from_list(optimizer_options, key_optimizer)

                new_row = pd.DataFrame({"N": [N],
                                        "B": [B],
                                        "K": [K],
                                        "lr": [lr],
                                        "end_lr": [end_lr],
                                        "momentum": [momentum],
                                        "lr_schedule": [lr_schedule],
                                        "optimizer": [optimizer]})
                df = pd.concat([df, new_row], axis = 0)
                samples_collected += 1

        df["type"] = "random_isoflop"
        return df
    
    def grid_sample():
        df = pd.DataFrame()

        Ns = array_with_max_min_space(max_N, min_N, space_N, 2)
        Bs = array_with_max_min_space(max_B, min_B, space_B, 2)
        Ks = array_with_max_min_space(max_K, min_K, space_K, 2)
        lrs = array_with_max_min_space(max_lr, min_lr, space_lr, 10)
        end_lrs = array_with_max_min_space(max_end_lr, min_end_lr, space_end_lr, 10)
        oneminusmomentums = array_with_max_min_space(1-max_momentum, 1-min_momentum, space_momentum, 10)
        momentums = 1 - oneminusmomentums

        for N in Ns:
            for B in Bs:
                for K in Ks:
                    for lr in lrs:
                        for end_lr in end_lrs:
                            for momentum in momentums:
                                for lr_schedule in lr_schedule_options:
                                    for optimizer in optimizer_options:
                                        new_row = pd.DataFrame({"N": [N],
                                            "B": [B],
                                            "K": [K],
                                            "lr": [lr],
                                            "end_lr": [end_lr],
                                            "momentum": [momentum],
                                            "lr_schedule": [lr_schedule],
                                            "optimizer": [optimizer]})
                                        df = pd.concat([df, new_row], axis = 0)
        df["type"] = "grid"
        return df

    key = jax.random.PRNGKey(rand_key)

    df = pd.DataFrame()
    if use_random_isoflops:
        iso_flop_fracs_list = [item for item in isoflop_fracs]
        for iso_flop_frac in iso_flop_fracs_list:
            key_i, key = jax.random.split(key)
            iso_flop_budget = iso_flop_frac * compute_budget
            df_i = random_sample_from_isoflop(iso_flop_budget, n_samples, key_i)
            df = pd.concat([df, df_i], axis = 0)
    if use_grid:
        df_grid = grid_sample()
        df = pd.concat([df, df_grid], axis = 0)
    
    # make sure N, B, K are integers
    df["N"] = df["N"].astype(int)
    df["B"] = df["B"].astype(int)
    df["K"] = df["K"].astype(int)

    df["C"] = 6 * df["N"]/1000 * df["B"]/1000 * df["K"]/1000
    df["T"] = df["K"]

        
    return {"samples": df}

def sample_six_params_rand(n_samples, seed, 
                    a_min, a_max,
                    b_min, b_max,
                    ma_min, ma_max,
                    mb_min, mb_max,
                    eps_min, eps_max,
                    sigma_min, sigma_max
                    ):
    '''
    saves a csv of a, b, mb in the file
    randomly sampled from a range
    '''
    # start with an empty dataframe
    df = pd.DataFrame()
    # the ranges are tuples of the form (min, max)
    # sample a from a_range, use scipy uniform distribution
    # jax random key:
    key = jax.random.PRNGKey(seed)
    for i in range(n_samples): # use jax random sampling function
        key_i, key = jax.random.split(key)
        # split key_i into 3 keys
        key_a, key_b, key_ma, key_mb, key_eps, key_sigma = jax.random.split(key_i, 6)
        a = jax.random.uniform(key_a, minval = a_min, maxval = a_max)
        b = jax.random.uniform(key_b, minval = b_min, maxval = b_max)
        # for mb, use uniform on the log scale
        ma = jnp.exp(jax.random.uniform(key_ma, minval = jnp.log(ma_min), maxval = jnp.log(ma_max)))
        mb = jnp.exp(jax.random.uniform(key_mb, minval = jnp.log(mb_min), maxval = jnp.log(mb_max)))
        eps = jnp.exp(jax.random.uniform(key_eps, minval = jnp.log(eps_min), maxval = jnp.log(eps_max)))
        sigma = jnp.exp(jax.random.uniform(key_sigma, minval = jnp.log(sigma_min), maxval = jnp.log(sigma_max)))
        # use custom rounding
        #mb = custom_round(mb, log_base = 10, good_multiples = [1,2,3,4,5,6,7,8,9,10])

        a = to_float_and_round(a, 2)
        b = to_float_and_round(b, 2)
        ma = to_float_and_round(ma, 4)
        mb = to_float_and_round(mb, 4)
        eps = to_float_and_round(eps, 3)
        sigma = to_float_and_round(sigma, 4)
        # convert a, b, lr_mult to float
        
        # append the a, b, lr_mult to the dataframe
        new_row = pd.DataFrame({"a": [a], "b": [b], 
                                "m_a": [ma], "m_b": [mb], 
                                "eps": [eps], "sigma": [sigma]})
        df = pd.concat([df, new_row], axis = 0)

        #raise ValueError("df: "+str(df))

    return {"samples": df}

def sample_six_params_grid(
                    a_min, a_max, a_levels,
                    b_min, b_max, b_levels,
                    ma_min, ma_max, ma_levels,
                    mb_min, mb_max, mb_levels,
                    eps_min, eps_max, eps_levels,
                    sigma_min, sigma_max, sigma_levels
                    ):
    '''
    saves a csv of a, b, mb in the file
    randomly sampled from a range
    '''
    # start with an empty dataframe
    df = pd.DataFrame()
    # space a on the normal scale
    aa = jnp.linspace(a_min, a_max, a_levels)
    bb = jnp.linspace(b_min, b_max, b_levels)

    norm_const = jnp.exp(1) #1e5
    # space ma on the log scale
    mas = norm_const**jnp.linspace(jnp.log(ma_min)/jnp.log(norm_const), jnp.log(ma_max)/jnp.log(norm_const), ma_levels)
    mbs = norm_const**jnp.linspace(jnp.log(mb_min)/jnp.log(norm_const), jnp.log(mb_max)/jnp.log(norm_const), mb_levels)
    epss = norm_const**jnp.linspace(jnp.log(eps_min)/jnp.log(norm_const), jnp.log(eps_max)/jnp.log(norm_const), eps_levels)
    sigmas =norm_const**jnp.linspace(jnp.log(sigma_min)/jnp.log(norm_const), jnp.log(sigma_max)/jnp.log(norm_const), sigma_levels)
    
    # round and make sure they are floats
    aa = [to_float_and_round(x, 2) for x in aa]
    bb = [to_float_and_round(x, 2) for x in bb]
    mas = [to_float_and_round(x, 4) for x in mas]
    mbs = [to_float_and_round(x, 4) for x in mbs]
    epss = [to_float_and_round(x, 3) for x in epss]
    sigmas = [to_float_and_round(x, 4) for x in sigmas]

    # loop over all combinations of a, b, ma, mb
    for a in aa:
        for b in bb:
            for ma in mas:
                for mb in mbs:
                    for eps in epss:
                        for sigma in sigmas:
                            # append the a, b, lr_mult to the dataframe
                            new_row = pd.DataFrame({"a": [a], "b": [b], 
                                                    "m_a": [ma], "m_b": [mb], 
                                                    "eps": [eps], "sigma": [sigma]})
                            df = pd.concat([df, new_row], axis = 0)

    # remove duplicates
    df = df.drop_duplicates()

    return {"samples": df}

def sample_a_b_mb(n_samples, seed, 
                    a_min, a_max,
                    b_min, b_max,
                    mb_min, mb_max):
    '''
    saves a csv of a, b, mb in the file
    randomly sampled from a range
    '''
    # start with an empty dataframe
    df = pd.DataFrame()
    # the ranges are tuples of the form (min, max)
    # sample a from a_range, use scipy uniform distribution
    # jax random key:
    key = jax.random.PRNGKey(seed)
    for i in range(n_samples): # use jax random sampling function
        key_i, key = jax.random.split(key)
        # split key_i into 3 keys
        key_a, key_b, key_mb = jax.random.split(key_i, 3)
        a = jax.random.uniform(key_a, minval = a_min, maxval = a_max)
        b = jax.random.uniform(key_b, minval = b_min, maxval = b_max)
        # for mb, use uniform on the log scale
        mb = jnp.exp(jax.random.uniform(key_mb, minval = jnp.log(mb_min), maxval = jnp.log(mb_max)))
        # use custom rounding
        #mb = custom_round(mb, log_base = 10, good_multiples = [1,2,3,4,5,6,7,8,9,10])

        a = to_float_and_round(a, 2)
        b = to_float_and_round(b, 2)
        mb = to_float_and_round(mb, 4)
        # convert a, b, lr_mult to float
        
        # append the a, b, lr_mult to the dataframe
        new_row = pd.DataFrame({"a": [a], "b": [b], "m_b": [mb]})
        df = pd.concat([df, new_row], axis = 0)

    return {"samples": df}





def h_sampler_grid_and_random_isoflop(
                                    neural_net,     
                                    compute_budget, time_budget,
                                    use_grid,use_random_isoflops,
                                    isoflop_fracs,
                                    n_samples,
                                    rand_key,
                                    space_N,space_B,space_K,
                                    space_lr,space_end_lr,space_momentum,
                                    max_N,min_N,
                                    max_B,min_B,
                                    max_K,min_K,
                                    min_lr,max_lr,
                                    min_end_lr,max_end_lr,
                                    min_momentum,max_momentum,
                                    lr_schedule_options, optimizer_options,
                                    step_decay_schedule_options,
                                    seq_len,
                                    vocab_size, 
                                    path = None,
                                    respect_nn_archi = True):
    '''
    produce a csv file with hyper values
    random sampling, first from a series of isoflops, then from a grid
    '''

    compute_budget  = compute_budget/seq_len # this is 6N*D/seq_len = 6 NBK
    
    if respect_nn_archi:

        def actual_N(N):
            N = int(N)
            nn_dict = dict(neural_net.copy())
            h_dict = {}
            # add key "N" to h_dict
            h_dict["N"] = N
            actual_N_value = build_nn(nn_dict, h_dict)
                # get actual N using the design architecture class
            # archi = ModelArchitecture.design_model_architecture(N, vocab_size)
            # actual_N_value = ModelArchitecture.calculate_params_simplified(hidden_size=archi["hidden_size"], num_layers=archi["num_hidden_layers"]) + vocab_size * archi["hidden_size"]
            return actual_N_value

    else:
        def actual_N(N):
            N = int(N)
            return N


    def random_sample_from_simplex(key):
        # dirichlet distribution
        x = jax.random.dirichlet(key, jnp.ones(3))
        return x
    
    def random_sample_from_line_segment(key):
        # uniform distribution
        x = jax.random.uniform(key, minval = 0, maxval = 1)
        return x
    
    def array_with_max_min_space(X_max, X_min, X_space, logbase):
        # randomly sample from an array with max, min, and space, incl. the endpoints
        choices = jnp.arange(jnp.log(X_min)/jnp.log(logbase), 
                             jnp.log(X_max)/jnp.log(logbase) + X_space, 
                             X_space)
        choices = logbase**choices
        return choices
    
    def random_choice_from_list(inlist, key):
        # randomly sample from a list
        index = jax.random.randint(key, minval = 0, maxval = len(inlist), shape = ())
        index = int(index)
        inlist = [item for item in inlist]
        return inlist[index]

    def random_NKB_from_isoflop(iso_flop_budget, key):

        # fixed batch size, vary N, K => this is used for resource allocation
        if max_B == min_B and max_N > min_N and max_K > min_K:
            B = max_B
            NK_budget = iso_flop_budget/6/B
            log_NK_budget = jnp.log2(NK_budget) + jnp.log2(1000)*3
            log_NK_min = jnp.log2(min_N) + jnp.log2(min_K)
            log_NK_max = jnp.log2(max_N) + jnp.log2(max_K)
            # check that min < min(budget, max), if not raise an error
            if log_NK_min > log_NK_budget or log_NK_max < log_NK_budget:
                raise ValueError("NK bound does not contain budget")
            # define a line segment of budget that is within the bounds 
            log_N_min = jnp.maximum(jnp.log2(min_N), log_NK_budget - jnp.log2(max_K))
            log_N_max = jnp.minimum(jnp.log2(max_N), log_NK_budget - jnp.log2(min_K))
            log_N = random_sample_from_line_segment(key) * (log_N_max - log_N_min) + log_N_min
            
            N = actual_N(2**log_N)
            log_N = jnp.log2(N)
            K = 2**(log_NK_budget - log_N)


        

        # fixed N, vary B, K => this is used for hyper_selection (sampling logic for NQS)
        elif max_N == min_N and max_B > min_B and max_K > min_K:
            N = actual_N(max_N)
            BK_budget = iso_flop_budget/6/N
            log_BK_budget = jnp.log2(BK_budget) + jnp.log2(1000)*3
            log_BK_min = jnp.log2(min_B) + jnp.log2(min_K)
            log_BK_max = jnp.log2(max_B) + jnp.log2(max_K)
            # check that min < min(budget, max), if not raise an error
            if log_BK_min > log_BK_budget or log_BK_max < log_BK_budget:
                raise ValueError("BK bound does not contain budget")
            # define a line segment of budget that is within the bounds
            log_B_min = jnp.maximum(jnp.log2(min_B), log_BK_budget - jnp.log2(max_K))
            log_B_max = jnp.minimum(jnp.log2(max_B), log_BK_budget - jnp.log2(min_K))
            log_B = random_sample_from_line_segment(key) * (log_B_max - log_B_min) + log_B_min
            B = 2**log_B
            K = 2**(log_BK_budget - log_B)
            #raise ValueError("N, B, K: "+str(N)+", "+str(B)+", "+str(K))
        
        #elif max_N == min_N and max_B > min_B and max_K > min_K:
        #    N = actual_N(max_N)
        #    BK_budget = iso_flop_budget/6/N
        #    log_BK_budget = jnp.log2(BK_budget) + jnp.log2(1000)*3
        #    log_B = random_sample_from_line_segment(key) * log_BK_budget
        #    B = 2**log_B
        #    K = 2**(log_BK_budget - log_B)

        elif max_N > min_N and max_B > min_B and max_K > min_K:
            NKB_budget = iso_flop_budget/6
            log_NKB_budget = jnp.log2(NKB_budget) + jnp.log2(1000)*3
            log_NKB = random_sample_from_simplex(key) * log_NKB_budget
            tar_N, tar_K, B = 2**log_NKB[0], 2**log_NKB[1], 2**log_NKB[2]
            N = actual_N(tar_N)
            K = tar_K * tar_N/N # so that NK = tar_N * tar_K 

        else:
            raise ValueError("Invalid NKB bounds for random isoflop samples" +

                            "\nN: "+str(min_N)+", "+str(max_N)+
                            "\nB: "+str(min_B)+", "+str(max_B)+
                            "\nK: "+str(min_K)+", "+str(max_K))

        return N, K, B
    
    def reject(N, B, K):
        min_actual_N = actual_N(min_N)
        max_actual_N = actual_N(max_N)
        N_reject = N < min_actual_N or N > max_actual_N
        B_reject = B < min_B or B > max_B
        K_reject = K < min_K or K > max_K
        T_reject = K > time_budget

        return N_reject or B_reject or K_reject or T_reject
    
    def random_sample_from_isoflop(iso_flop_budget, n_samples, key):
        samples_collected = 0
        df = pd.DataFrame()
        attempts = 0
        while samples_collected < n_samples:
            attempts += 1
            if attempts > 1000:
                raise ValueError("Too many attempts to sample from isoflop")
            key_NKB, key = jax.random.split(key)
            N,K,B = random_NKB_from_isoflop(iso_flop_budget, key_NKB)
            if reject(N, B, K):
                continue
            else:
                # randomly sample lr, momentum, lr_schedule, optimizer
                key_lr, key = jax.random.split(key)
                lr = jnp.exp(jax.random.uniform(key_lr, minval = jnp.log(min_lr), maxval = jnp.log(max_lr)))
                # round to 5 decimal places
                lr = jnp.round(lr, 5)
                #lr = custom_round(lr, log_base = 1.5)

                key_end_lr, key = jax.random.split(key)
                end_lr = jnp.exp(jax.random.uniform(key_end_lr, minval = jnp.log(min_end_lr), maxval = jnp.log(max_end_lr)))
                end_lr = custom_round(end_lr, log_base = 10)
                
                key_momentum, key = jax.random.split(key)
                oneminusmomentum = jnp.exp(jax.random.uniform(key_momentum, minval = jnp.log(1-min_momentum), maxval = jnp.log(1-max_momentum)))
                oneminusmomentum = custom_round(oneminusmomentum, log_base = 10)
                momentum = 1 - oneminusmomentum
                
                key_lr_schedule, key = jax.random.split(key)
                lr_schedule = random_choice_from_list(lr_schedule_options, key_lr_schedule)
                
                key_optimizer, key = jax.random.split(key)
                optimizer = random_choice_from_list(optimizer_options, key_optimizer)

                key_step_decay_schedule, key = jax.random.split(key)
                step_decay_schedule = random_choice_from_list(step_decay_schedule_options, key_step_decay_schedule)
                # convert step_decay_schedule to a dictionary of lists
                step_decay_schedule = {key: list(value) for key, value in step_decay_schedule.items()}

                new_row = pd.DataFrame({"N": [N],
                                        "B": [B],
                                        "K": [K],
                                        "lr": [lr],
                                        "end_lr": [end_lr],
                                        "momentum": [momentum],
                                        "lr_schedule": [lr_schedule],
                                        "optimizer": [optimizer],
                                        "step_decay_schedule": [json.dumps(step_decay_schedule)]})
                df = pd.concat([df, new_row], axis = 0)
                samples_collected += 1

        df["type"] = "random_isoflop"
        return df
    
    def grid_sample():
        df = pd.DataFrame()

        Ns = array_with_max_min_space(max_N, min_N, space_N, 2)
        # apply actual_N to Ns
       # Ns = [actual_N(N) for N in Ns] 
       # on second thought, don't need to apply actual_N to Ns - only use nqs on the grid points so its okay

        Bs = array_with_max_min_space(max_B, min_B, space_B, 2)
        Ks = array_with_max_min_space(max_K, min_K, space_K, 2)
        lrs = array_with_max_min_space(max_lr, min_lr, space_lr, 10)
        end_lrs = array_with_max_min_space(max_end_lr, min_end_lr, space_end_lr, 10)
        oneminusmomentums = array_with_max_min_space(1-max_momentum, 1-min_momentum, space_momentum, 10)
        momentums = 1 - oneminusmomentums

        for N in Ns:
            for B in Bs:
                for K in Ks:
                    for lr in lrs:
                        for end_lr in end_lrs:
                            for momentum in momentums:
                                for lr_schedule in lr_schedule_options:
                                    for optimizer in optimizer_options:
                                        for step_decay_schedule in step_decay_schedule_options:
                                            step_decay_schedule = {key: list(value) for key, value in step_decay_schedule.items()}
                                            new_row = pd.DataFrame({"N": [N],
                                                "B": [B],
                                                "K": [K],
                                                "lr": [lr],
                                                "end_lr": [end_lr],
                                                "momentum": [momentum],
                                                "lr_schedule": [lr_schedule],
                                                "optimizer": [optimizer],
                                                "step_decay_schedule": [json.dumps(step_decay_schedule)]})
                                            df = pd.concat([df, new_row], axis = 0)
        df["type"] = "grid"
        return df

    key = jax.random.PRNGKey(rand_key)

    df = pd.DataFrame()
    if use_random_isoflops:
        iso_flop_fracs_list = [item for item in isoflop_fracs]
        for iso_flop_frac in iso_flop_fracs_list:
            key_i, key = jax.random.split(key)
            iso_flop_budget = iso_flop_frac * compute_budget
            df_i = random_sample_from_isoflop(iso_flop_budget, n_samples, key_i)
            df = pd.concat([df, df_i], axis = 0)
    if use_grid:
        df_grid = grid_sample()
        df = pd.concat([df, df_grid], axis = 0)
    
    # make sure N, B, K are integers
    df["N"] = df["N"].astype(int)
    df["B"] = df["B"].astype(int)
    df["K"] = df["K"].astype(int)

    df["C"] = 6 * df["N"]/1000 * df["B"]/1000 * df["K"]/1000 * seq_len
    # round C to the nearest thousand
    df["C"] = df["C"].round(-3)
    # if C == 58900000, set C to 59000000
    # be careful of scalar vs array
   # df["C"] = df["C"].replace(58900000.0, 59000000.0)
    #raise ValueError("C: "+str(df["C"].unique()))
    df["T"] = df["K"]

        
    return {"samples": df}