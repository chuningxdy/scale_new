# create, look up, and maintain archives
# (the two samples, nn and nqs)


import os
from omegaconf import OmegaConf
import yaml
import pandas as pd
import hydra

import os
from omegaconf import OmegaConf

import json
import datetime


def get_timestamp():
    """
    Returns a formatted timestamp string: yyyy_mm_dd_hh_mm_ss
    Suitable for use in folder names.
    """
    
    # Get current date and time
    now = datetime.datetime.now()
    
    # Format as yyyy_mm_dd_hh_mm_ss
    formatted_timestamp = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
    return formatted_timestamp


def df_row_to_dict(df, row_id):
    if row_id >= len(df):
        raise ValueError("row_id: " + str(row_id) + " is out of range")
    # check if lr_schedule is one of the columns
    dict = df.iloc[row_id].to_dict()

   # if "lr_schedule" in df.columns and df.iloc[row_id]["lr_schedule"] == "optimized":
   #     step_decay_schedule = "ana"
   #     dict.update({"step_decay_schedule": step_decay_schedule})
    if "lr_schedule" in df.columns:
        step_decay_schedule = "na"
        # check in row_id of that column lr_schedule, the value = step
        if df.iloc[row_id]["lr_schedule"] in ["step"]:
            # load json formatted value string to dictionary
            step_decay_schedule = json.loads(df.iloc[row_id]["step_decay_schedule"])
            # check if step_decay_schedule decay_amt is all 1.0 
            # if so, then step_decay_schedule = "na"
            
            if (not ("B_decay_amt" in step_decay_schedule)) and all([x == 1.0 for x in step_decay_schedule["decay_amt"]]):
                step_decay_schedule = "na"
                dict.update({"lr_schedule": "constant"})
            #else:
            #    raise ValueError("step_decay_schedule: " + str(step_decay_schedule) + " is not valid")
        elif df.iloc[row_id]["lr_schedule"] in ["cosine"]:
            step_decay_schedule = json.loads(df.iloc[row_id]["step_decay_schedule"])
            if (not ("B_decay_amt" in step_decay_schedule)) and all([x == 1.0 for x in step_decay_schedule["decay_amt"]]):
                step_decay_schedule = "na"
                dict.update({"lr_schedule": "cosine"})
                #raise ValueError("step_decay_schedule: " + str(step_decay_schedule) + " is not valid for cosine")
        elif step_decay_schedule == "optimized": # this only happens for NQS
            #raise ValueError("optimized lr_schedule not supported in df_row_to_dict")
            if not df.iloc[row_id]["step_decay_schedule"] == "na":
                step_decay_schedule  = json.loads(df.iloc[row_id]["step_decay_schedule"])
                if (not ("B_decay_amt" in step_decay_schedule)) and all([x == 1.0 for x in step_decay_schedule["decay_amt"]]):
                    step_decay_schedule = "na"
                    dict.update({"lr_schedule": "constant"})

        # update the entry "step_decay_schedule" with the dictionary
        #raise ValueError("step_decay_schedule before update:", step_decay_schedule)
        dict.update({"step_decay_schedule": step_decay_schedule})
        print("step_decay_schedule:", step_decay_schedule)
    return dict



def df_row_to_dict_old(df, row_id):
    if row_id >= len(df):
        raise ValueError("row_id: " + str(row_id) + " is out of range")
    # check if lr_schedule is one of the columns
    dict = df.iloc[row_id].to_dict()
    
    if "lr_schedule" in df.columns:
        step_decay_schedule = "na"
        # check in row_id of that column lr_schedule, the value = step
        if df.iloc[row_id]["lr_schedule"] in ["step","cosine"]:
            # load json formatted value string to dictionary
            step_decay_schedule = json.loads(df.iloc[row_id]["step_decay_schedule"])
            # check if step_decay_schedule decay_amt is all 1.0 
            # if so, then step_decay_schedule = "na"
            if all([x == 1.0 for x in step_decay_schedule["decay_amt"]]):
                step_decay_schedule = "na"
                dict.update({"lr_schedule": "constant"})
            else:
                raise ValueError("step_decay_schedule: " + str(step_decay_schedule) + " is not valid")
        # update the entry "step_decay_schedule" with the dictionary
        dict.update({"step_decay_schedule": step_decay_schedule})
    return dict


def update_default_option(parent_cfg, option_cfg, option_name):
    for d in parent_cfg["defaults"]:
        if option_name in d:
            #print("Updating", option_name)
            parent_cfg[option_name] = option_cfg
            #print(parent_cfg[option_name])
            # remove d from the list of defaults
            parent_cfg["defaults"].remove(d)
            break
    # if the default list contains nothing but maybe "_self_" then remove it
    pop_defaults_1 = len(parent_cfg["defaults"]) == 1 and "_self_" in parent_cfg["defaults"]
    pop_defaults_2 = len(parent_cfg["defaults"]) == 0
    if pop_defaults_1 or pop_defaults_2:
        parent_cfg.pop("defaults")

    return parent_cfg


def hdict_to_hcfg_nqs(hdict):
    h_path = "conf/h/hypers_cfg.yaml"
    h_cfg = OmegaConf.load(h_path)
    
    # update h_cfg with hdict
    h_cfg.update(hdict)
    h_cfg.pop("defaults")
    raise ValueError("h_cfg: " + str(h_cfg))
    return h_cfg

def hdict_to_hcfg(hdict):
    """
    hdict is a dictionary with keys: N, B, K, lr, end_lr, momentum, lr_schedule, optimizer
    create an OmegaConf object h with the information in hdict
    """
    h_path = "conf/h/hypers_cfg.yaml"
    h_cfg = OmegaConf.load(h_path)
    
    # update h_cfg with hdict
    h_cfg.update(hdict)

    if hdict["lr_schedule"] in ["step", "constant"]:
        
        h_cfg.pop("lr_schedule")
        h_cfg.pop("optimizer")

        # load the default configuration
        lr_schedule_path = "conf/h/optimizer/learning_rate/" + hdict["lr_schedule"] + ".yaml"
        lr_schedule_cfg = OmegaConf.load(lr_schedule_path)

        optimizer_path = "conf/h/optimizer/" + hdict["optimizer"] + ".yaml"
        optimizer_cfg = OmegaConf.load(optimizer_path)

        update_default_option(optimizer_cfg, lr_schedule_cfg, "learning_rate")
        update_default_option(h_cfg, optimizer_cfg, "optimizer")

        if hdict["lr_schedule"] == "step":
            step_decay_schedule = hdict["step_decay_schedule"]
            h_cfg.optimizer.learning_rate.boundaries_and_scales.decay_schedule = step_decay_schedule

        h_cfg.lr_schedule = hdict["lr_schedule"]
    #else:
    #    raise ValueError("optimized lr_schedule not available for NN")
    #raise ValueError('h_cfg:', h_cfg)
    return h_cfg

def nndict_to_nncfg(nn_dict):
    """
    nn_dict is a dictionary with keys: data, model, loss
    create an OmegaConf object nn with the information in nn_dict
    """
    nn_path = "conf/neural_net/nn_cfg.yaml"
    nn_cfg = OmegaConf.load(nn_path)

    model_path = "conf/neural_net/model/" + nn_dict["model"] + ".yaml"
    model_cfg = OmegaConf.load(model_path)
    loss_path = "conf/neural_net/loss/" + nn_dict["loss"] + ".yaml"
    loss_cfg = OmegaConf.load(loss_path)
    data_path = "conf/neural_net/data/" + nn_dict["data"] + ".yaml"
    data_cfg = OmegaConf.load(data_path)

    update_default_option(nn_cfg, model_cfg, "model")
    update_default_option(nn_cfg, loss_cfg, "loss")
    update_default_option(nn_cfg, data_cfg, "data")

    return nn_cfg

def nqsdict_to_nqscfg(nqs_dict):

    nqs_path = "conf/nqs/nqs_cfg.yaml"
    nqs_cfg = OmegaConf.load(nqs_path)
    
    nqs_cfg.update(nqs_dict)

    return nqs_cfg

def mhdict_to_cfg(mdict, hdict):

    # if mdict has key "data", then it is a nn configuration
    if "data" in mdict:
        m_cfg = nndict_to_nncfg(mdict)
        h_cfg = hdict_to_hcfg(hdict)
    if "a" in mdict or "e_irr" in mdict: # nqs
        m_cfg = nqsdict_to_nqscfg(mdict)
        h_cfg = hdict_to_hcfg(hdict)
        #raise ValueError('h_cfg: ' + str(h_cfg))
    if "lr_schedule" not in h_cfg:
        raise ValueError("lr_schedule not in hdict - hdict_to_hcfg")
    m_cfg.update({"h": h_cfg})
    return m_cfg
    

def create_lookup_entry_for_archive(mdict, hdict, archive_file):
    """
    create a new entry to look up existing records in the archive
    """
    # merge mdict and hdict
    mh_dict = {**mdict, **hdict}
    if mh_dict["lr_schedule"] in ["step","cosine"] and not mh_dict["step_decay_schedule"] == "na":
        mh_dict["step_decay_schedule"] = json.dumps(mh_dict["step_decay_schedule"])
    elif mh_dict["lr_schedule"] in ["step","cosine"] and mh_dict["step_decay_schedule"] == "na":
        mh_dict["step_decay_schedule"] = "na"
    elif mh_dict["lr_schedule"] == "constant":
        mh_dict["step_decay_schedule"] = "na"
    lookup_entry = pd.DataFrame([mh_dict])
    # only keep the columns in the archive
    archive = pd.read_csv(archive_file)
    keep_columns = list(archive.columns)
    # remove run_id and path from keep_columns
    keep_columns.remove("run_id")
    keep_columns.remove("path")
    lookup_entry = lookup_entry[keep_columns]
    # check if list(lookup_entry.columns) contains all columns in keep_columns
    if not set(lookup_entry.columns) == set(keep_columns):
        raise ValueError("Columns in lookup_entry do not match columns in archive: \n" 
                         + "lookup_entry: " + str(lookup_entry.columns) + "\n"
                            + "needed for archive: " + str(keep_columns))
    
    if "m_a" in lookup_entry.columns: # i.e. is nqs
        nqs_cols = ["a", "b", "m_a", "m_b", "eps", "sigma"]
        for col in nqs_cols:
            # round to 6 decimal places
            lookup_entry[col] = lookup_entry[col].round(6)
    #raise ValueError("lookup_entry: " + str(lookup_entry))
    return lookup_entry


def create_archive(archive_file):
    """
    create an archive file with the following columns:
    run_id, N, B, K, lr, end_lr, momentum, lr_schedule, optimizer
    return a message.
    """

    if not os.path.exists(archive_file):
        if "nn" in archive_file:
            cols = ["data","model","loss"]
        elif "nqs" in archive_file:
            cols = ["a","b","m_a","m_b","eps","sigma"]
        else:
            # raise an error - invalid archive file
            raise ValueError("Invalid archive file")
        h_columns=["run_id", "N", "B", "K", "lr", 
                   "end_lr", "momentum", "lr_schedule", "optimizer",
                   "step_decay_schedule", "path"]

        cols = cols + h_columns
        df = pd.DataFrame(columns=cols)
        # make sure the path exists
        if not os.path.exists(os.path.dirname(archive_file)):
            os.makedirs(os.path.dirname(archive_file))
        df.to_csv(archive_file, index=False)
        # create a folder called runs
        runs_folder = os.path.join(os.path.dirname(archive_file), "runs")
        if not os.path.exists(runs_folder):
            os.makedirs(runs_folder)
        msg = "Archive file created"
    
    else:
        msg = "Archive file already exists"
        
    return msg

def look_up_archive(lookup_entry, archive_file):
        
        archive = pd.read_csv(archive_file)
        # check if the record already exists
        # merge new_df with df_archive, right join on all keys in df_archive
        # except run_id, path
        
        join_keys = list(archive.columns)
        join_keys.remove("run_id")
        join_keys.remove("path")


        # in lookup_entry, change the data type of all columns to 
        # match the data type of the archive
        #for col in join_keys:
         #   if col in lookup_entry.columns:
          #      lookup_entry[col] = lookup_entry[col].astype(archive[col].dtype)
        
        # print the data type of columns "step_decay_schedule" 
       # archive.to_csv("archive.csv", index=False)
        #archive = pd.read_csv("archive.csv")
       # lookup_entry.to_csv("lookup_entry.csv", index=False)
        #lookup_entry = pd.read_csv("lookup_entry.csv")
      #  print("archive step_decay_schedule:", archive["step_decay_schedule"].dtype)
      #  print("lookup_entry step_decay_schedule:", lookup_entry["step_decay_schedule"].dtype)
        merge_attempt = pd.merge(lookup_entry, archive, how="left", on=join_keys)
        # check if run_id of merge_attempt is not null
       # merge_attempt.to_csv("merge_attempt_master.csv", index=False)
        #raise ValueError("merge_attempt: " + str(merge_attempt))    
        if merge_attempt["run_id"].isnull().all():
            # save lookup_entry as a csv file 
            lookup_entry.to_csv("lookup_entry.csv", index=False)
            archive.to_csv("archive.csv", index=False)
            #raise ValueError("No matching record: " + str(merge_attempt))
            return None, None
        else:
            run_id = merge_attempt["run_id"].values[0]
            path = merge_attempt["path"].values[0]
            path_prefix = '/mfs1/u/chuning/scale/'
            # check if path starts with path_prefix
            if not str(path).startswith(path_prefix):
                path = os.path.join(path_prefix, str(path).lstrip('/'))
            #raise ValueError("Run ID: " + str(run_id) + "\n" + "Path: " + str(path))
            return run_id, path
        
def get_run_id_and_path_archive(lookup_entry, archive_file):
        df_archive = pd.read_csv(archive_file)
        # check if df_archive is empty
        if df_archive.empty:
            run_id = 0
        else:
            #run_id = df_archive["run_id"].max() + 1
            run_id = "Run_" + get_timestamp()
        archive_entry = lookup_entry.copy()
        archive_entry["run_id"] = run_id
        # path = directory of the archive file + "runs"/run_id
        run_path = os.path.join(os.path.dirname(archive_file), "runs")
        path = os.path.join(run_path, str(run_id)) + "/"
        path_prefix = '/mfs1/u/chuning/scale/'
        # check if path starts with path_prefix
        if not str(path).startswith(path_prefix):
            path = os.path.join(path_prefix, str(path).lstrip('/'))
        archive_entry["path"] = path
        return archive_entry

def save_entry_to_archive(archive_entry, archive_file):
    '''
    '''

    df_archive = pd.read_csv(archive_file)

    # concatenate new_df to df_archive
    df_archive = pd.concat([df_archive, archive_entry])
    # save df_archive as a csv file
    df_archive.to_csv(archive_file,
                        index=False)
    return None

def archive_wrapper(func, calc = True):
    '''
    func takes arguments: cfg, path, and
    outputs func_out (a dictionary)
    wrapped_func takes arguments: cfg, hdict, archive_file, and 
    outputs a message and func_out
    '''

        
    def wrapped_func(mdict, hdict, archive_file):




        hdict = hdict.copy()
        # look up the archive
        # if the key cfg.h.lr_schedule does not exist, raise an error

        # if nn then set lr schedule to constant
        #if "data" in mdict:
        #    hdict["lr_schedule"] = "constant"

        if "a" in mdict or "e_irr" in mdict: # nqs
            if hdict["optimizer"] == "adamw":
                hdict["lr"] = 1.999 * hdict["lr"]/0.001 #1.999
                #1.999 #dict["lr"] * 10.0
            hdict["optimizer"] = "sgd"
            
            if hdict["lr_schedule"] == "cosine":
                hdict["lr_schedule"] = "constant"
                #raise ValueError("updated hdict to: ", hdict["lr_schedule"])
            #raise ValueError("hdict after nqs update: " + str(hdict))

            
        if "a" in mdict or "e_irr" in mdict: # nqs
            run_id = None
            path = None
        else:
            lookup_entry = create_lookup_entry_for_archive(mdict, hdict, archive_file)
            #if hdict["B"] == 6 and hdict["K"] == 320000 and hdict["lr_schedule"] == "step":
            #    raise ValueError("lookup_entry: " + str(lookup_entry))
            #print("lookup_entry: " + str(lookup_entry))
            #raise ValueError("lookup_entry: " + str(lookup_entry))
            run_id, path = look_up_archive(lookup_entry, archive_file)
            #raise ValueError(run_id, path)
        
        if run_id is None:
            #raise ValueError(mdict, hdict)
            #if not "a" in mdict:
            
            if "a" in mdict or "e_irr" in mdict: # nqs
                archive = None
                archive_entry = None
                path = "./outputs/temp/nqs/"

            else:
                #    raise ValueError("Running new neural net: " + str(lookup_entry))
                archive = pd.read_csv(archive_file)
                #archive_172 = archive[archive["run_id"] == 172]
                #raise ValueError("Archive file is empty. Run ID: " + str(run_id) + "\n" + str(archive_172) + "\n" + str(lookup_entry))
                # get the run_id and path
                archive_entry = get_run_id_and_path_archive(lookup_entry, archive_file)
                
                # run the function with the path
                path = archive_entry["path"].values[0]
            
            # if the path does not exist, create the directory
            if not os.path.exists(path):
                os.makedirs(path)
                

                # in error msg. print hdict and mdict
                
            with open(path + "hdict.yaml", "w") as f:
                yaml.dump(hdict, f)
            with open(path + "mdict.yaml", "w") as f:
                yaml.dump(mdict, f)
   
            # create a folder out
            func_out_folder_path = path + "out/"
            if not os.path.exists(func_out_folder_path):
                os.makedirs(func_out_folder_path)

            # if model is in dict and model is in the llm models list, then run the function
            # on mdict and hdict

            if "a" in mdict or "e_irr" in mdict: #NQS
                #raise ValueError("NQS not supported yet")
                #raise ValueError(hdict)
                cfg = mhdict_to_cfg(mdict, hdict)
                with open(path + "cfg.yaml", "w") as f:
                    OmegaConf.save(cfg, f)
                #raise ValueError("cfg:" + str(cfg))
            
                func_out = func(cfg, path)

            elif "model" in mdict: # NN
                if calc:
                    func_out = func(mdict, hdict, path)
                else:
                    # create empty dataframes for func_out
                    func_out = {}

            else:
                raise ValueError("Invalid mdict type")
            # loop thru the func_output dict,
            for key, value in func_out.items():
                value.to_csv(func_out_folder_path + key + ".csv", index=False)

            if calc:
                if "a" in mdict:
                    msg = "this is nqs, not saved"
                if "e_irr" in mdict:
                    msg = "this is nqs, not saved"
                else:
                    # save the entry to the archive
                    save_entry_to_archive(archive_entry, archive_file)
                    msg = "Run Completed. Saved to" + path


            else:
                msg = "Run not found in archive. Skipping calculation."
        else:
           # raise ValueError("Run already exists. Run ID: " + str(run_id))
            func_out_folder_path = path + "out/"
            msg = "Run already exists. Run ID: " + str(run_id)
            # read every file in the func_out folder, and add it to func_out
            func_out = {}
            for file in os.listdir(func_out_folder_path):
                key = file.split(".")[0]
                print("Loading file:", func_out_folder_path + file)
                value = pd.read_csv(func_out_folder_path + file)
                func_out[key] = value
        
        return msg, func_out
    
    return wrapped_func


if __name__ == "__main__":





    hdict = {"N": 250000, "B": 5, "K": 7, "lr": 0.01, "end_lr": 0.001, "momentum": 0.9, 
            "lr_schedule": "constant", "optimizer": "sgd", "step_decay_schedule": "na"}
    

    #raise ValueError(remove_embedding_params(hdict, 3000))
    
    h = hdict_to_hcfg(hdict)
    h_path = "./outputs/tst/h.yaml"
    with open(h_path, "w") as f:
        OmegaConf.save(h, f)

    # load a configuration file in /conf/nqs/nqs_cfg.yaml
    cfg_path = "./conf/neural_net/cifar5m_simplecnn.yaml"# "./conf/nqs/nqs_cfg.yaml"
    #cfg_path = "./conf/nqs/nqs_cfg.yaml"
    nn_dict = {"data": "infimnist", "model": "simplecnn", "loss": "condcrossent"}
    nqs_dict = {"a": 1.5, "b": 2, "m_a": 2, "m_b": 2, "eps": 0.1, "sigma": 0.1}
    
    # create an archive file
    #archive_file = "./outputs/tst/nna/archive_nn.csv"
    archive_file = "./outputs/tst/nqsa/archive_nqs.csv"
    def nn_func(cfg, path):
        # save a text file in the path
        with open(path + "test.txt", "w") as f:
            f.write("Hello World")
        # create a dataframe from cfg
        df1 = pd.DataFrame({"N": [1]})
        df2 = pd.DataFrame({"B": [2]})
        return {"df1": df1, "df2": df2}
    
    # create an archive file
    print(create_archive(archive_file))
    # run the function with the archive wrapper
    print(archive_wrapper(nn_func, incl_embedding_params=False,
                          vocab_size=3000)(nqs_dict, hdict, archive_file))