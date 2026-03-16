


# OWT data
# --------------------------------
# nqs fitted on all isoflops: "nqs_loss" column, from 
#     scale_new/outputs/runs/Paper/2026-01-23-19-07_openwebtext2_pythia_adam_cosine_all_isoflops_nqs/5_resource_allocation/h_samples_with_nqs_loss.csv
# nn on all isoflops: "NN_loss" column, from
#     /mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-19-07_openwebtext2_pythia_adam_cosine_all_isoflops_nqs/5_resource_allocation/to_get_nn_samples.csv
# nqs/nn fitted on all isotokens: "NN_loss" column and "nqs_loss" column, from
#     /mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-19-14_openwebtext2_pythia_adam_cosine_all_isotokens_nqs/6_critical_batch_size/h_samples_with_nn_loss.csv
# chin fitted on all isoflops: "chin_loss" column, from 
#    /mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-19-23_openwebtext2_pythia_adam_cosine_chin_eval/4_loss_estimation/chinchilla_2/eval_df.csv


# join the tables, on "N","B","K"
# divide into train/test, put in a column "split",  when C > 60000000 it's test, otherwise train

# save the processed csv

