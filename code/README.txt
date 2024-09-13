Code for the analyses and figures reported in:

Martinez, D. R. Q., Rubio, G. F., Bonetti, L., Achyutuni, K. G., Tzovara, A., Knight, R. T., & Vuust, P. (2024). 
Decoding reveals the neural representation of perceived and imagined musical sounds (p. 2023.08.15.553456). 
bioRxiv. https://doi.org/10.1101/2023.08.15.553456

Description

## custom libraries
src/preprocessing.py # Epoching
src/decoding.py # Decoding
src/group_stats.py # Statistics

## Maxwell filtering
APR1a_Maxfilter.py 

## ICA cleaning
APR2a_runICA_cluster.py
APR2b_ICA_submit_to_cluster.py
APR2c_ICA_visual_rejection.py

## MRI segmentation and forward modeling
APR3a_segmentation.py
APR3b_bem.py
APR3c_coregistration.py
APR3d_source_spaces.py
APR3e_forward_model.py

## Event related field analyses
APR4a_ERF_analyses.py
APR4b_ERF_submit_to_cluster.py

## Behavior
APR5a_behavior_and_demographics.R (Fig. 1c, A1 & A2)

## Neural decoding
APR6a_decoding.py
APR6b_decoding_submit_to_cluster.py
APR6c_decoding_plot_imagined_acc_stats.py (Fig. 2 & A3)
APR6d_decoding_plot_imagined_sensor_stats.py (Fig. 3 & 4)
APR6e_decoding_invert_patterns.py
APR6f_decoding_invert_submit_to_cluster.py
APR6g_decoding_patterns_source_stats.py
APR6h_decoding_patterns_source_stats_list_vs_im.py
APR6i_decoding_patterns_source_stats_wrapper.py
APR6j_decoding_patterns_plot_source_stats.py (Fig. 3, 4 & A4)
APR6k_decoding_patterns_plot_source_stats_list_vs_im.py (Fig. 3 & 4)
APR6l_decoding_extract_ROIs.py
APR6m_decoding_extract_ROIs_submit_to_cluster.py
APR6n_decoding_ROI_stats.py
APR6o_decoding_ERF_ROI_stats.py

## Neural accuracies and behavior
APR7a_analyze_neural_accuracies.R (Fig. A6 & A7)