# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

SIDES <- function(ancova_outcome_arg, ancova_censor_arg, ancova_treatment_arg, cont_covariates, class_covariates, n_cont_covariates, n_class_covariates, project_filename, output_filename) {
    .Call(`_rsides_SIDES`, ancova_outcome_arg, ancova_censor_arg, ancova_treatment_arg, cont_covariates, class_covariates, n_cont_covariates, n_class_covariates, project_filename, output_filename)
}

SIDESAdjP <- function(ancova_outcome_arg, ancova_censor_arg, ancova_treatment_arg, cont_covariates, class_covariates, n_cont_covariates, n_class_covariates, random_seed, project_filename) {
    .Call(`_rsides_SIDESAdjP`, ancova_outcome_arg, ancova_censor_arg, ancova_treatment_arg, cont_covariates, class_covariates, n_cont_covariates, n_class_covariates, random_seed, project_filename)
}

FixedSIDEScreenAdjP <- function(ancova_outcome_arg, ancova_censor_arg, ancova_treatment_arg, cont_covariates, class_covariates, n_cont_covariates, n_class_covariates, random_seed, project_file) {
    .Call(`_rsides_FixedSIDEScreenAdjP`, ancova_outcome_arg, ancova_censor_arg, ancova_treatment_arg, cont_covariates, class_covariates, n_cont_covariates, n_class_covariates, random_seed, project_file)
}

AdaptiveSIDEScreenAdjP <- function(ancova_outcome_arg, ancova_censor_arg, ancova_treatment_arg, cont_covariates, class_covariates, n_cont_covariates, n_class_covariates, random_seed, project_file, output_file) {
    .Call(`_rsides_AdaptiveSIDEScreenAdjP`, ancova_outcome_arg, ancova_censor_arg, ancova_treatment_arg, cont_covariates, class_covariates, n_cont_covariates, n_class_covariates, random_seed, project_file, output_file)
}

Quant <- function(vec_arg, nperc) {
    .Call(`_rsides_Quant`, vec_arg, nperc)
}

