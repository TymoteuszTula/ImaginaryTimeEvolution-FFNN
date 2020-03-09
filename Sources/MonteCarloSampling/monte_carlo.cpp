#include <eigen3/Eigen/Dense>
#include "monte_carlo.hpp"

using Eigen::MatrixXd;

MonteCarloSampleGenerator::MonteCarloSampleGenerator(
    MatrixXd& weights_old[n_l], MatrixXd& weights_new[n_l],
        MatrixXd& biases_old[n_l], MatrixXd& biases_new[n_l], int no_samples)
{
    ns = no_samples;
    w_old = weights_old;
    w_new = weights_new;
    b_old = biases_old;
    b_new = biases_new;
    sample_matrix = MatrixXd::Zero(no_samples, w_old[0].rows());
}

MatrixXd MonteCarloSampleGenerator::GetSamples()
{
    return sample_matrix;
}
