#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;

const int n_l = 5;

class MonteCarloSampleGenerator
{
public:
    MonteCarloSampleGenerator(MatrixXd& weights_old[n_l], MatrixXd& weights_new[n_l],
        MatrixXd& biases_old[n_l], MatrixXd& biases_new[n_l], int no_samples);
    MatrixXd GetSamples();
private:
    MatrixXd sample_matrix;
    int ns;
    MatrixXd w_old[n_l];
    MatrixXd w_new[n_l];
    MatrixXd b_old[n_l];
    MatrixXd b_new[n_l];
    void GenerateSamples();
    MatrixXd NetworkOutput();
    MatrixXd MonteCarloStep();
};


//PYBIND11_MODULE(monte_carlo, m)
//{
//    m.doc() = "pybind example plugin";
//
//}
