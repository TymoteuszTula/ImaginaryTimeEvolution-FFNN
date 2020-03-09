#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

using Eigen::MatrixXd

class MonteCarloSampleGenerator
{
public:
    MonteCarloSampleGenerator(MatrixXd weights_old, MatrixXd weights_new,
        MatrixXd biases_old, MatrixXd biases_new);
    MatrixXd GetSamples();
private:
    MatrixXd sampleMatrix;
    void GenerateSamples();
    MatrixXd NetworkOutput();
    MatrixXd MonteCarloStep();
}


PYBIND11_MODULE(monte_carlo, m)
{
    m.doc() = "pybind example plugin";

}
