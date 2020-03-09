#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using namespace std;

int main()
{
   MatrixXd w_new[5];
   MatrixXd w_old[5];
   MatrixXd b_new[5];
   MatrixXd b_old[5];
   
   int n_v[6] = {10, 5, 50, 30, 3, 1}; 
   
   for (int i = 0; i != 4; ++i)
   {
        w_new[i] = MatrixXd::Random(n_v[i], n_v[i+1] );
        w_old[i] = MatrixXd::Random(n_v[i], n_v[i+1] );
        b_new[i] = MatrixXd::Random(1, n_v[i+1]);
        b_old[i] = MatrixXd::Random(1, n_v[i+1]);
   }
   
   MatrixXd initial_state = MatrixXd::Random(1, n_v[0])
   
   MatrixXd nn_out(MatrixXd input_state)
   {
        MatrixXd output = 
   }
   
   
}
