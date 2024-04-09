#include "CNN.h"

double CNN::loss_function(const Eigen::VectorXd& cap_final, Eigen::Index index) {

    VectorXd F_A = CNN::SoftMax(cap_final);
    double prob = F_A[index];

    return -log(prob + 1e-10); // El

    // Formula: -log(clase correcta del vector final con softmax)

}

VectorXd CNN::Descent_Gradient(const Eigen::VectorXd& ultimo_score) {

    VectorXd SoftMax = CNN::SoftMax(ultimo_score);

    return SoftMax - etiqueta_real;

    // Formula: SoftMax del último score (score entre la penúltima y última capa) - One Hot
}
