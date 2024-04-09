#include "CNN.h"

void CNN::ForwardPropagation() {

    // Para la propagación para adelante se utilizó la siguiente fórmula:
    // z = w * cap + b || Para conseguir los scores
    // función de activación(z) || Para conseguir la capa próxima

    z1 = w1 * cap_entrada + b1;
    cap_oculta1 = z1.unaryExpr([](double x){ return CNN::Relu(x); });

    z2 = w2 * cap_oculta1 + b2;
    cap_oculta2 = z2.unaryExpr([](double x){ return CNN::Relu(x);});

    z3 = w3 * cap_oculta2 + b3;
    cap_oculta3 = z3.unaryExpr([](double x){ return CNN::Relu(x);});

    z4 = w4 * cap_oculta3 + b4;
    cap_oculta4 = z4.unaryExpr([](double x){ return CNN::Relu(x);});

    z5 = w5 * cap_oculta4 + b5;
    cap_oculta5 = z5.unaryExpr([](double x){ return CNN::Relu(x);});

    z6 = w6 * cap_oculta5 + b6;
    cap_salida = CNN::SoftMax(z6);

}


void CNN::BackPropagation(const Index& index) {

    CNN::ForwardPropagation();

    // Fórmula matemática que he seguido para conseguir el gradiante entre capas:
    // Grad_FP * cap.transpose() = aL/aw = aL/az * az/aw || az/aw = transpuesta de la capa
    // Grad_FP * 1 = aL/ab = aL/az * az/ab || az/ab = 1

    // Forma de sacar la gradiante entre la penúltima capa y capa de salida (SOLO SI SE UTILIZA LA FA softmax)

    VectorXd Grad_FP6 = CNN::Descent_Gradient(z6);
    w6 = w6 - tasa_aprendizaje * (Grad_FP6 * cap_oculta5.transpose());
    b6 = b6 - tasa_aprendizaje * (Grad_FP6 * 1);

    VectorXd Grad_FP5 = (w6.transpose() * Grad_FP6).cwiseProduct(z5.unaryExpr([](double x){ return CNN::Derivada_Relu(x);}));
    w5 = w5 - tasa_aprendizaje * (Grad_FP5 * cap_oculta4.transpose());
    b5 = b5 - tasa_aprendizaje * (Grad_FP5 * 1);

    VectorXd Grad_FP4 = (w5.transpose() * Grad_FP5).cwiseProduct(z4.unaryExpr([](double x){return CNN::Derivada_Relu(x);}));
    w4 = w4 - tasa_aprendizaje * (Grad_FP4 * cap_oculta3.transpose());
    b4 = b4 - tasa_aprendizaje * (Grad_FP4 * 1);

    VectorXd Grad_FP3 = (w4.transpose() * Grad_FP4).cwiseProduct(z3.unaryExpr([](double x){return CNN::Derivada_Relu(x);}));
    w3 = w3 - tasa_aprendizaje * (Grad_FP3 * cap_oculta2.transpose());
    b3 = b3 - tasa_aprendizaje * (Grad_FP3 * 1);

    VectorXd Grad_FP2 = (w3.transpose() * Grad_FP3).cwiseProduct(z2.unaryExpr([](double x){return CNN::Derivada_Relu(x);}));
    w2 = w2 - tasa_aprendizaje * (Grad_FP2 * cap_oculta1.transpose());
    b2 = b2 - tasa_aprendizaje * (Grad_FP2 * 1);

    VectorXd Grad_FP1 = (w2.transpose() * Grad_FP2).cwiseProduct(z1.unaryExpr([](double x){return CNN::Derivada_Relu(x);}));
    w1 = w1 - tasa_aprendizaje * (Grad_FP1 * cap_entrada.transpose());
    b1 = b1 - tasa_aprendizaje * (Grad_FP1 * 1);


    costo = CNN::loss_function(cap_salida, index);

}