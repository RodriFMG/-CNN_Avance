#include "CNN.h"

// Formula matemática: raíz de 2/(tamaño de la capa de entrada - tamaño de la capa de salida)

// el 2/(tamaño de la capa de entrada - tamaño de la capa de salida) vendría siendo la varianza
// factor xavier = raíz de la varianza.
void CNN::Inicilizar_Xavier() {
    factor_xavier = sqrt( 2.0 / ( static_cast<int>(cap_entrada.size()+n7) ) );
}

void CNN::Dimensiones_Contenedores() {

    // Se definen previamente las dimensiones de los vectores y matrices eigen, ya que, si no se define fallará el programa.
    // Es fundamental definir correctamente las dimensiones para poder realizar las operaciones entre matrices-vectores.
    // (suma, resta y producto son las que se utilizaron)

    // Dimension para el vector de neuronas: (n)
    // Dimension para la matriz de pesos: (n+1,n)
    // Dimension para el vector de sesgos: (n+1)
    // Dimension para el vector de scores: (n+1)

    Inicilizar_Xavier();
    cap_entrada = VectorXd::Zero(64);

    w1 = MatrixXd::Random(n2,cap_entrada.size()) * factor_xavier;
    z1 = VectorXd::Zero(n2);
    b1 = VectorXd::Zero(n2);

    w2 = MatrixXd::Random(n3,n2) * factor_xavier;
    cap_oculta1 = VectorXd::Zero(n2);
    z2 = VectorXd::Zero(n3);
    b2 = VectorXd::Zero(n3);

    w3 = MatrixXd::Random(n4,n3) * factor_xavier;
    cap_oculta2 = VectorXd::Zero(n3);
    z3 = VectorXd::Zero(n4);
    b3 = VectorXd::Zero(n4);

    w4 = MatrixXd::Random(n5,n4) * factor_xavier;
    cap_oculta3 = VectorXd::Zero(n4);
    z4 = VectorXd::Zero(n5);
    b4 = VectorXd::Zero(n5);

    w5 = MatrixXd::Random(n6,n5) * factor_xavier;
    cap_oculta4 = VectorXd::Zero(n5);
    z5 = VectorXd::Zero(n6);
    z5 = VectorXd::Zero(n6);

    w6 = MatrixXd::Random(n7,n6) * factor_xavier;
    cap_oculta5 = VectorXd::Zero(n6);
    z6 = VectorXd::Zero(n7);
    b6 = VectorXd::Zero(n7);

    cap_salida = VectorXd::Zero(n7);

    // VectorXd para el método de entrenamiento: One-Hot
    etiqueta_real = VectorXd::Zero(n7);
}