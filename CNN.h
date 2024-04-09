//
// Created by RODRIGO on 4/01/2024.
//

#ifndef CNN_CNN_H
#define CNN_CNN_H

#include <iostream>

// Eigen
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <vector>
#include <cmath>
#include <fstream>

using namespace std;
using namespace Eigen;
using namespace cv;

class CNN{
private:

    // Filtros
    Matrix<int,3,3> Sobel_Horiz;
    Matrix<int,3,3> Sobel_Verti;

    Matrix<int,3,3> Gabor_Horiz;
    Matrix<int,3,3> Gabor_Verti;

    Matrix<int,3,3> Harriz_Esquinas;
    Matrix<int,3,3> Suavizado;

    // Hiperparámetro
    double tasa_aprendizaje = 0.001;

    // Dimensiones generales de los vectores-matrices
    int n2,n3,n4,n5,n6,n7; // Cada "n", es el tamaño de cada capa, no hay n1 porque la capa de entrada siempre será
    // igual a la matriz del rgb (no necesita definirse previamente).

    // Capas
    VectorXd cap_entrada, cap_oculta1, cap_oculta2, cap_oculta3, cap_oculta4, cap_oculta5 ,cap_salida;

    // Sesgos
    VectorXd b1, b2, b3, b4, b5, b6;

    // Scores
    VectorXd z1, z2, z3, z4, z5, z6;

    // Enlaces
    MatrixXd w1, w2, w3, w4, w5, w6;

    // Factor para inicializar las matrices de enlaces
    double factor_xavier{};

    // Contenedor para la convolución, función de activación y pooling
    vector<vector<Vector3i>> imagen;

    // Imagen (Uso del opencv)
    // vector<pair<cv::Mat,Index>> img;

    // One Hot
    VectorXd etiqueta_real;

    // Valor de costo
    double costo{};


public:

    void Establecer_Filtros();

    // Capa convolucional
    void MatToVector(const pair<cv::Mat, Index>& date);
    static vector<vector<Vector3i>> Convolucion_Filtro(const vector<vector<Vector3i>>& mtz_img, Matrix<int, 3, 3> filtro);
    static vector<vector<Vector3i>> max_pooling(const vector<vector<Vector3i>>& mtz_img);

    // Inicialización de las capas (fully connected)
    void Inicilizar_Xavier();
    void Dimensiones_Contenedores();

    // Función de activación
    static double Relu(double x);
    static VectorXd SoftMax(const VectorXd& x);

    // Derivadas de las funciones de activación
    static double Derivada_Relu(double x);

    // Función de coste - Descenso del gradiante
    [[nodiscard]] static double loss_function(const VectorXd& cap_final, Index index);
    [[nodiscard]] VectorXd Descent_Gradient(const VectorXd& ultimo_score);

    // Propagación
    void ForwardPropagation();
    void BackPropagation(const Index& index);

    // Sacando los datos

    // En esta función tengo que tener la imagen, y asegurar que todas las imagenes tengan las mismas filas y columnas.
    void Normalizar_Dimenciones_Espaciales(const Index& index);
    void Datos_Img(const string& cifar_10_train);

    // Entrenamiento
    void Entrenamiento(const vector<vector<Vector3i>>& img_filtrada, const Index& etiqueta);

    // Constructor - Destructor
    CNN(const string& cifar_10_train, const string& cifar_10_test, int n2_, int n3_, int n4_, int n5_, int n6_, int n7_,int file);

    // Resultados
    static pair<cv::Mat, Index> Sacar_img_prediccion(const string& cifar_10_test);
    void Prediccion(const string& cifar_10_test);
    void Resultados(const string& cifar_10_test);





};

#endif //CNN_CNN_H
