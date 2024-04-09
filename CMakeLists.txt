cmake_minimum_required(VERSION 3.26)
project(CNN)
set(CMAKE_CXX_STANDARD 23)

# Eigen
set(EIGEN_DIR "C:/Users/RODRIGO/CNN_NV/cmake-build-debug/eigen-3.4.0")
if(NOT EXISTS ${EIGEN_DIR})
    message(FATAL_ERROR "Please check that the set Eigen directory is valid!")
endif()
include_directories(${EIGEN_DIR})

# Ejecutable
add_executable(CNN main.cpp
        Matrices_Filtros.cpp
        Capa_Convolucional.cpp
        Derivadas_Funcion_Activacion.cpp
        Funcion_Activacion.cpp
        Fully_Connected.cpp
        Descenso_Gradiante.cpp
        Propagacion.cpp
        Datos_BDD.cpp
        Entrenamiento.cpp
        Convolucion.cpp
        Constructor.cpp
        Resultados.cpp)

# OpenCV
set(ENV{OPENCV_DIR} "C:\\tools\\opencv\\build")
FIND_PACKAGE(OpenCV REQUIRED)
set(OpenCV_FOUND TRUE)
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
target_link_libraries(CNN ${OpenCV_LIBS})