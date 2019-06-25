TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

CUDA_LIBS = -lcuda -lcudart -lcublas -lcurand -L/usr/local/cuda-8.0/lib64

LIBS +=  -L/usr/lib \
         -L /home/elian/Documents/app/openalpr/src/build/openalpr/ -lopenalpr \

LIBS += $$CUDA_LIBS

QMAKE_CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic -fopenmp -DDLIB_JPEG_SUPPORT

PKGCONFIG += opencv
CONFIG += link_pkgconfig

SOURCES += main.cpp

HEADERS += \
    plate1.h \
    compared.h

