#pragma once

#include <vector>
#include <iostream>
#include <cmath>


using namespace std;

struct Connection {
    double weight;
    double deltaWeight;
};


class Neuron;

typedef vector<Neuron> Layer;


//Net---------------------------------------------------------
class Net {
    
public:
    Net();
    Net(vector<int>&);
    void feedForward(vector<double>&);
    void backProp(vector<double>&);
    void getResults(vector<double>&);
    double getRecentAvgError();
    
//private:
    //[layerNum][neuronNum]
    vector<Layer> m_layers;
    double m_error;
    double m_recentAvgError;
    static double m_recentAvgSmoothFac;
};

//Neuron---------------------------------------------------------
class Neuron {
    
public:
    Neuron(int,int);
    void feedForward(Layer&);
    void setOutputVal(double);
    double getOutputVal();
    void calcOutputGradients(double);
    void calcHiddenGradients(Layer&);
    void updateInputWeights(Layer&);
    
//private:
    static double eta; //[0.0...1.0] overal net training rate
    static double alpha; //[0.0...n]multiplier of last weight change, momentum
    double transferFunction(double);
    double transferFunctionDerivative(double);
    double m_outputVal;
    vector<Connection> m_outputWeights;
    double randomWeight();
    int m_myIndex;
    double m_gradient;
    double sumDOW(Layer&);
};




