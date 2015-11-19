#include "Net.h"


//Net---------------------------------------------------------

double Net::m_recentAvgSmoothFac = 100.0;

Net::Net(){
    
}

Net::Net(vector<int> &topology){
    
    int numLayers = topology.size();
    
    for (int i = 0; i < numLayers; i++) {
        //add layer
        m_layers.push_back(Layer());
        
        int numOutputs;
        
        if (i == topology.size()-1) {
            numOutputs = 0;
        } else {
            numOutputs = topology[i+1];
        }
        
        for (int j = 0; j <= topology[i]; j++ ) {
            //add neuron + 1 bias
            m_layers.back().push_back(Neuron(numOutputs, j));
            cout << "made a neuron" << endl;
        }
        
        //force the bias node output value to 1.0, last node created
        m_layers.back().back().setOutputVal(1.0);
    }
    
}


void Net::feedForward(vector<double> &inputVals) {
    
    // assert(inputVals.size() == m_layers[0].size()-1);
    
    //assign input values into the input neurons
    for (int i = 0; i < inputVals.size(); i++) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    
    //forward propagade
    for (int i = 1; i < m_layers.size(); i++) {
        Layer &prevLayer = m_layers[i-1];
        for (int j = 0; j < m_layers[i].size()-1; j++) {
            m_layers[i][j].feedForward(prevLayer);
        }
    }
    
}

void Net::backProp(vector<double> &targetVals) {
    
    //Calculate overall net error
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    
    for (int i = 0; i < outputLayer.size()-1; i++) {
        double delta = targetVals[i] - outputLayer[i].getOutputVal();
        m_error += delta * delta;
    }
    
    //rms
    m_error /= outputLayer.size()-1;
    m_error = sqrt(m_error);
    
    //recent average measurement
    m_recentAvgError = (m_recentAvgError * m_recentAvgSmoothFac + m_error) / (m_recentAvgSmoothFac + 1.0);
    
    //calculate output layer gradient
    for (int i = 0; i < outputLayer.size() -1; i++) {
        outputLayer[i].calcOutputGradients(targetVals[i]);
    }
    
    //calculate gradients on hidden layers
    for (int i = m_layers.size()-2; i > 0; i--) {
        Layer &hiddenLayer = m_layers[i];
        Layer &nextLayer = m_layers[i+1];
        
        for (int j = 0; j < hiddenLayer.size(); j++) {
            hiddenLayer[j].calcHiddenGradients(nextLayer);
        }
    }
    
    //update connection weights, from output -> first hidden layer
    for (int i = m_layers.size() -1; i > 0; i--) {
        Layer &layer = m_layers[i];
        Layer &prevLayer = m_layers[i-1];
        
        for (int j = 0;j < layer.size()-1; j++) {
            layer[j].updateInputWeights(prevLayer);
        }
    }
    
}

void Net::getResults(vector <double> &resultVals) {
    
    resultVals.clear();
    for (int i = 0;i < m_layers.back().size() -1; i++) {
        resultVals.push_back(m_layers.back()[i].getOutputVal());
    }
    
}

double Net::getRecentAvgError(){
    
    return m_recentAvgError;
    
}

//Neuron---------------------------------------------------------

double Neuron::eta = 0.15;    // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.5;   // momentum, multiplier of last deltaWeight, [0.0..1.0]

Neuron::Neuron(int numOutputs, int myIndex){
    
    for (int i = 0; i < numOutputs; i++) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    
    m_myIndex = myIndex;
    
}

void Neuron::feedForward(Layer &prevLayer){
    
    double sum = 0.0;
    
    //Sum from previous layers output + bias
    for (int i = 0; i < prevLayer.size(); i++) {
        sum += prevLayer[i].getOutputVal() * prevLayer[i].m_outputWeights[m_myIndex].weight;
    }
    
    //activation function
    m_outputVal = transferFunction(sum);
    
}

double Neuron::randomWeight(){
    
    return rand() / double(RAND_MAX);
    
}

void Neuron::setOutputVal(double val) {
    
    m_outputVal = val;
    
}

double Neuron::getOutputVal(){
    
    return m_outputVal;
    
}

double Neuron::transferFunction(double x){
    
    //tanh output range -1...1
    return tanh(x);
    
}

double Neuron::transferFunctionDerivative(double x){
    
    //tanh derivative
    return 1 - x * x;
    
}

void Neuron::calcOutputGradients(double targetVal){
    
    double delta = targetVal - m_outputVal;
    m_gradient = delta * transferFunctionDerivative(m_outputVal);
    
}

void Neuron::calcHiddenGradients(Layer &nextLayer){
    
    double dow = sumDOW(nextLayer);
    m_gradient = dow * transferFunctionDerivative(m_outputVal);
    
}

double Neuron::sumDOW(Layer &nextLayer){
    
    double sum = 0.0;
    for (int i = 0; i < nextLayer.size()-1;i++){
        sum += m_outputWeights[i].weight * nextLayer[i].m_gradient;
    }
    return sum;
    
}

void Neuron::updateInputWeights(Layer &prevLayer) {
    
    //update input weights from the preceding layer
    for (int i = 0; i < prevLayer.size(); i++) {
        
        Neuron &neuron = prevLayer[i];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        
        double newDeltaWeight =
        
        //input magnified by the gradient and train rate
        eta * neuron.getOutputVal() * m_gradient
        //add momentum, fraction of the previous delta weight
        + alpha * oldDeltaWeight;
        
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
    
}

