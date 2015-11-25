#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    
    
    topology.clear();
    ofLogToFile("output.txt", false);
    
    topology.push_back(2);
    
    topology.push_back(4);
    topology.push_back(4);
    topology.push_back(4);


    topology.push_back(1);
    
    myNet = Net(topology);
    
    
    inputVals.clear();
    targetVals.clear();
    ofEnableAlphaBlending();
    ofDisableAntiAliasing();
    
    
     // cout << i << endl;
     int a = (int) ofRandom(2);
     int b = (int) ofRandom(2);
     
     inputVals.clear();
     inputVals.push_back(a);
     inputVals.push_back(b);
     
     myNet.feedForward(inputVals);
     
    
     myNet.getResults(resultVals);
     
     
    
     
     int result = a ^ b;
     
     targetVals.clear();
     targetVals.push_back(result);
     
    
     myNet.backProp(targetVals);
     
    
}

//--------------------------------------------------------------
void ofApp::update(){
    
    
    
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofBackground(4, 4, 10);
    ofTranslate(100, 100);
    ofNoFill();
    ofSetColor(255, 255, 255);
    
    float grid = 100;
    float dia = 20;
    
    //input
    for (int j = 0; j <= topology.at(0); j++) {
        ofDrawBitmapString(ofToString(myNet.m_layers[0][j].getOutputVal()), 0, j*grid-20);
        ofDrawEllipse(0, j*grid, dia, dia);
        
        ofPushStyle();
        ofSetColor(55, 55, 55);
        for (int k = 0; k < topology.at(1); k++) {
            ofDrawLine(0, j*grid, grid, k*grid);
            }
        ofPopStyle();
    }
    

    
    //hidden
    
    for (int i = 1; i < topology.size()-1; i++) {
        for (int j = 0; j <= topology.at(i); j++) {
            ofDrawBitmapString(ofToString(myNet.m_layers[i][j].getOutputVal()), i*grid, j*grid-20);
            ofDrawEllipse(i*grid, j*grid, dia, dia);
            ofPushStyle();
            ofSetColor(55, 55, 55);
            for (int k =0; k < topology.at(i+1);k++){
                ofDrawLine(i*grid, j*grid, (i+1)*grid, k*grid);
            }
            ofPopStyle();
        }
    }
    
    //output
    
    for (int j = 0; j <= topology.at(topology.size()-1); j++) {
        ofDrawBitmapString(ofToString(myNet.m_layers[topology.size()-1][j].getOutputVal()), (topology.size()-1)*grid,j*grid-20);
        ofDrawEllipse((topology.size()-1)*grid, j*grid, dia, dia);
    }
    
    
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    
    for (int i = 0; i < 20000; i++){
    int a = (int) ofRandom(2);
    int b = (int) ofRandom(2);
    
    inputVals.clear();
    inputVals.push_back(a);
    inputVals.push_back(b);
    
    myNet.feedForward(inputVals);
    
    
    myNet.getResults(resultVals);
    
    
    
    
    int result = a ^ b;
    
    targetVals.clear();
    targetVals.push_back(result);
    
    
    myNet.backProp(targetVals);
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){
    
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){
    
}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){
    
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){
    
}
