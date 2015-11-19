#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    
    
    topology.clear();
    ofLogToFile("output.txt", false);
    
    topology.push_back(2);
    topology.push_back(4);
    topology.push_back(1);
    
    myNet = Net(topology);
    
    
    inputVals.clear();
    targetVals.clear();
    
    
    
    for (int i = 0; i < 500; i++){
        // cout << i << endl;
        ofLog() << i<<":";
        int a = (int) ofRandom(2);
        int b = (int) ofRandom(2);
        
        inputVals.clear();
        inputVals.push_back(a);
        inputVals.push_back(b);
        
        myNet.feedForward(inputVals);
        
        ofLog() <<"input " << a << " " << b;
        
        myNet.getResults(resultVals);
        
        
        for (int i = 0; i < resultVals.size(); i++) {
            ofLog()<< "Net Result: " << resultVals[i];
        }
        
        int result = a ^ b;
        
        targetVals.clear();
        targetVals.push_back(result);
        
        ofLog() << "Target Result: " << result;
        
        myNet.backProp(targetVals);
        
        ofLog() << myNet.getRecentAvgError();
        ofLog() << "________________________";
    }
    
}

//--------------------------------------------------------------
void ofApp::update(){
    
}

//--------------------------------------------------------------
void ofApp::draw(){
    
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    
    
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
