//This program creates a simple 2 layer multilayer perceptron for playing around
//with handwritten digits MNIST data
//the weights are tuned using backpropagation algoritm with momentum
//The number of hidden nodes can be modified easily in the code,
//while adding more layers require more work
//For a real version convolutional networks, deeper networks etc should be implemented,
//mainly this program was made for myself to remind me of how to code with c++

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <math.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;

//because of stack overflow issues I define things here and most of the data are not anymore members of classes
//some of the old definitions are still in the file since I have been using this program to remind myself
//of how to code c++

#define NUMCOLS 783 //number of data columns
#define dataLen 40000 // number of datapoints taken, test data has 42 000 entries
#define num_hid_nodes 30 // number of hidden nodes
const int trainepochs=400; //maximum number of epoch
double wantederror=1500.0; //stop the propagation when the error is max this
double init_learningrate=0.0002;
double weightscale=0.02; //initially the weights are ramdomized and scaled by this
double initmomentum=0.9; //the momentum term for backpropagation

double momentum; // 


struct Digitdata {
    int label;
    double pixels[NUMCOLS+1];
};


Digitdata data[dataLen];
double weights_1st_layer[NUMCOLS+1][num_hid_nodes];
double weights_2nd_layer[num_hid_nodes+1][10];
double weights_1st_layer_mom[NUMCOLS+1][num_hid_nodes];
double weights_2nd_layer_mom[num_hid_nodes+1][10];
double r_it[dataLen][10];
double hiddennodes[dataLen][num_hid_nodes+1];
double y_out[dataLen][10];


//function for reading input
 //Digitdata * Readinput(int dataLen){
void Readinput(ifstream &traindata){

	string line,cell;
	stringstream lineStream;
	int label;

	for (int i=0; i<dataLen;++i) {

		lineStream.str("");	//clear stringstream
		lineStream.clear(); // Clear state flags.
		getline (traindata,line); //read a row
		lineStream << line;		//convert to stringstream
		
		getline(lineStream, cell, ',');	//get the first value
		data[i].label=stoi(cell);			//which is the label

		for (int j=0; j<NUMCOLS;++j) {
	
			getline(lineStream, cell, ',');	//get one cell	
			
			//data is given as a pixel grayness values, set pixels values to 0/1
			//=black/white
			/*if(stod(cell)>50){
			data[i].pixels[j]=1.0;
			} else {
			data[i].pixels[j]=0.0;
			}*/
			//option 2:
			data[i].pixels[j]=stod(cell)/255.0; //convert to double and normalize
			
		} //over columns
		data[i].pixels[NUMCOLS]=1.0; // the constant input node
	} //over rows
	
	//construct r_it for backpropagation
	//r_i^t=1 if the label(t)=1, 0 otherwise
	//also reset r_i from previous dataset 
	memset(r_it, 0, sizeof(r_it[0][0]) * dataLen * 10);	
	
    for (int i = 0; i < dataLen; ++i){
    	
    	label=data[i].label;
    	r_it[i][label]=1.0;
    }
//	return inputdata;
}

//Class for the network
class MLP {
    //int num_hid_nodes;
    //int dataLen;
    //Digitdata *inputdata;
    //double **weights_1st_layer;
    //double **weights_2nd_layer;
    //double **weights_1st_layer_mom;
    //double **weights_2nd_layer_mom;
    //double **r_it;


  public:
  
	//double **hiddennodes;
    //double **y_out;
    
    MLP();//(int x, Digitdata* data, int dataLeninput, double weightscale);
    ~MLP();
    void Output(int dataind);
    void Epoch (int start_ind,int end_ind);
    double Error_out (int start_ind, int end_ind);
    void Backpropagate (double learningrate,double momentum, int trainLen);
};

//Constructor. Initialize the weights randomly
MLP::MLP(){//(int x, Digitdata* data, int dataLeninput, double weightscale){

	int label;
	//inputdata=data;
	/* All the commented things were for the dynamically allocated version
	
	num_hid_nodes=x;
	inputdata=data;
	dataLen=dataLeninput;
	
	//The first layer is weights from inputs to hidden nodes
	//The second layer is weights from hidden layers to outputs
	//add a weight from constant=1 input and hidden node
	//format: [from][to]
	//initialize the momentum matrizes for backpropagation to 0
	weights_1st_layer = new double*[NUMCOLS+1];
	weights_2nd_layer = new double*[num_hid_nodes+1];
	weights_1st_layer_mom = new double*[NUMCOLS+1];
	weights_2nd_layer_mom = new double*[num_hid_nodes+1];
	
  	for (int i = 0; i < NUMCOLS+1; ++i)  	
    	weights_1st_layer[i] = new double[num_hid_nodes];
    
    for (int i = 0; i < num_hid_nodes+1; ++i)  	
    	weights_2nd_layer[i] = new double[10];
    	
      for (int i = 0; i < NUMCOLS+1; ++i)  	
    	weights_1st_layer_mom[i] = new double[num_hid_nodes]();
    
    for (int i = 0; i < num_hid_nodes+1; ++i)  	
    	weights_2nd_layer_mom[i] = new double[10]();
    	
    //record outputvalues and the hiddennodes values
    //format[traindata][output]
    hiddennodes = new double*[dataLen];
    y_out = new double*[dataLen];
    r_it = new double*[dataLen];
    
  	for (int i = 0; i < dataLen; ++i)  	
    	hiddennodes[i] = new double[num_hid_nodes+1];    
    	
  	for (int i = 0; i < dataLen; ++i)  	
    	y_out[i] = new double[10]; 
    	
    	
    	
    //allocate r_i^t for backpropagation algorithm
    for (int i = 0; i < dataLen; ++i)  	
    	r_it[i] = new double[10]();     
    	*/	
    
    //initialize with random weights 
    random_device rd;  //Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    uniform_real_distribution<> dis(-weightscale, weightscale); //range for random numbers

	//+1 here is for the constant node/input (whose value is 1)
	for (int i = 0; i < NUMCOLS+1; ++i){
		for (int j = 0; j < num_hid_nodes; ++j){
			weights_1st_layer[i][j]=dis(gen);
		}
	}
	for (int i = 0; i < num_hid_nodes+1; ++i){
		for (int j = 0; j < 10; ++j){
			weights_2nd_layer[i][j]=dis(gen);
		}
	}    
}


//destructor, no need really for this in the non-dynamically allocated version

MLP::~MLP(){
/*
	for (int i = 0; i < NUMCOLS+1; ++i){
		delete [] weights_1st_layer[i];
  		delete [] weights_1st_layer;
  	}
  		
  	  for (int i = 0; i < num_hid_nodes+1; ++i){
    	delete [] weights_2nd_layer[i];
  		delete [] weights_2nd_layer;
  		  delete [] weights_2nd_layer_mom[i];
  		delete [] weights_2nd_layer_mom;
  	}
  		
  	for (int i = 0; i < dataLen; ++i){
    	delete [] hiddennodes[i];
  		delete [] hiddennodes;
  		delete [] y_out[i];
  		delete [] y_out;
  	}*/
} 

//produce output from the network with the current weights
void MLP::Output (int dataind) {
	
	//calculate the hidden nodes sigmoid values
	for (int i = 0; i < num_hid_nodes; ++i){
	
		double tempnum=0;

		for (int j = 0; j < NUMCOLS; ++j){
		
			tempnum+=weights_1st_layer[j][i]*data[dataind].pixels[j];
		}
		//map the value with sigmoid function
		hiddennodes[dataind][i]=1.0/(1.0+exp(-tempnum));
		hiddennodes[dataind][num_hid_nodes]=1.0; //constant node
		
	}
	
	//calculate the output values
	double softmaxes[10] = {0};
	double totsum=0;
	for (int i = 0; i < 10; ++i){
	
		double tempnum=0;

		for (int j = 0; j < num_hid_nodes+1; ++j){
		
			tempnum+=weights_2nd_layer[j][i]*hiddennodes[dataind][j];
		}
		
		//softmax		
		y_out[dataind][i]=exp(tempnum);
		totsum+=y_out[dataind][i];
	}
	
	//normalize softmaxes
	for (int i = 0; i < 10; ++i)
	y_out[dataind][i]=y_out[dataind][i]/totsum;	
}

void MLP::Epoch (int start_ind,int end_ind) {
for(int i=start_ind;i<end_ind;++i)
	Output(i);
}


double MLP::Error_out (int start_ind, int end_ind) {

	double error=0;
	int label;
	
	//count the total error (cross entropy) over the data (images) and outputs
	// sum r_i log y_i, where r_i=1 for i which is the label of the corresponding image (0-9)
	//other r_j=1 (j \neq i), thus:
	for(int i=start_ind;i<end_ind;++i){ //for images
		
		label=data[i].label;
		error+=-log(y_out[i][label]);
	}
	
	return error;
}

//calculate the learning rate just the simple way. Here momentum or adjusting learning rate
//etc could be used and fairly simply implemented.
void MLP::Backpropagate (double learningrate, double momentum,int trainLen) {
	
	double tempnum;
	double tempnum2;
	double update;
	
	//Update the first layer weights
	for (int h = 0; h < num_hid_nodes+1; ++h){
		for (int j = 0; j < NUMCOLS+1; ++j){
					
			//the whole epoch data is needed for each weight
			//sumsum (r_i^t-y_i^t)*hiddennode(1-hiddennode)*input,
			tempnum=0.0;
			for (int dataind = 0; dataind < trainLen; ++dataind){
			
				tempnum2=0.0;
				for (int i = 0; i < 10; ++i){
			
					tempnum2+=(r_it[dataind][i]-y_out[dataind][i])*weights_2nd_layer[h][i];
				}
		
				tempnum+=tempnum2*hiddennodes[dataind][h]*(1.0-hiddennodes[dataind][h])*data[dataind].pixels[j];
			}

			//update the weight, use momentum term (the last update)
			update=learningrate*tempnum+momentum*weights_1st_layer_mom[j][h];
			weights_1st_layer[j][h]+=update;
			//update momentum
			weights_1st_layer_mom[j][h]=update;
		}
	}   
	
	
	
	//Update the second layer weights
	for (int h = 0; h < num_hid_nodes+1; ++h){
		for (int i = 0; i < 10; ++i){
					
			//the whole epoch data is needed for each weight
			//sum_t (r_i^t-y_i^t)*hiddennode
			tempnum=0.0;
			for (int dataind = 0; dataind < trainLen; ++dataind){
			
				tempnum+=(r_it[dataind][i]-y_out[dataind][i])*hiddennodes[dataind][h];
			}

			//update the weight
			update=learningrate*tempnum+momentum*weights_2nd_layer_mom[h][i];
			weights_2nd_layer[h][i]+=update;
			weights_2nd_layer_mom[h][i]=update;
		}
	}   

}


int main()
{	

	//Program parameters	
	//const int dataLen=1000; //how many images are taken to learning and test
	//const int trainepochs=100;
	//double init_learningrate=0.002;
	//double weightscale=0.05; //initially the weights are ramdomized ans scaled by this
	//double momentum=0.8; // the momentum term for backpropagation
	
	
	int trainLen;
	int label;
	double error;
	double learningrate;
	double depoch;
	double tempnum=0.0;
	double successrate=0.0;
	double maxprob;
	string line;
	
	
	//70% of the data is used for training and 30% for testing
	trainLen=0.7*dataLen;
	
	//create struct for input data
	//Digitdata data[dataLen];

	//create the MLP network 
	MLP Mlp; //(60,data,dataLen,weightscale);

	//initialize the datafile for reading input
	ifstream traindata ("train.csv");
	getline (traindata,line); //read off the header
	//read input to the struct
	Readinput(traindata);
	
	
	//TRAINING:-------------------------
	//while error > wantederror and epoch <trainepochs

	//for(int epoch=0;epoch<trainepochs;++epoch){
	int epoch=0;
	error=10000000.0;
	while ((epoch < trainepochs) && (error > wantederror)) {
		
		if ( (epoch+1) % 1000 == 0 ){ //not used now
		
			//read new set of input to the struct data
			cout << "newdata" << endl;
			Readinput(traindata);
			
			//take a careful step into new dataset, small learningrate
			//and no momentum
			momentum=0.0;
			//learningrate=init_learningrate/10.0;
			} else {
			momentum=initmomentum;
			//learningrate=init_learningrate;
			}
			
		//calculate the output from the network over the tranLen (one epoch)
		Mlp.Epoch(0,trainLen); //(startindex, endindex)
	
		//calculate and print the error
		error=Mlp.Error_out(0,trainLen); //(startindex, endindex)

		//some time dependent scaling of the learningrate can be used for Backprobagation
		learningrate=init_learningrate; //*2.0*(1.0-(double)epoch/((double)trainepochs+1.0));
				
		//tune the weights by using backpropagation
		Mlp.Backpropagate (learningrate,momentum,trainLen);
		
		cout << "error was " << error << " at epoch "<< epoch << " with lrate "<<learningrate<< endl;
		++epoch;
	}
	
	
	
	//TESTING:------------------------
	//calculate the outputs
	Mlp.Epoch(trainLen,dataLen);
	
	
	//loop over the test set
	for(int i = trainLen; i<dataLen;++i){
	
		label=data[i].label;
	
		//here 0.8 is set to treshold for knowing the state
		if (y_out[i][label]>0.5)
			tempnum+=1.0; //this one was correct
	
		
		//so far correct
		successrate=tempnum/(1.0+((double)i-(double)trainLen));
		maxprob=*max_element(y_out[i], y_out[i] + 10);
		
		cout <<"data ind " << i << " y_out: "<< y_out[i][label]<< " max: " << maxprob << " correctpros: "<< successrate << " corlabel: " << data[i].label <<endl;
	
	}
	
	cout <<"Error in the trainset: " << Mlp.Error_out(trainLen,dataLen) << endl;

	traindata.close();
	return 0;
}